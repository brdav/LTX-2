#!/usr/bin/env python3
"""Encode Natix satellite crops into LTX video VAE latents (a still image = a 1-frame video).

The crops live in one zip per region and month, ``<sat-root>/<region>/<YYYY-MM>.zip`` (``region`` can be nested, e.g.
``United States/Alabama``), with members ``<date>/<session>/<HH-MM-SS>_sNNNN.webp``, one per 5-second sample, 768x768.
Samples without imagery are simply absent. Each zip carries its own ``index.jsonl`` (``image``, ``osm_image``,
``sample_id`` = ``<source_video_path>#f<start_frame>``, ...), so the satellite latents join the video clips like the OSM
maps; a zip without it (or not yet closed) is treated as still being generated.

Output mirrors ``tokenize_natix_osm.py``: one safetensors file per recording session (not per zip, which can hold
100k images), ``<output-dir>/latents/<region>/<date>__<session>.safetensors``, one ``[C, 1, H, W]`` tensor per image
under the key ``<HH-MM-SS>_sNNNN/satellite``, plus index lines in ``<output-dir>/index/shard<N>.jsonl``.
CPU work (zip read, webp decode, resize/crop) runs in DataLoader workers, one session per item; run one process per
GPU with its own ``--shard-index`` (sessions are sharded by the md5 of the OSM zip path, as in the OSM script).

Some zips may still be generated: an incomplete zip is skipped with a warning, and a session whose file exists is
skipped only if it holds as many images as the zip currently has for it, so a rerun picks up late zips and zips that
grew (a re-encoded session appends duplicate index lines; ``PackedLatentDataset`` dedupes them).
"""

import argparse
import hashlib
import io
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from PIL import Image
from safetensors import safe_open
from tokenize_natix_osm import _lower_priority, _save_zip, flat_relpath
from torch.utils.data import DataLoader, Dataset

from ltx_trainer import logger
from ltx_trainer.model_loader import load_video_vae_encoder, read_video_scale_factors
from ltx_trainer.process_videos import MediaDataset, _encode_video

VIEWS = ("satellite",)


def load_items(sat_root: Path, country: str | None) -> pd.DataFrame:
    """One row per satellite image, from the ``index.jsonl`` inside every zip, sorted by session and start frame."""
    rows, unreadable = [], []
    for zip_path in sorted(sat_root.glob("**/*.zip")):
        sat_zip = str(zip_path.relative_to(sat_root))
        region = str(Path(sat_zip).parent)
        if country and not (region == country or region.startswith(f"{country}/")):
            continue
        try:
            with zipfile.ZipFile(zip_path) as zf:
                index = [json.loads(line) for line in zf.read("index.jsonl").decode().splitlines()]
        except (zipfile.BadZipFile, KeyError, OSError) as e:  # still being written: no central directory / no index
            unreadable.append(sat_zip)
            logger.warning(f"Skipping {sat_zip} (not complete yet: {type(e).__name__}: {e})")
            continue
        for r in index:
            osm_zip, member = r["osm_image"].split("::")
            source_video_path, start_frame = r["sample_id"].split("#f")
            rows.append((sat_zip, r["image"], osm_zip, member, source_video_path, int(start_frame)))
    logger.info(f"{len(rows)} satellite images indexed, {len(unreadable)} zips skipped")
    cols = ["sat_zip", "name", "osm_zip", "member", "source_video_path", "start_frame"]
    return pd.DataFrame(rows, columns=cols).sort_values(["osm_zip", "start_frame"])


class SatelliteSessions(Dataset):
    """One item = one session -> bf16 tensor [n_images, 1, 3, H, W] in [-1, 1] (preprocessing as ``MediaDataset``)."""

    def __init__(self, zips: list[str], groups: dict, sat_root: Path, out_dir: Path, height: int, width: int) -> None:
        self.zips, self.groups, self.sat_root, self.out_dir = zips, groups, sat_root, out_dir
        self.height, self.width = height, width
        self._open: dict[str, zipfile.ZipFile] = {}  # per worker; a big zip's directory is only parsed once

    def __len__(self) -> int:
        return len(self.zips)

    def _zip(self, sat_zip: str) -> zipfile.ZipFile:
        if sat_zip not in self._open:
            while len(self._open) >= 2:  # sessions are sorted, so consecutive items mostly share a zip
                self._open.pop(next(iter(self._open))).close()
            self._open[sat_zip] = zipfile.ZipFile(self.sat_root / sat_zip)
        return self._open[sat_zip]

    def _image(self, data: bytes) -> torch.Tensor:
        arr = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
        t = torch.from_numpy(arr.copy()).permute(2, 0, 1).unsqueeze(0).float().div(255.0)
        t = MediaDataset._resize_and_crop(SimpleNamespace(reshape_mode="center"), t, self.height, self.width)
        t = t.reshape(3, self.height, self.width).clamp(0.0, 1.0)
        return ((t - 0.5) / 0.5).to(torch.bfloat16)

    def _up_to_date(self, osm_zip: str, n_images: int) -> bool:
        out = self.out_dir / flat_relpath(osm_zip)
        if not out.is_file():
            return False
        with safe_open(str(out), framework="pt", device="cpu") as f:
            return int((f.metadata() or {}).get("n_images", -1)) == n_images

    def __getitem__(self, i: int) -> dict:
        torch.set_num_threads(1)
        osm_zip = self.zips[i]
        rows = list(self.groups[osm_zip].itertuples())
        res: dict = {"zip": osm_zip, "rows": None, "images": None, "t_decode": 0.0, "skipped": 0}
        if self._up_to_date(osm_zip, len(rows)):
            res["skipped"] = len(rows)
            return res
        try:
            t = time.time()
            images = torch.stack([self._image(self._zip(r.sat_zip).read(r.name)) for r in rows])
            res["t_decode"] = time.time() - t
            res["rows"] = [(r.source_video_path, int(r.start_frame), r.member) for r in rows]
            res["images"] = images.unsqueeze(1)
        except Exception as e:
            res["error"] = str(e)[:500]
        return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sat-root", required=True, help="Directory holding <region>/<YYYY-MM>.zip")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--vae-path", required=True)
    ap.add_argument("--resolution", default="768x768", help="WIDTHxHEIGHT (multiples of the VAE spatial factor)")
    ap.add_argument("--num-sessions", type=int, default=None, help="Only the first N sessions (testing)")
    ap.add_argument("--max-samples", type=int, default=None, help="Stop after N images in total (pilot)")
    ap.add_argument("--country", default=None, help="Only this region path, e.g. Switzerland or 'United States'")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=14, help="CPU worker processes (zip read/decode/resize)")
    ap.add_argument("--prefetch", type=int, default=2, help="Sessions prefetched per worker")
    ap.add_argument("--encode-batch", type=int, default=16, help="Images per VAE forward")
    ap.add_argument("--vae-tiling", action="store_true")
    args = ap.parse_args()

    width, height = (int(v) for v in args.resolution.lower().split("x"))
    sat_root, out_dir = Path(args.sat_root), Path(args.output_dir)
    device = torch.device("cuda")

    df = load_items(sat_root, args.country)
    zips = sorted(df.osm_zip.unique())
    if args.num_sessions:
        zips = zips[: args.num_sessions]
    zips = [z for z in zips if int(hashlib.md5(z.encode()).hexdigest(), 16) % args.num_shards == args.shard_index]
    groups = {z: g for z, g in df[df.osm_zip.isin(set(zips))].groupby("osm_zip")}
    logger.info(f"Shard {args.shard_index}/{args.num_shards}: {len(zips)} sessions, resolution {width}x{height}")

    scale = read_video_scale_factors(args.vae_path)
    assert width % scale.width == 0 and height % scale.height == 0, f"{width}x{height} not multiple of VAE factors"
    vae = load_video_vae_encoder(args.vae_path, device=device, dtype=torch.bfloat16)

    out_dir.mkdir(parents=True, exist_ok=True)
    fail_log = out_dir / f"failures_shard{args.shard_index}.jsonl"
    loader = DataLoader(
        SatelliteSessions(zips, groups, sat_root, out_dir, height, width),
        batch_size=None,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch if args.num_workers else None,
        worker_init_fn=_lower_priority,
    )

    index_file = out_dir / "index" / f"shard{args.shard_index}.jsonl"
    index_file.parent.mkdir(parents=True, exist_ok=True)

    def write_index(rows: list[dict]) -> None:
        with index_file.open("a") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)

    saver = ThreadPoolExecutor(max_workers=4)
    pending_saves: list = []
    done = skipped = 0
    t0 = time.time()
    t_decode = t_enc = t_wait = 0.0
    t_wait_start = time.time()
    for si, item in enumerate(loader):
        t_wait += time.time() - t_wait_start
        skipped += item["skipped"]
        t_decode += item["t_decode"]
        osm_zip = item["zip"]
        if "error" in item:
            logger.error(f"{osm_zip}: {item['error']}")
            with fail_log.open("a") as f:
                f.write(json.dumps({"zip": osm_zip, "error": item["error"]}) + "\n")
        elif item["images"] is not None:
            rows, images = item["rows"], item["images"]
            if args.max_samples:
                rows, images = rows[: args.max_samples - done], images[: args.max_samples - done]
            flat = images.flatten(0, 1)  # [n, 3, H, W]
            t = time.time()
            encoded = []
            for b in range(0, len(flat), args.encode_batch):
                x = flat[b : b + args.encode_batch].to(device, non_blocking=True).unsqueeze(2)  # [B, 3, 1, H, W]
                with torch.inference_mode():
                    lat = _encode_video(vae, x, use_tiling=args.vae_tiling, scale_factors=scale)
                encoded.append(lat["latents"].cpu())
            t_enc += time.time() - t
            latents = torch.cat(encoded).unflatten(0, (len(rows), len(VIEWS)))  # [n, 1, C, 1, H', W']
            done += len(rows)
            pending_saves.append(
                saver.submit(_save_zip, latents, rows, lat, out_dir / flat_relpath(osm_zip), out_dir, VIEWS)
            )
        # Snapshot once: a save finishing between two done() checks would otherwise lose its index rows.
        finished = [f for f in pending_saves if f.done()]
        for fut in finished:
            write_index(fut.result())
        pending_saves = [f for f in pending_saves if f not in finished]
        if (si + 1) % 20 == 0 or si + 1 == len(zips):
            el = time.time() - t0
            logger.info(
                f"{si + 1}/{len(zips)} sessions, {done} latents ({done / max(el, 1e-9):.1f}/s) | "
                f"gpu-side encode {t_enc:.0f}s, waiting on workers {t_wait:.0f}s | worker-time decode {t_decode:.0f}s"
            )
        if args.max_samples and done >= args.max_samples:
            break
        t_wait_start = time.time()

    for fut in pending_saves:
        write_index(fut.result())
    saver.shutdown()
    logger.info(f"Done: {done} encoded, {skipped} images already existed, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
