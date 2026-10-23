#!/usr/bin/env python3
"""Encode Natix OSM BEV map images into LTX video VAE latents (a still image = a 1-frame video).

The maps live in one zip per recording session, ``<osm-root>/<variant>/<country>/<date>/<session>.zip``, with one webp
per 5-second sample. Two variants (``maps_map_matched`` and ``maps_predicted``) hold the same images drawn from a
map-matched and a predicted path. The index (``<variant>/index.shard*of*.jsonl``) maps ``sample_id``
(``<source_video_path>#f<start_frame>``) to ``<zip>::<member>``.

CPU work (zip read, webp decode, resize/crop) runs in DataLoader worker processes, one zip per work item; the GPU
process only encodes. Run one process per GPU, each with its own ``--shard-index``.

Latents are packed one safetensors file per zip, ``<output-dir>/latents/<country>/<date>__<session>.safetensors``, with
one ``[C, 1, H, W]`` tensor per image under the key ``<zip member stem>/<view>`` (e.g. ``21-32-44_s0000/map_matched``;
a zip holds a whole session, so ``start_frame`` alone is not unique within a file). A map joins its video clip on
``(source_video_path, start_frame)`` through the index. Each shard also appends one JSON line per image to
``<output-dir>/index/shard<N>.jsonl``. A zip whose file exists is skipped, so a rerun resumes.
"""

import argparse
import hashlib
import io
import json
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from PIL import Image
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset

from ltx_trainer import logger
from ltx_trainer.model_loader import load_video_vae_encoder, read_video_scale_factors
from ltx_trainer.process_videos import MediaDataset, _encode_video

# view name -> variant directory under --osm-root
VARIANTS = {"map_matched": "maps_map_matched", "predicted": "maps_predicted"}


def flat_relpath(zip_rel: str) -> Path:
    """``<country>/<a>/<b>.zip`` -> ``latents/<country>/<a>__<b>.safetensors`` (one directory per country)."""
    parts = Path(zip_rel).with_suffix("").parts
    return Path("latents") / parts[0] / ("__".join(parts[1:]) + ".safetensors")


def _lower_priority(_worker_id: int) -> None:
    """Keep CPU workers from starving the GPU-feeding main process."""
    os.nice(10)


def load_index(osm_root: Path) -> pd.DataFrame:
    """One row per image present in *all* variants: columns zip, member, source_video_path, start_frame."""
    per_view = []
    for view, variant in VARIANTS.items():
        files = sorted((osm_root / variant).glob("index.shard*of*.jsonl"))
        df = pd.concat([pd.read_json(f, lines=True)[["image", "sample_id", "start_frame"]] for f in files])
        per_view.append(df.drop_duplicates("image").set_index("image").add_suffix(f"_{view}"))
    df = pd.concat(per_view, axis=1, join="inner").reset_index()
    first = next(iter(VARIANTS))
    df = df.rename(columns={f"sample_id_{first}": "sample_id", f"start_frame_{first}": "start_frame"})
    df = df[["image", "sample_id", "start_frame"]]
    df["zip"] = df.image.str.split("::").str[0]
    df["member"] = df.image.str.split("::").str[1]
    df["source_video_path"] = df.sample_id.str.split("#").str[0]
    return df.drop(columns=["image", "sample_id"]).sort_values(["zip", "start_frame"])


class MapZips(Dataset):
    """One item = one zip -> bf16 tensor [n_images, n_views, 3, H, W] in [-1, 1] (preprocessing as ``MediaDataset``)."""

    def __init__(self, zips: list[str], groups: dict, osm_root: Path, out_dir: Path, height: int, width: int) -> None:
        self.zips, self.groups, self.osm_root, self.out_dir = zips, groups, osm_root, out_dir
        self.height, self.width = height, width

    def __len__(self) -> int:
        return len(self.zips)

    def _image(self, data: bytes) -> torch.Tensor:
        arr = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
        t = torch.from_numpy(arr.copy()).permute(2, 0, 1).unsqueeze(0).float().div(255.0)
        t = MediaDataset._resize_and_crop(SimpleNamespace(reshape_mode="center"), t, self.height, self.width)  # noqa: SLF001
        t = t.reshape(3, self.height, self.width).clamp(0.0, 1.0)
        return ((t - 0.5) / 0.5).to(torch.bfloat16)

    def __getitem__(self, i: int) -> dict:
        torch.set_num_threads(1)
        zip_rel = self.zips[i]
        res: dict = {"zip": zip_rel, "rows": None, "images": None, "t_decode": 0.0, "skipped": 0}
        if (self.out_dir / flat_relpath(zip_rel)).is_file():
            res["skipped"] = len(self.groups[zip_rel])
            return res
        rows = list(self.groups[zip_rel].itertuples())
        try:
            t = time.time()
            per_view = []
            for variant in VARIANTS.values():
                with zipfile.ZipFile(self.osm_root / variant / zip_rel) as zf:
                    per_view.append(torch.stack([self._image(zf.read(r.member)) for r in rows]))
            res["t_decode"] = time.time() - t
            res["rows"] = [(r.source_video_path, int(r.start_frame), r.member) for r in rows]
            res["images"] = torch.stack(per_view, dim=1)
        except Exception as e:  # noqa: BLE001
            res["error"] = str(e)[:500]
        return res


def _save_zip(
    latents: torch.Tensor, rows: list, lat: dict, out: Path, root: Path, views: tuple[str, ...] = tuple(VARIANTS)
) -> list[dict]:
    """Write all images of one zip (``latents``: [n, len(views), C, 1, H, W]) to one safetensors file.

    Returns the index rows (one per image and view) for the caller to append once the file is in place.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    entries = [
        {
            "source_video_path": vp,
            "start_frame": s,
            "view": view,
            "key": f"{Path(member).stem}/{view}",
            "n": n,
            "v": v,
        }
        for n, (vp, s, member) in enumerate(rows)
        for v, view in enumerate(views)
    ]
    assert len({e["key"] for e in entries}) == len(entries), f"duplicate keys in {out}"
    tmp = out.with_suffix(f"{out.suffix}.tmp.{os.getpid()}")
    save_file(
        {e["key"]: latents[e["n"], e["v"]].contiguous() for e in entries},
        str(tmp),
        metadata={k: str(lat[k]) for k in ("num_frames", "height", "width")} | {"n_images": str(len(rows))},
    )
    tmp.replace(out)
    rel = str(out.relative_to(root))
    return [
        {k: e[k] for k in ("source_video_path", "start_frame", "view")} | {"file": rel, "key": e["key"]}
        for e in entries
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--osm-root", required=True, help="Directory holding maps_map_matched/ and maps_predicted/")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--vae-path", required=True)
    ap.add_argument("--resolution", default="768x768", help="WIDTHxHEIGHT (multiples of the VAE spatial factor)")
    ap.add_argument("--num-zips", type=int, default=None, help="Only the first N zips (testing)")
    ap.add_argument("--max-samples", type=int, default=None, help="Stop after N images in total (pilot)")
    ap.add_argument("--country", default=None, help="Only zips of this country (pilot)")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=14, help="CPU worker processes (zip read/decode/resize)")
    ap.add_argument("--prefetch", type=int, default=2, help="Zips prefetched per worker")
    ap.add_argument("--encode-batch", type=int, default=16, help="Images per VAE forward")
    ap.add_argument("--vae-tiling", action="store_true")
    args = ap.parse_args()

    width, height = (int(v) for v in args.resolution.lower().split("x"))
    osm_root, out_dir = Path(args.osm_root), Path(args.output_dir)
    device = torch.device("cuda")

    df = load_index(osm_root)
    if args.country:
        df = df[df.zip.str.startswith(f"{args.country}/")]
    zips = sorted(df.zip.unique())
    if args.num_zips:
        zips = zips[: args.num_zips]
    zips = [z for z in zips if int(hashlib.md5(z.encode()).hexdigest(), 16) % args.num_shards == args.shard_index]
    groups = {z: g for z, g in df[df.zip.isin(set(zips))].groupby("zip")}
    logger.info(f"Shard {args.shard_index}/{args.num_shards}: {len(zips)} zips, resolution {width}x{height}")

    scale = read_video_scale_factors(args.vae_path)
    assert width % scale.width == 0 and height % scale.height == 0, f"{width}x{height} not multiple of VAE factors"
    vae = load_video_vae_encoder(args.vae_path, device=device, dtype=torch.bfloat16)

    out_dir.mkdir(parents=True, exist_ok=True)
    fail_log = out_dir / f"failures_shard{args.shard_index}.jsonl"
    loader = DataLoader(
        MapZips(zips, groups, osm_root, out_dir, height, width),
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
    for zi, item in enumerate(loader):
        t_wait += time.time() - t_wait_start
        skipped += item["skipped"]
        t_decode += item["t_decode"]
        zip_rel = item["zip"]
        if "error" in item:
            logger.error(f"{zip_rel}: {item['error']}")
            with fail_log.open("a") as f:
                f.write(json.dumps({"zip": zip_rel, "error": item["error"]}) + "\n")
        elif item["images"] is not None:
            rows, images = item["rows"], item["images"]
            n_views = images.shape[1]
            if args.max_samples:
                left = args.max_samples - done // n_views
                rows, images = rows[:left], images[:left]
            flat = images.flatten(0, 1)  # [n * n_views, 3, H, W]
            t = time.time()
            encoded = []
            for b in range(0, len(flat), args.encode_batch):
                x = flat[b : b + args.encode_batch].to(device, non_blocking=True).unsqueeze(2)  # [B, 3, 1, H, W]
                with torch.inference_mode():
                    lat = _encode_video(vae, x, use_tiling=args.vae_tiling, scale_factors=scale)
                encoded.append(lat["latents"].cpu())
            t_enc += time.time() - t
            latents = torch.cat(encoded).unflatten(0, (len(rows), n_views))  # [n, n_views, C, 1, H', W']
            done += len(rows) * n_views
            pending_saves.append(saver.submit(_save_zip, latents, rows, lat, out_dir / flat_relpath(zip_rel), out_dir))
        # Snapshot once: a save finishing between two done() checks would otherwise lose its index rows.
        finished = [f for f in pending_saves if f.done()]
        for fut in finished:
            write_index(fut.result())
        pending_saves = [f for f in pending_saves if f not in finished]
        if (zi + 1) % 20 == 0 or zi + 1 == len(zips):
            el = time.time() - t0
            logger.info(
                f"{zi + 1}/{len(zips)} zips, {done} latents ({done / max(el, 1e-9):.1f}/s) | "
                f"gpu-side encode {t_enc:.0f}s, waiting on workers {t_wait:.0f}s | worker-time decode {t_decode:.0f}s"
            )
        if args.max_samples and done >= args.max_samples * len(VARIANTS):
            break
        t_wait_start = time.time()

    for fut in pending_saves:
        write_index(fut.result())
    saver.shutdown()
    logger.info(f"Done: {done} encoded, {skipped} images already existed, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
