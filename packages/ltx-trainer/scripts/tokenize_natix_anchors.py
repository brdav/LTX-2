#!/usr/bin/env python3
"""Encode Natix anchor frames into LTX video VAE latents; each image is encoded on its own (a 1-frame video).

Anchors are Mapillary thumbnails (``<anchors-root>/anchors.parquet``, ``<anchors-root>/thumbs/thumbs_<SSS>.tar`` plus
``thumbs/index.parquet`` with the byte offset of every jpg). A 5-second clip has a variable number of anchors; they are
never stacked into a video, every image is a separate ``[C, 1, H, W]`` latent.

Anchor rows join the video manifest on ``chunk`` (the 60 s video directory = ``source_video_path`` without its last two
path parts; anchors spell ``United_States`` / ``Puerto_Rico`` with underscores) and ``segment`` (= ``sample_index``).
Only rows with ``keep=True`` whose clip is in the manifest are encoded.

Output mirrors ``tokenize_natix_osm.py``: one safetensors file per recording session (``<country>/<date>/<session>``,
several 60 s chunks), ``<output-dir>/latents/<country>/<date>__<session>.safetensors``, one tensor per distinct image
under the key ``<candidate_id>`` (an image shared by several clips of a session is stored once), plus index lines in
``<output-dir>/index/shard<N>.jsonl`` with one row per (clip, anchor): ``source_video_path``, ``start_frame``,
``view="anchor"``, ``file``, ``key``, ``candidate_id`` and the anchor columns ``role``, ``is_pano``, ``group_size``.

CPU work (tar read, jpg decode, resize/crop) runs in DataLoader workers, items are slices of at most ``--part-size``
images of one session; the GPU process encodes and saves a session once all its parts have arrived. Run one process per
GPU with its own ``--shard-index`` (sessions are sharded by md5). A rerun resumes (also after preemption): a session
whose file exists with the right ``n_images`` is skipped, its index rows are re-appended if the job died before writing
them (only the missing rows),
and stale ``.tmp`` files and a half-written last index line are cleaned up.
"""

import argparse
import hashlib
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file
from tokenize_natix_osm import _lower_priority, flat_relpath
from torch.utils.data import DataLoader, Dataset

from ltx_trainer import logger
from ltx_trainer.model_loader import load_video_vae_encoder, read_video_scale_factors
from ltx_trainer.process_videos import MediaDataset, _encode_video

ANCHOR_COLS = ["role", "is_pano", "group_size"]


def load_tables(anchors_root: Path, manifest: Path, country: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (clip rows, image rows). Clip rows: one per (clip, anchor); image rows: one per (session, image)."""
    cols = ["chunk", "segment", "candidate_id", "keep", *ANCHOR_COLS]
    a = pd.read_parquet(anchors_root / "anchors.parquet", columns=cols)
    a = a[a.keep].drop(columns="keep")
    a["chunk"] = a.chunk.str.replace(r"^(United|Puerto)_", r"\1 ", regex=True)
    if country:
        a = a[a.chunk.str.startswith(f"{country}/")]
    m = pd.read_parquet(manifest, columns=["source_video_path", "sample_index", "start_frame"])
    m["chunk"] = m.source_video_path.str.rsplit("/", n=2).str[0]
    n_rows = len(a)
    a = a.merge(m, left_on=["chunk", "segment"], right_on=["chunk", "sample_index"]).drop(columns=["sample_index"])
    logger.info(f"{len(a)} of {n_rows} keep anchors belong to a manifest clip")
    a["session"] = a.chunk.str.rsplit("/", n=1).str[0]

    thumbs = pd.read_parquet(
        anchors_root / "thumbs" / "index.parquet", columns=["candidate_id", "shard", "offset", "size"]
    )
    imgs = a.drop_duplicates(["session", "candidate_id"])[["session", "candidate_id"]].merge(thumbs, on="candidate_id")
    assert len(imgs) == len(a.drop_duplicates(["session", "candidate_id"])), "anchors without a thumbnail"
    imgs = imgs.sort_values(["session", "shard", "offset"], ignore_index=True)
    return a, imgs


class AnchorParts(Dataset):
    """One item = up to ``part_size`` images of one session -> bf16 tensor [n, 3, H, W] in [-1, 1]."""

    def __init__(self, parts: list[tuple[str, int, pd.DataFrame]], thumbs_dir: Path, height: int, width: int) -> None:
        self.parts, self.thumbs_dir, self.height, self.width = parts, thumbs_dir, height, width

    def __len__(self) -> int:
        return len(self.parts)

    def _image(self, data: bytes) -> torch.Tensor:
        arr = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
        t = torch.from_numpy(arr.copy()).permute(2, 0, 1).unsqueeze(0).float().div(255.0)
        t = MediaDataset._resize_and_crop(SimpleNamespace(reshape_mode="center"), t, self.height, self.width)
        t = t.reshape(3, self.height, self.width).clamp(0.0, 1.0)
        return ((t - 0.5) / 0.5).to(torch.bfloat16)

    def __getitem__(self, i: int) -> dict:
        torch.set_num_threads(1)
        session, part, rows = self.parts[i]
        res: dict = {"session": session, "part": part, "ids": rows.candidate_id.tolist(), "images": None}
        try:
            t = time.time()
            images = []
            for shard, g in rows.groupby("shard", sort=False):
                with open(self.thumbs_dir / f"thumbs_{shard}.tar", "rb") as f:
                    for r in g.itertuples():
                        f.seek(r.offset)
                        images.append(self._image(f.read(r.size)))
            # groupby(sort=False) keeps the order of first appearance and rows are sorted by shard, so order is kept
            res["images"] = torch.stack(images)
            res["t_decode"] = time.time() - t
        except Exception as e:
            res["error"] = str(e)[:500]
        return res


def _save_session(latents: torch.Tensor, ids: list[str], lat: dict, out: Path) -> None:
    """Write one session (``latents``: [n, C, 1, H, W], keyed by candidate_id) to one safetensors file."""
    out.parent.mkdir(parents=True, exist_ok=True)
    assert len(set(ids)) == len(ids), f"duplicate keys in {out}"
    tmp = out.with_suffix(f"{out.suffix}.tmp.{os.getpid()}")
    save_file(
        {k: latents[n].contiguous() for n, k in enumerate(ids)},
        str(tmp),
        metadata={k: str(lat[k]) for k in ("num_frames", "height", "width")} | {"n_images": str(len(ids))},
    )
    tmp.replace(out)


def _index_rows(clips: pd.DataFrame, rel: str) -> list[dict]:
    keep = ["source_video_path", "start_frame", "candidate_id", *ANCHOR_COLS]
    rows = clips[keep].to_dict("records")
    return [
        {
            "source_video_path": r["source_video_path"],
            "start_frame": int(r["start_frame"]),
            "view": "anchor",
            "file": rel,
            "key": r["candidate_id"],
            "candidate_id": r["candidate_id"],
            "role": r["role"],
            "is_pano": bool(r["is_pano"]),
            "group_size": int(r["group_size"]),
        }
        for r in rows
    ]


def _up_to_date(path: Path, n_images: int) -> bool:
    if not path.is_file():
        return False
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return int((f.metadata() or {}).get("n_images", -1)) == n_images


def _repair_index(index_file: Path) -> dict[str, set[tuple]]:
    """Cut a half-written last line (job killed mid-append) and return the (clip, key) rows indexed per session file."""
    indexed: dict[str, set[tuple]] = {}
    if not index_file.is_file():
        return indexed
    data = index_file.read_bytes()
    if data and not data.endswith(b"\n"):
        data = data[: data.rfind(b"\n") + 1]
        index_file.write_bytes(data)
    for line in data.decode().splitlines():
        r = json.loads(line)
        indexed.setdefault(r["file"], set()).add((r["source_video_path"], r["start_frame"], r["key"]))
    return indexed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors-root", required=True, help="Directory with anchors.parquet and thumbs/")
    ap.add_argument("--manifest", required=True, help="samples_24fps.parquet")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--vae-path", required=True)
    ap.add_argument("--resolution", default="768x512", help="WIDTHxHEIGHT (multiples of the VAE spatial factor)")
    ap.add_argument("--num-sessions", type=int, default=None, help="Only the first N sessions (testing)")
    ap.add_argument("--country", default=None, help="Only this country, e.g. Switzerland or 'United States'")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=14, help="CPU worker processes (tar read/decode/resize)")
    ap.add_argument("--prefetch", type=int, default=2, help="Parts prefetched per worker")
    ap.add_argument("--part-size", type=int, default=64, help="Images per work item (bounds worker memory)")
    ap.add_argument("--encode-batch", type=int, default=16, help="Images per VAE forward")
    ap.add_argument("--vae-tiling", action="store_true")
    args = ap.parse_args()

    width, height = (int(v) for v in args.resolution.lower().split("x"))
    anchors_root, out_dir = Path(args.anchors_root), Path(args.output_dir)
    device = torch.device("cuda")

    clips, imgs = load_tables(anchors_root, Path(args.manifest), args.country)
    sessions = sorted(imgs.session.unique())
    if args.num_sessions:
        sessions = sessions[: args.num_sessions]
    sessions = [
        s for s in sessions if int(hashlib.md5(s.encode()).hexdigest(), 16) % args.num_shards == args.shard_index
    ]
    sess_imgs = dict(iter(imgs[imgs.session.isin(set(sessions))].groupby("session")))
    sess_clips = dict(iter(clips[clips.session.isin(set(sessions))].groupby("session")))

    rel = {s: flat_relpath(f"{s}.zip") for s in sessions}
    out_dir.mkdir(parents=True, exist_ok=True)
    index_file = out_dir / "index" / f"shard{args.shard_index}.jsonl"
    index_file.parent.mkdir(parents=True, exist_ok=True)
    indexed = _repair_index(index_file)
    todo, skipped, backfilled = [], 0, 0
    for s in sessions:
        path = out_dir / rel[s]
        for stale in path.parent.glob(f"{path.name}.tmp.*"):  # left behind by a killed run
            stale.unlink(missing_ok=True)
        if _up_to_date(path, len(sess_imgs[s])):
            skipped += len(sess_imgs[s])
            # killed between saving the file and indexing it (or midway through): add the rows that are missing
            have = indexed.get(str(rel[s]), set())
            rows = [
                r
                for r in _index_rows(sess_clips[s], str(rel[s]))
                if (r["source_video_path"], r["start_frame"], r["key"]) not in have
            ]
            if rows:
                with index_file.open("a") as f:
                    f.writelines(json.dumps(r) + "\n" for r in rows)
                backfilled += 1
        else:
            todo.append(s)
    parts = []
    for s in todo:
        g = sess_imgs[s]
        n_parts = -(-len(g) // args.part_size)
        parts += [(s, p, g.iloc[p * args.part_size : (p + 1) * args.part_size]) for p in range(n_parts)]
    n_parts_of = {s: -(-len(sess_imgs[s]) // args.part_size) for s in todo}
    logger.info(
        f"Shard {args.shard_index}/{args.num_shards}: {len(sessions)} sessions ({len(todo)} to do, {skipped} images "
        f"already done, {backfilled} re-indexed), {len(parts)} parts, resolution {width}x{height}"
    )

    scale = read_video_scale_factors(args.vae_path)
    assert width % scale.width == 0 and height % scale.height == 0, f"{width}x{height} not multiple of VAE factors"
    vae = load_video_vae_encoder(args.vae_path, device=device, dtype=torch.bfloat16)

    fail_log = out_dir / f"failures_shard{args.shard_index}.jsonl"
    loader = DataLoader(
        AnchorParts(parts, anchors_root / "thumbs", height, width),
        batch_size=None,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch if args.num_workers else None,
        worker_init_fn=_lower_priority,
    )

    def save_and_index(latents: torch.Tensor, ids: list[str], lat: dict, session: str) -> list[dict]:
        _save_session(latents, ids, lat, out_dir / rel[session])
        return _index_rows(sess_clips[session], str(rel[session]))

    saver = ThreadPoolExecutor(max_workers=4)
    pending: list = []
    acc: dict[str, tuple[list, list]] = {}  # session -> (ids, latent chunks) of the parts received so far
    failed: set[str] = set()
    done = 0
    t0 = time.time()
    t_enc = t_wait = t_decode = 0.0
    t_wait_start = time.time()
    for pi, item in enumerate(loader):
        t_wait += time.time() - t_wait_start
        s = item["session"]
        if "error" in item:
            logger.error(f"{s}: {item['error']}")
            with fail_log.open("a") as f:
                f.write(json.dumps({"session": s, "error": item["error"]}) + "\n")
            failed.add(s)
        elif s not in failed:
            t_decode += item["t_decode"]
            t = time.time()
            encoded = []
            for b in range(0, len(item["images"]), args.encode_batch):
                x = item["images"][b : b + args.encode_batch].to(device, non_blocking=True).unsqueeze(2)
                with torch.inference_mode():
                    lat = _encode_video(vae, x, use_tiling=args.vae_tiling, scale_factors=scale)
                encoded.append(lat["latents"].cpu())
            t_enc += time.time() - t
            ids, chunks = acc.setdefault(s, ([], []))
            ids += item["ids"]
            chunks.append(torch.cat(encoded))
            done += len(item["ids"])
            if item["part"] + 1 == n_parts_of[s]:
                ids, chunks = acc.pop(s)
                pending.append(saver.submit(save_and_index, torch.cat(chunks), ids, lat, s))
        # Snapshot once: a save finishing between two done() checks would otherwise lose its index rows.
        finished = [f for f in pending if f.done()]
        for fut in finished:
            with index_file.open("a") as f:
                f.writelines(json.dumps(r) + "\n" for r in fut.result())
        pending = [f for f in pending if f not in finished]
        if (pi + 1) % 100 == 0 or pi + 1 == len(parts):
            el = time.time() - t0
            logger.info(
                f"{pi + 1}/{len(parts)} parts, {done} latents ({done / max(el, 1e-9):.1f}/s) | "
                f"gpu-side encode {t_enc:.0f}s, waiting on workers {t_wait:.0f}s | worker-time decode {t_decode:.0f}s"
            )
        t_wait_start = time.time()

    for fut in pending:
        with index_file.open("a") as f:
            f.writelines(json.dumps(r) + "\n" for r in fut.result())
    saver.shutdown()
    logger.info(
        f"Done: {done} encoded, {skipped} images already existed, {len(failed)} failed sessions, "
        f"{time.time() - t0:.0f}s"
    )


if __name__ == "__main__":
    main()
