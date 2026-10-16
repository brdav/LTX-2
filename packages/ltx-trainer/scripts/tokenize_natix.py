#!/usr/bin/env python3
"""Encode Natix manifest clips into LTX video VAE latents.

For every source video referenced by the manifest:
  1. resample to 24 FPS with the canonical ffmpeg command,
  2. slice each manifest row's [start_frame, end_frame) (exactly 121 frames),
  3. resize + center-crop to the target resolution and encode with the video VAE.

CPU work (ffmpeg, decoding, resize/crop) runs in DataLoader worker processes, one video per work item; the GPU
process only normalizes and encodes. Run one process per GPU, each with its own ``--shard-index``.

Latents are written as ``<output-dir>/latents/<source_video_path without .mp4>/f<start_frame>.pt`` in the same
format as ``process_videos.py``. Existing outputs are skipped, so a rerun resumes.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import av
import imageio_ffmpeg
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ltx_trainer import logger
from ltx_trainer.model_loader import load_video_vae_encoder, read_video_scale_factors
from ltx_trainer.process_videos import MediaDataset, _atomic_save, _encode_video

EXPECTED_FRAMES = 121
OUTPUT_FPS = 24.0


def resample(src: Path, dst: Path, ffmpeg: str) -> None:
    """Canonical resampling command (do not change flags)."""
    cmd = [ffmpeg, "-y", "-nostdin", "-i", str(src), "-vf", "fps=24", "-vsync", "cfr", "-an", str(dst)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def _lower_priority(_worker_id: int) -> None:
    """Keep CPU workers (and their ffmpeg children) from starving the GPU-feeding main process."""
    os.nice(10)


def _save(latents: torch.Tensor, lat: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_save(
        {
            "latents": latents.clone(),  # detach from the batch storage (torch.save writes the whole storage)
            "num_frames": lat["num_frames"],
            "height": lat["height"],
            "width": lat["width"],
            "fps": OUTPUT_FPS,
        },
        out,
    )


def output_file(output_dir: Path, source_video_path: str, start_frame: int) -> Path:
    return output_dir / "latents" / Path(source_video_path).with_suffix("") / f"f{start_frame:06d}.pt"


class VideoClips(Dataset):
    """One item = one source video -> bf16 tensor [n_clips, 121, 3, H, W] of resized/cropped, [-1, 1]-normalized clips.

    The preprocessing is the one in ``MediaDataset`` (float32 resize/crop, clamp, Normalize), done frame by frame on the
    CPU. ``_encode_video`` casts the VAE input to bf16 anyway, so casting here is lossless w.r.t. the encoder input.
    """

    def __init__(self, videos: list[str], groups: dict, args: argparse.Namespace, height: int, width: int) -> None:
        self.videos, self.groups, self.args, self.height, self.width = videos, groups, args, height, width
        self.out_dir, self.data_root = Path(args.output_dir), Path(args.data_root)
        self.ffmpeg = args.ffmpeg or shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()

    def __len__(self) -> int:
        return len(self.videos)

    def _frame(self, arr: np.ndarray) -> torch.Tensor:
        """Same ops as MediaDataset (resize/crop, clamp, normalize) for one RGB frame; returns bf16 [3, H, W]."""
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().div(255.0)
        t = MediaDataset._resize_and_crop(SimpleNamespace(reshape_mode="center"), t, self.height, self.width)  # noqa: SLF001
        t = t.reshape(3, self.height, self.width).clamp(0.0, 1.0)
        return ((t - 0.5) / 0.5).to(torch.bfloat16)

    def __getitem__(self, i: int) -> dict:
        torch.set_num_threads(1)
        vpath = self.videos[i]
        res: dict = {"vpath": vpath, "starts": [], "clips": None, "t_ffmpeg": 0.0, "t_decode": 0.0, "skipped": 0}
        rows = [r for r in self.groups[vpath].itertuples() if not output_file(self.out_dir, vpath, r.start_frame).is_file()]
        res["skipped"] = len(self.groups[vpath]) - len(rows)
        if not rows:
            return res
        try:
            with tempfile.TemporaryDirectory(dir=self.args.tmp_dir) as td:
                resampled = Path(td) / "resampled.mp4"
                t = time.time()
                resample(self.data_root / vpath, resampled, self.ffmpeg)
                res["t_ffmpeg"] = time.time() - t

                t = time.time()
                pending = {r.start_frame: r for r in rows}
                starts = sorted(pending)
                buf: dict[int, torch.Tensor] = {}
                clips, done_starts, n_frames = [], [], 0
                with av.open(str(resampled)) as c:
                    for idx, fr in enumerate(c.decode(video=0)):
                        n_frames = idx + 1
                        # Only frames that belong to a still-pending clip need resizing.
                        if starts and idx >= starts[0]:
                            buf[idx] = self._frame(fr.to_ndarray(format="rgb24"))
                        while starts and idx + 1 >= pending[starts[0]].end_frame:
                            r = pending[starts.pop(0)]
                            clip = torch.stack([buf[k] for k in range(r.start_frame, r.end_frame)])
                            assert clip.shape[0] == EXPECTED_FRAMES == r.num_frames
                            clips.append(clip)
                            done_starts.append(int(r.start_frame))
                            keep_from = starts[0] if starts else n_frames
                            for k in [k for k in buf if k < keep_from]:
                                del buf[k]
                if starts:
                    raise RuntimeError(f"video has {n_frames} frames, clips {starts} out of range")
                expected_total = int(rows[0].source_num_resampled_frames)
                if n_frames != expected_total:
                    res["warning"] = f"{n_frames} resampled frames, manifest says {expected_total}"
                res["t_decode"] = time.time() - t
                res["starts"], res["clips"] = done_starts, torch.stack(clips)
        except Exception as e:  # noqa: BLE001
            res["error"] = str(e)[:500]
        return res


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-root", required=True, help="Directory that source_video_path is relative to")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--vae-path", required=True)
    ap.add_argument("--resolution", default="768x512", help="WIDTHxHEIGHT (multiples of the VAE spatial factor)")
    ap.add_argument("--num-videos", type=int, default=None, help="Only the first N source videos (testing)")
    ap.add_argument("--max-samples", type=int, default=None, help="Stop after N clips in total (pilot)")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=24, help="CPU worker processes (ffmpeg/decode/resize)")
    ap.add_argument("--prefetch", type=int, default=2, help="Videos prefetched per worker")
    ap.add_argument("--encode-batch", type=int, default=1, help="Clips per VAE forward (1 = as process_videos.py)")
    ap.add_argument("--tmp-dir", default=None, help="Node-local scratch for resampled mp4s (default: $TMPDIR)")
    ap.add_argument("--ffmpeg", default=None, help="ffmpeg binary (default: system ffmpeg, else bundled imageio-ffmpeg)")
    ap.add_argument("--vae-tiling", action="store_true")
    args = ap.parse_args()

    width, height = (int(v) for v in args.resolution.lower().split("x"))
    out_dir = Path(args.output_dir)
    device = torch.device("cuda")

    df = pd.read_parquet(
        args.manifest,
        columns=["source_video_path", "start_frame", "end_frame", "num_frames", "source_num_resampled_frames"],
    ).sort_values(["source_video_path", "start_frame"])
    videos = sorted(df.source_video_path.unique())
    if args.num_videos:
        videos = videos[: args.num_videos]
    # Deterministic sharding by video path (stable across runs).
    videos = [v for v in videos if int(hashlib.md5(v.encode()).hexdigest(), 16) % args.num_shards == args.shard_index]
    groups = {v: g for v, g in df[df.source_video_path.isin(set(videos))].groupby("source_video_path")}
    logger.info(f"Shard {args.shard_index}/{args.num_shards}: {len(videos)} videos, resolution {width}x{height}")

    scale = read_video_scale_factors(args.vae_path)
    assert width % scale.width == 0 and height % scale.height == 0, f"{width}x{height} not multiple of VAE factors"
    assert (EXPECTED_FRAMES - 1) % scale.time == 0
    vae = load_video_vae_encoder(args.vae_path, device=device, dtype=torch.bfloat16)

    out_dir.mkdir(parents=True, exist_ok=True)
    fail_log = out_dir / f"failures_shard{args.shard_index}.jsonl"
    loader = DataLoader(
        VideoClips(videos, groups, args, height, width),
        batch_size=None,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch if args.num_workers else None,
        worker_init_fn=_lower_priority,
    )

    saver = ThreadPoolExecutor(max_workers=4)
    pending_saves: list = []
    done = skipped = 0
    t0 = time.time()
    t_ffmpeg = t_decode = t_enc = t_wait = 0.0
    t_wait_start = time.time()
    for vi, item in enumerate(loader):
        t_wait += time.time() - t_wait_start
        skipped += item["skipped"]
        t_ffmpeg += item["t_ffmpeg"]
        t_decode += item["t_decode"]
        vpath = item["vpath"]
        if "warning" in item:
            logger.warning(f"{vpath}: {item['warning']}")
        if "error" in item:
            logger.error(f"{vpath}: {item['error']}")
            with fail_log.open("a") as f:
                f.write(json.dumps({"source_video_path": vpath, "error": item["error"]}) + "\n")
        elif item["clips"] is not None:
            clips, starts = item["clips"], item["starts"]
            if args.max_samples:
                clips, starts = clips[: args.max_samples - done], starts[: args.max_samples - done]
            t = time.time()
            for b in range(0, len(starts), args.encode_batch):
                x = clips[b : b + args.encode_batch].to(device, non_blocking=True)  # [B, F, C, H, W] bf16 in [-1, 1]
                x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, F, H, W], as MediaDataset/process_videos
                with torch.inference_mode():
                    lat = _encode_video(vae, x, use_tiling=args.vae_tiling, scale_factors=scale)
                latents = lat["latents"].cpu()
                for j, s in enumerate(starts[b : b + args.encode_batch]):
                    pending_saves.append(saver.submit(_save, latents[j], lat, output_file(out_dir, vpath, s)))
                    done += 1
            t_enc += time.time() - t
        for fut in [f for f in pending_saves if f.done()]:
            fut.result()
        pending_saves = [f for f in pending_saves if not f.done()]
        if (vi + 1) % 20 == 0 or vi + 1 == len(videos):
            el = time.time() - t0
            logger.info(
                f"{vi + 1}/{len(videos)} videos, {done} clips ({done / max(el, 1e-9):.2f} clips/s) | "
                f"gpu-side encode {t_enc:.0f}s, waiting on workers {t_wait:.0f}s | "
                f"worker-time ffmpeg {t_ffmpeg:.0f}s decode+resize {t_decode:.0f}s"
            )
        if args.max_samples and done >= args.max_samples:
            break
        t_wait_start = time.time()

    for fut in pending_saves:
        fut.result()
    saver.shutdown()
    logger.info(f"Done: {done} encoded, {skipped} already existed, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
