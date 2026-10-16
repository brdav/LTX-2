#!/usr/bin/env python3
"""Encode Natix manifest clips into LTX video VAE latents.

For every source video referenced by the manifest:
  1. resample to 24 FPS with the canonical ffmpeg command,
  2. slice each manifest row's [start_frame, end_frame) (exactly 121 frames),
  3. resize + center-crop to the target resolution and encode with the video VAE.

Latents are written as ``<output-dir>/latents/<source_video_path without .mp4>/f<start_frame>.pt`` in the same
format as ``process_videos.py``. Existing outputs are skipped, so a rerun resumes.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import av
import imageio_ffmpeg
import numpy as np
import pandas as pd
import torch

from ltx_trainer import logger
from ltx_trainer.model_loader import load_video_vae_encoder, read_video_scale_factors
from ltx_trainer.process_videos import MediaDataset, _atomic_save, _encode_video

EXPECTED_FRAMES = 121
OUTPUT_FPS = 24.0


def resample(src: Path, dst: Path, ffmpeg: str) -> None:
    """Canonical resampling command (do not change flags)."""
    cmd = [ffmpeg, "-y", "-nostdin", "-i", str(src), "-vf", "fps=24", "-vsync", "cfr", "-an", str(dst)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def output_file(output_dir: Path, source_video_path: str, start_frame: int) -> Path:
    return output_dir / "latents" / Path(source_video_path).with_suffix("") / f"f{start_frame:06d}.pt"


def preprocess(frames: list, height: int, width: int, device: torch.device) -> torch.Tensor:
    """uint8 HWC frames -> [1, C, F, H, W] in [-1, 1], same resize/crop/normalize as MediaDataset."""
    clip = torch.from_numpy(np.stack(frames)).to(device).permute(0, 3, 1, 2).float().div_(255.0)
    ctx = SimpleNamespace(reshape_mode="center")
    clip = MediaDataset._resize_and_crop(ctx, clip, height, width)  # noqa: SLF001
    clip = (clip.clamp(0.0, 1.0) - 0.5) / 0.5
    return clip.permute(1, 0, 2, 3).unsqueeze(0).contiguous()


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-root", required=True, help="Directory that source_video_path is relative to")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--vae-path", required=True)
    ap.add_argument("--resolution", default="768x512", help="WIDTHxHEIGHT (multiples of the VAE spatial factor)")
    ap.add_argument("--num-videos", type=int, default=None, help="Only the first N source videos (pilot)")
    ap.add_argument("--max-samples", type=int, default=None, help="Stop after N clips in total (pilot)")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--tmp-dir", default=None, help="Node-local scratch for resampled mp4s (default: $TMPDIR)")
    ap.add_argument("--ffmpeg", default=None, help="ffmpeg binary (default: bundled imageio-ffmpeg)")
    ap.add_argument("--vae-tiling", action="store_true")
    args = ap.parse_args()

    width, height = (int(v) for v in args.resolution.lower().split("x"))
    ffmpeg = args.ffmpeg or shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
    out_dir = Path(args.output_dir)
    data_root = Path(args.data_root)
    device = torch.device("cuda")

    df = pd.read_parquet(
        args.manifest,
        columns=["sample_id", "source_video_path", "start_frame", "end_frame", "num_frames", "source_num_resampled_frames"],
    ).sort_values(["source_video_path", "start_frame"])
    videos = sorted(df.source_video_path.unique())
    if args.num_videos:
        videos = videos[: args.num_videos]
    # Deterministic sharding by video path (stable across runs / shard counts).
    videos = [v for v in videos if int(hashlib.md5(v.encode()).hexdigest(), 16) % args.num_shards == args.shard_index]
    groups = {v: g for v, g in df[df.source_video_path.isin(set(videos))].groupby("source_video_path")}
    logger.info(f"Shard {args.shard_index}/{args.num_shards}: {len(videos)} videos, resolution {width}x{height}")

    scale = read_video_scale_factors(args.vae_path)
    assert width % scale.width == 0 and height % scale.height == 0, f"{width}x{height} not multiple of VAE factors"
    assert (EXPECTED_FRAMES - 1) % scale.time == 0
    vae = load_video_vae_encoder(args.vae_path, device=device, dtype=torch.bfloat16)

    out_dir.mkdir(parents=True, exist_ok=True)
    fail_log = out_dir / f"failures_shard{args.shard_index}.jsonl"
    tmp_root = Path(args.tmp_dir or tempfile.gettempdir())
    done = skipped = 0
    t0 = time.time()
    t_ffmpeg = t_decode = t_enc = 0.0

    for vi, vpath in enumerate(videos):
        rows = [r for r in groups[vpath].itertuples() if not output_file(out_dir, vpath, r.start_frame).is_file()]
        if not rows:
            skipped += len(groups[vpath])
            continue
        if args.max_samples and done >= args.max_samples:
            break
        if args.max_samples:
            rows = rows[: args.max_samples - done]
        try:
            with tempfile.TemporaryDirectory(dir=tmp_root) as td:
                resampled = Path(td) / "resampled.mp4"
                t = time.time()
                resample(data_root / vpath, resampled, ffmpeg)
                t_ffmpeg += time.time() - t

                expected_total = int(rows[0].source_num_resampled_frames)
                pending = {r.start_frame: r for r in rows}
                starts = sorted(pending)
                buf: dict[int, object] = {}
                n_frames = 0
                t = time.time()
                with av.open(str(resampled)) as c:
                    for idx, fr in enumerate(c.decode(video=0)):
                        n_frames = idx + 1
                        buf[idx] = fr.to_ndarray(format="rgb24")
                        while starts and idx + 1 >= pending[starts[0]].end_frame:
                            r = pending[starts.pop(0)]
                            frames = [buf[i] for i in range(r.start_frame, r.end_frame)]
                            assert len(frames) == EXPECTED_FRAMES == r.num_frames
                            t_decode += time.time() - t
                            t = time.time()
                            with torch.inference_mode():
                                video = preprocess(frames, height, width, device)
                                lat = _encode_video(vae, video, use_tiling=args.vae_tiling, scale_factors=scale)
                            t_enc += time.time() - t
                            _f = output_file(out_dir, vpath, r.start_frame)
                            _f.parent.mkdir(parents=True, exist_ok=True)
                            _atomic_save(
                                {
                                    "latents": lat["latents"][0].cpu().contiguous(),
                                    "num_frames": lat["num_frames"],
                                    "height": lat["height"],
                                    "width": lat["width"],
                                    "fps": OUTPUT_FPS,
                                },
                                _f,
                            )
                            done += 1
                            t = time.time()
                        keep_from = starts[0] if starts else n_frames
                        for k in [k for k in buf if k < keep_from]:
                            del buf[k]
                if starts:
                    raise RuntimeError(f"video has {n_frames} frames, clips {starts} out of range")
                if n_frames != expected_total:
                    logger.warning(f"{vpath}: {n_frames} resampled frames, manifest says {expected_total}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"{vpath}: {e}")
            with fail_log.open("a") as f:
                f.write(json.dumps({"source_video_path": vpath, "error": str(e)[:500]}) + "\n")
        if (vi + 1) % 5 == 0 or vi + 1 == len(videos):
            el = time.time() - t0
            logger.info(
                f"{vi + 1}/{len(videos)} videos, {done} clips encoded ({done / max(el, 1e-9):.2f} clips/s) | "
                f"ffmpeg {t_ffmpeg:.0f}s decode {t_decode:.0f}s encode {t_enc:.0f}s"
            )
        if args.max_samples and done >= args.max_samples:
            break

    logger.info(f"Done: {done} encoded, {skipped} already existed, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
