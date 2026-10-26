import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from safetensors import safe_open
from torch import Tensor
from torch.utils.data import Dataset

from ltx_trainer import logger

# Constants for precomputed data directories
PRECOMPUTED_DIR_NAME = ".precomputed"


class DummyDataset(Dataset):
    """Produce random latents and prompt embeddings. For minimal demonstration and benchmarking purposes"""

    def __init__(
        self,
        width: int = 1024,
        height: int = 1024,
        num_frames: int = 25,
        fps: int = 24,
        dataset_length: int = 200,
        latent_dim: int = 128,
        latent_spatial_compression_ratio: int = 32,
        latent_temporal_compression_ratio: int = 8,
        prompt_embed_dim: int = 4096,
        prompt_sequence_length: int = 256,
    ) -> None:
        if width % 32 != 0:
            raise ValueError(f"Width must be divisible by 32, got {width=}")

        if height % 32 != 0:
            raise ValueError(f"Height must be divisible by 32, got {height=}")

        if num_frames % 8 != 1:
            raise ValueError(f"Number of frames must have a remainder of 1 when divided by 8, got {num_frames=}")

        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.fps = fps
        self.dataset_length = dataset_length
        self.latent_dim = latent_dim
        self.num_latent_frames = (num_frames - 1) // latent_temporal_compression_ratio + 1
        self.latent_height = height // latent_spatial_compression_ratio
        self.latent_width = width // latent_spatial_compression_ratio
        self.latent_sequence_length = self.num_latent_frames * self.latent_height * self.latent_width
        self.prompt_embed_dim = prompt_embed_dim
        self.prompt_sequence_length = prompt_sequence_length

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> dict[str, dict[str, Tensor]]:
        return {
            "latent_conditions": {
                "latents": torch.randn(
                    self.latent_dim,
                    self.num_latent_frames,
                    self.latent_height,
                    self.latent_width,
                ),
                "num_frames": self.num_latent_frames,
                "height": self.latent_height,
                "width": self.latent_width,
                "fps": self.fps,
            },
            "text_conditions": {
                "video_prompt_embeds": torch.randn(
                    self.prompt_sequence_length,
                    self.prompt_embed_dim,
                ),
                "audio_prompt_embeds": torch.randn(
                    self.prompt_sequence_length,
                    self.prompt_embed_dim,
                ),
                "prompt_attention_mask": torch.ones(
                    self.prompt_sequence_length,
                    dtype=torch.bool,
                ),
            },
        }


class PrecomputedDataset(Dataset):
    def __init__(self, data_root: str, data_sources: dict[str, str] | list[str] | None = None) -> None:
        """
        Generic dataset for loading precomputed data from multiple sources.
        Args:
            data_root: Root directory containing preprocessed data
            data_sources: Either:
              - Dict mapping directory names to output keys
              - List of directory names (keys will equal values)
              - None (defaults to ["latents", "conditions"])
        Example:
            # Standard mode (list)
            dataset = PrecomputedDataset("data/", ["latents", "conditions"])
            # Standard mode (dict)
            dataset = PrecomputedDataset("data/", {"latents": "latent_conditions", "conditions": "text_conditions"})
            # IC-LoRA mode
            dataset = PrecomputedDataset("data/", ["latents", "conditions", "reference_latents"])
        Note:
            Latents are always returned in non-patchified format [C, F, H, W].
            Legacy patchified format [seq_len, C] is automatically converted.
        """
        super().__init__()

        self.data_root = self._setup_data_root(data_root)
        self.data_sources = self._normalize_data_sources(data_sources)
        self.source_paths = self._setup_source_paths()
        self.sample_files = self._discover_samples()
        self._validate_setup()

    @staticmethod
    def _setup_data_root(data_root: str) -> Path:
        """Setup and validate the data root directory."""
        data_root = Path(data_root).expanduser().resolve()

        if not data_root.exists():
            raise FileNotFoundError(f"Data root directory does not exist: {data_root}")

        # If the given path is the dataset root, use the precomputed subdirectory
        if (data_root / PRECOMPUTED_DIR_NAME).exists():
            data_root = data_root / PRECOMPUTED_DIR_NAME

        return data_root

    @staticmethod
    def _normalize_data_sources(data_sources: dict[str, str] | list[str] | None) -> dict[str, str]:
        """Normalize data_sources input to a consistent dict format."""
        if data_sources is None:
            # Default sources
            return {"latents": "latent_conditions", "conditions": "text_conditions"}
        elif isinstance(data_sources, list):
            # Convert list to dict where keys equal values
            return {source: source for source in data_sources}
        elif isinstance(data_sources, dict):
            return data_sources.copy()
        else:
            raise TypeError(f"data_sources must be dict, list, or None, got {type(data_sources)}")

    def _setup_source_paths(self) -> dict[str, Path]:
        """Map data source names to their actual directory paths."""
        source_paths = {}

        for dir_name in self.data_sources:
            source_path = self.data_root / dir_name
            source_paths[dir_name] = source_path

            # Check that all sources exist.
            if not source_path.exists():
                raise FileNotFoundError(f"Required {dir_name} directory does not exist: {source_path}")

        return source_paths

    def _discover_samples(self) -> dict[str, list[Path]]:
        """Discover all valid sample files across all data sources.
        Uses a fast two-pass approach: first globs all sources in parallel to build
        full-path sets in memory, then checks expected paths via set membership.
        This avoids O(N * num_sources) stat calls on networked filesystems while
        correctly handling path remapping (e.g. latent_X.pt -> condition_X.pt).
        """
        if not self.data_sources:
            raise ValueError("No data sources configured")

        data_key = "latents" if "latents" in self.data_sources else next(iter(self.data_sources.keys()))
        data_path = self.source_paths[data_key]

        # Pass 1: Glob all sources in parallel, build full-path sets
        def _glob_source(dir_name: str) -> tuple[list[Path], set[str]]:
            source_path = self.source_paths[dir_name]
            paths = list(source_path.glob("**/*.pt"))
            path_set = {str(p) for p in paths}
            return paths, path_set

        with ThreadPoolExecutor(max_workers=len(self.data_sources)) as executor:
            glob_results = dict(
                zip(
                    self.data_sources.keys(),
                    executor.map(_glob_source, self.data_sources.keys()),
                    strict=True,
                )
            )

        # Get primary source files (cached from glob, no second scan)
        data_files, _ = glob_results[data_key]
        if not data_files:
            raise ValueError(f"No data files found in {data_path}")
        data_files.sort()

        # Build path sets for non-primary sources
        other_path_sets = {
            dir_name: path_set for dir_name, (_, path_set) in glob_results.items() if dir_name != data_key
        }

        # Pass 2: For each primary file, check if expected paths exist in other sources' sets
        sample_files: dict[str, list[Path]] = {output_key: [] for output_key in self.data_sources.values()}
        valid_count = 0

        for data_file in data_files:
            rel_path = data_file.relative_to(data_path)

            # Check all other sources via set lookup (O(1) per source, no stat calls)
            all_exist = True
            for dir_name, path_set in other_path_sets.items():
                expected = self._get_expected_file_path(dir_name, data_file, rel_path)
                if str(expected) not in path_set:
                    logger.debug(f"Skipping {data_file.name}: no matching {dir_name} file at {expected}")
                    all_exist = False
                    break

            if all_exist:
                self._fill_sample_data_files(data_file, rel_path, sample_files)
                valid_count += 1

        skipped = len(data_files) - valid_count
        if skipped > 0:
            logger.info(f"Skipped {skipped} invalid samples from {len(data_files)} total")

        return sample_files

    def _get_expected_file_path(self, dir_name: str, data_file: Path, rel_path: Path) -> Path:
        """Get the expected file path for a given data source."""
        source_path = self.source_paths[dir_name]

        # For conditions, handle legacy naming where latent_X.pt maps to condition_X.pt
        if dir_name == "conditions" and data_file.name.startswith("latent_"):
            return source_path / f"condition_{data_file.stem[7:]}.pt"

        return source_path / rel_path

    def _fill_sample_data_files(self, data_file: Path, rel_path: Path, sample_files: dict[str, list[Path]]) -> None:
        """Add a valid sample to the sample_files tracking."""
        for dir_name, output_key in self.data_sources.items():
            expected_path = self._get_expected_file_path(dir_name, data_file, rel_path)
            sample_files[output_key].append(expected_path.relative_to(self.source_paths[dir_name]))

    def _validate_setup(self) -> None:
        """Validate that the dataset setup is correct."""
        sample_counts = {key: len(files) for key, files in self.sample_files.items()}
        if not sample_counts or all(count == 0 for count in sample_counts.values()):
            raise ValueError(
                f"No valid samples found in {self.data_root} - all configured data sources "
                f"({list(self.data_sources)}) must have matching files (per-source counts: {sample_counts})"
            )

        # Verify all output keys have the same number of samples
        if len(set(sample_counts.values())) > 1:
            raise ValueError(f"Mismatched sample counts across sources: {sample_counts}")

    def __len__(self) -> int:
        # Use the first output key as reference count
        first_key = next(iter(self.sample_files.keys()))
        return len(self.sample_files[first_key])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        result = {}

        for dir_name, output_key in self.data_sources.items():
            source_path = self.source_paths[dir_name]
            file_rel_path = self.sample_files[output_key][index]
            file_path = source_path / file_rel_path

            try:
                data = torch.load(file_path, map_location="cpu", weights_only=True)

                # Normalize video latent format if this is a latent source
                if "latent" in dir_name.lower():
                    data = self._normalize_video_latents(data)

                result[output_key] = data
            except Exception as e:
                raise RuntimeError(f"Failed to load {output_key} from {file_path}: {e}") from e

        # Add index for debugging
        result["idx"] = index
        return result

    @staticmethod
    def _normalize_video_latents(data: dict) -> dict:
        """
        Normalize video latents to non-patchified format [C, F, H, W].
        Used for keeping backward compatibility with legacy datasets.
        """
        latents = data["latents"]

        # Check if latents are in legacy patchified format [seq_len, C]
        if latents.dim() == 2:
            # Legacy format: [seq_len, C] where seq_len = F * H * W
            num_frames = data["num_frames"]
            height = data["height"]
            width = data["width"]

            # Unpatchify: [seq_len, C] -> [C, F, H, W]
            latents = rearrange(
                latents,
                "(f h w) c -> c f h w",
                f=num_frames,
                h=height,
                w=width,
            )

            # Update the data dict with unpatchified latents
            data = data.copy()
            data["latents"] = latents

        return data


def _read_packed_index(root: Path) -> list[dict]:
    """All rows of ``<root>/index/*.jsonl`` (one dict per latent: source_video_path, start_frame, view, file, key)."""
    return [json.loads(line) for f in sorted((root / "index").glob("*.jsonl")) for line in f.read_text().splitlines()]


@dataclass
class _AuxSource:
    """One auxiliary packed source of a ``PackedLatentDataset`` (e.g. OSM maps, satellite crops)."""

    root: Path
    weights: dict[str, float]  # view -> sampling weight
    per_clip: list[dict[str, tuple[str, str]]]  # per clip: {drawable view: (file, key)}, empty if none
    zero: Tensor  # shape template for clips without it


@dataclass
class _AnchorSource:
    """Anchor frames of a ``PackedLatentDataset``: a variable-length set of independent stills per clip.

    Stored flat (numpy) so that the many millions of (clip, anchor) pairs stay small and do not get copied into every
    DataLoader worker: clip ``i`` has the candidate ids ``keys[offsets[i]:offsets[i + 1]]`` in ``files[clip_file[i]]``.
    """

    root: Path
    num_anchors: int
    files: list[str]
    clip_file: np.ndarray  # [n_clips] int32, index into files (-1: no anchors)
    offsets: np.ndarray  # [n_clips + 1] int64
    keys: np.ndarray  # [total] int64, candidate ids
    zero: Tensor  # [C, 1, H, W] shape template

    def count(self, i: int) -> int:
        return int(self.offsets[i + 1] - self.offsets[i])


class PackedLatentDataset(Dataset):
    """Clip-level reader for latents packed one safetensors file per source video (see ``tokenize_natix.py``).

    Samples come from the JSONL index (``<root>/index/*.jsonl``) rather than a directory glob. Each item is read with a
    memory-mapped slice, so only that clip's bytes are touched. Returns the same dict as a ``latents/*.pt`` source:
    ``latents`` [C, F, H, W], ``num_frames``, ``height``, ``width``, ``fps``.
    Multi-view: pass several ``views``; the item then holds ``latents`` [V, C, F, H, W] stacked in that order, and
    only clips that have every requested view are used.

    Auxiliary sources: ``aux_roots`` maps a name to a packed root of per-clip stills, e.g. ``{"osm": <root of
    tokenize_natix_osm.py>, "satellite": <root of tokenize_natix_satellite.py>}``. Every item then also carries, per
    name, ``<name>_latents`` (a latents dict of the same form, [C, 1, H, W], ``fps`` taken from the clip),
    ``<name>_view`` (the view that was loaded) and ``<name>_available``. An auxiliary latent is joined to its clip on
    ``(source_video_path, start_frame)``. Each time an item is read, one view per source is drawn with probability
    proportional to ``aux_view_weights[name]`` (e.g. ``{"osm": {"map_matched": 0.7, "predicted": 0.3}}``; default:
    equal over the views in that source's index), among the views that exist for that clip.
    ``require`` lists the sources a clip must have to be kept (default: all of them; ``()``: keep every clip). Clips
    kept without a source get zero latents and ``<name>_available=False``, so default collation still works.

    Anchor frames: ``anchor_root`` is a packed root written by ``tokenize_natix_anchors.py`` (rows with
    ``view="anchor"``), ``num_anchors`` = K. A clip has a variable number of anchors (about 11 on average, up to 50),
    each an independent still (never a video), so every read draws a random subset of ``min(K, n)`` of them (without
    replacement, torch RNG: seeded per worker). The item holds ``anchors_latents`` (a latents dict whose ``latents`` is
    [K, C, 1, H, W], zero-padded when the clip has fewer than K), ``anchors_mask`` ([K] bool, True = real anchor) and
    ``anchors_available``. ``"anchors"`` can be named in ``require`` like an auxiliary source (and is part of the
    default), which keeps only the clips that have at least one anchor.
    """

    def __init__(
        self,
        root: str,
        views: tuple[str, ...] = ("front",),
        aux_roots: dict[str, str] | None = None,
        aux_view_weights: dict[str, dict[str, float]] | None = None,
        require: tuple[str, ...] | None = None,
        anchor_root: str | None = None,
        num_anchors: int = 8,
    ) -> None:
        super().__init__()
        self.root, self.views = Path(root).expanduser().resolve(), views
        rows: dict[tuple[str, str], dict[str, str]] = {}  # (file, start) -> {view: key}; dict dedupes requeued rows
        clip_ids: dict[tuple[str, str], tuple[str, int]] = {}  # (file, start) -> (source_video_path, start_frame)
        for row in _read_packed_index(self.root):
            rows.setdefault((row["file"], f"{row['start_frame']:06d}"), {})[row["view"]] = row["key"]
            clip_ids[(row["file"], f"{row['start_frame']:06d}")] = (row["source_video_path"], row["start_frame"])
        keep = [fs for fs, keys in sorted(rows.items()) if all(v in keys for v in views)]
        self.samples = [(f, [rows[(f, s)][v] for v in views]) for f, s in keep]
        if not self.samples:
            raise ValueError(f"No clips with views {views} found in {self.root / 'index'}")

        aux_roots, aux_view_weights = aux_roots or {}, aux_view_weights or {}
        if unknown := set(aux_view_weights) - set(aux_roots):
            raise ValueError(f"aux_view_weights for unknown sources {sorted(unknown)}; sources are {sorted(aux_roots)}")
        ids = [clip_ids[fs] for fs in keep]
        self.aux = {
            name: self._init_aux(name, Path(r).expanduser().resolve(), ids, aux_view_weights.get(name))
            for name, r in aux_roots.items()
        }
        if num_anchors < 1:
            raise ValueError(f"num_anchors must be >= 1, got {num_anchors}")
        self.anchors = (
            self._init_anchors(Path(anchor_root).expanduser().resolve(), ids, num_anchors) if anchor_root else None
        )
        names = (*self.aux, *(("anchors",) if self.anchors else ()))
        required = names if require is None else require
        if unknown := set(required) - set(names):
            raise ValueError(f"require names unknown sources {sorted(unknown)}; sources are {sorted(names)}")
        if required:
            kept = [i for i in range(len(self.samples)) if all(self._has(n, i) for n in required)]
            if not kept:
                raise ValueError(f"No clips of {self.root} have all of {required}")
            self.samples = [self.samples[i] for i in kept]
            for aux in self.aux.values():
                aux.per_clip = [aux.per_clip[i] for i in kept]
            if self.anchors:
                a = self.anchors
                a.clip_file = a.clip_file[kept]
                counts = np.diff(a.offsets)[kept]
                starts = a.offsets[:-1][kept]
                a.keys = np.concatenate([a.keys[s : s + c] for s, c in zip(starts, counts, strict=True)])
                a.offsets = np.concatenate([[0], np.cumsum(counts)])
        logger.info(
            f"PackedLatentDataset: {len(self.samples)} clips, views {views}"
            + "".join(
                f", {name} from {aux.root} ({sum(bool(m) for m in aux.per_clip)} with it)"
                for name, aux in self.aux.items()
            )
            + (
                f", anchors from {self.anchors.root} ({int((np.diff(self.anchors.offsets) > 0).sum())} with any, "
                f"{len(self.anchors.keys)} in total, {self.anchors.num_anchors} per read)"
                if self.anchors
                else ""
            )
        )

    def _has(self, name: str, i: int) -> bool:
        """Whether clip ``i`` has source ``name`` (an auxiliary source, or ``"anchors"``)."""
        return self.anchors.count(i) > 0 if name == "anchors" else bool(self.aux[name].per_clip[i])

    @staticmethod
    def _init_anchors(anchor_root: Path, ids: list[tuple[str, int]], num_anchors: int) -> _AnchorSource:
        """Look up, for every clip (``ids``: source_video_path, start_frame), its anchors in ``anchor_root``."""
        position = {c: i for i, c in enumerate(ids)}
        files: dict[str, int] = {}
        per_clip: dict[int, tuple[int, set[int]]] = {}  # clip position -> (file index, candidate ids); a set dedupes
        for row in _read_packed_index(anchor_root):
            i = position.get((row["source_video_path"], row["start_frame"]))
            if i is None or row.get("view", "anchor") != "anchor":
                continue
            file = files.setdefault(row["file"], len(files))
            entry = per_clip.setdefault(i, (file, set()))
            if entry[0] != file:
                raise ValueError(f"Anchors of {row['source_video_path']}#{row['start_frame']} span several files")
            entry[1].add(int(row["key"]))
        if not per_clip:
            raise ValueError(f"No anchor latents matching the clips in {anchor_root / 'index'}")
        clip_file = np.full(len(ids), -1, dtype=np.int32)
        counts = np.zeros(len(ids), dtype=np.int64)
        for i, (file, keys) in per_clip.items():
            clip_file[i], counts[i] = file, len(keys)
        offsets = np.concatenate([[0], np.cumsum(counts)])
        flat = np.empty(int(offsets[-1]), dtype=np.int64)
        for i, (_, keys) in per_clip.items():
            flat[offsets[i] : offsets[i + 1]] = sorted(keys)
        names = list(files)
        i0 = next(iter(per_clip))
        with safe_open(str(anchor_root / names[per_clip[i0][0]]), framework="pt", device="cpu") as f:
            zero = torch.zeros_like(f.get_tensor(str(next(iter(per_clip[i0][1])))))
        return _AnchorSource(anchor_root, num_anchors, names, clip_file, offsets, flat, zero)

    def _init_aux(
        self, name: str, aux_root: Path, ids: list[tuple[str, int]], weights: dict[str, float] | None
    ) -> _AuxSource:
        """Look up, for every clip (``ids``: source_video_path, start_frame), its latents per view in ``aux_root``."""
        wanted = set(ids)
        found: dict[tuple[str, int], dict[str, tuple[str, str]]] = {}
        for row in _read_packed_index(aux_root):
            clip = (row["source_video_path"], row["start_frame"])
            if clip in wanted:
                found.setdefault(clip, {})[row["view"]] = (sys.intern(row["file"]), row["key"])
        available_views = sorted({v for views in found.values() for v in views})
        if not available_views:
            raise ValueError(f"No {name} latents matching the clips of {self.root} in {aux_root / 'index'}")
        weights = dict(weights) if weights is not None else dict.fromkeys(available_views, 1.0)
        if set(weights) - set(available_views) or any(w < 0 for w in weights.values()) or not sum(weights.values()) > 0:
            raise ValueError(f"Invalid view weights {weights} for {name}; its views are {available_views}")
        # Per clip: only the views that can be drawn (positive weight), empty if the clip has no usable one.
        per_clip = [{v: fk for v, fk in found.get(c, {}).items() if weights.get(v, 0.0) > 0} for c in ids]
        file, key = next(iter(next(m for m in per_clip if m).values()))
        with safe_open(str(aux_root / file), framework="pt", device="cpu") as f:
            zero = torch.zeros_like(f.get_tensor(key))
        return _AuxSource(aux_root, weights, per_clip, zero)

    @staticmethod
    def _draw_view(aux: _AuxSource, available: dict[str, tuple[str, str]]) -> str:
        """One view of ``available``, with probability proportional to its weight (torch RNG: seeded per worker)."""
        names = sorted(available)
        u = torch.rand(()).item() * sum(aux.weights[v] for v in names)
        for v in names:
            u -= aux.weights[v]
            if u < 0:
                return v
        return names[-1]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        file, keys = self.samples[index]
        with safe_open(str(self.root / file), framework="pt", device="cpu") as f:
            meta = f.metadata()
            latents = torch.stack([f.get_tensor(k) for k in keys]) if len(keys) > 1 else f.get_tensor(keys[0])
        item = {
            "latents": latents,
            "num_frames": int(meta["num_frames"]),
            "height": int(meta["height"]),
            "width": int(meta["width"]),
            "fps": float(meta["fps"]),
        }
        for name, aux in self.aux.items():
            item |= self._load_aux(name, aux, index, item["fps"])
        if self.anchors:
            item |= self._load_anchors(self.anchors, index, item["fps"])
        return item

    @staticmethod
    def _load_anchors(anchors: _AnchorSource, index: int, fps: float) -> dict:
        """A random subset of at most ``num_anchors`` anchors of clip ``index``, zero-padded to ``num_anchors``."""
        k, n = anchors.num_anchors, anchors.count(index)
        latents, meta = anchors.zero.unsqueeze(0).repeat(k, 1, 1, 1, 1), None
        mask = torch.zeros(k, dtype=torch.bool)
        if n:
            start = int(anchors.offsets[index])
            chosen = anchors.keys[start + torch.randperm(n)[:k].numpy()]
            with safe_open(
                str(anchors.root / anchors.files[int(anchors.clip_file[index])]), framework="pt", device="cpu"
            ) as f:
                meta = f.metadata()
                latents[: len(chosen)] = torch.stack([f.get_tensor(str(c)) for c in chosen])
            mask[: len(chosen)] = True
        _, _, num_frames, height, width = latents.shape
        return {
            "anchors_latents": {
                "latents": latents,  # [K, C, 1, H, W]; each anchor is a separate still
                "num_frames": int(meta["num_frames"]) if meta else num_frames,
                "height": int(meta["height"]) if meta else height,
                "width": int(meta["width"]) if meta else width,
                "fps": fps,
            },
            "anchors_mask": mask,
            "anchors_available": bool(n),
        }

    def _load_aux(self, name: str, aux: _AuxSource, index: int, fps: float) -> dict:
        available = aux.per_clip[index]
        if not available:
            view, latents, meta = "", aux.zero, None
        else:
            view = self._draw_view(aux, available)
            file, key = available[view]
            with safe_open(str(aux.root / file), framework="pt", device="cpu") as f:
                meta, latents = f.metadata(), f.get_tensor(key)
        _, num_frames, height, width = latents.shape
        return {
            f"{name}_latents": {
                "latents": latents,
                "num_frames": int(meta["num_frames"]) if meta else num_frames,
                "height": int(meta["height"]) if meta else height,
                "width": int(meta["width"]) if meta else width,
                "fps": fps,  # a still has no fps of its own; it takes the clip's so its temporal position matches
            },
            f"{name}_view": view,
            f"{name}_available": bool(available),
        }
