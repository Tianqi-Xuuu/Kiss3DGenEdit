#!/usr/bin/env python3
"""
Batch RF editing utility.

Usage example:
    python batch_rf_edit.py \
        --bundle-root ./test_rf \
        --rf-gamma 0.6 0.8 \
        --rf-eta 0.95 \
        --rf-stop 0.8 \
        --guidance-scale 2.0

The script expects `--bundle-root` to contain one or more sub-folders. Each
sub-folder must provide a bundle image (e.g. `bundle.png`) and a `prompt.txt`.
All discovered bundles are processed as a batch, and every unique hyperparameter
combination is saved to a dedicated output directory.
"""
from __future__ import annotations

import argparse
import itertools
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from PIL import Image

from pipeline.kiss3d_wrapper import init_minimum_wrapper_from_config, run_edit_3d_bundle_rf
from pipeline.utils import logger


IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass
class BundleEntry:
    name: str
    image_path: Path
    prompt: str


def _fmt_float(value: float) -> str:
    """Create a readable float string for folder names."""
    if float(int(value)) == float(value):
        return str(int(value))
    text = f"{value:.4f}".rstrip("0")
    return text.rstrip(".")


def _find_bundle_image(folder: Path) -> Path | None:
    bundle_candidate = folder / "bundle.png"
    if bundle_candidate.is_file():
        return bundle_candidate

    for file in sorted(folder.iterdir()):
        if file.is_file() and file.suffix.lower() in IMG_EXTENSIONS:
            return file
    return None


def discover_bundle_entries(bundle_root: Path) -> List[BundleEntry]:
    """
    Discover bundle entries inside bundle_root.

    Every valid entry is a folder (bundle_root itself or its direct child folders)
    containing a prompt.txt and at least one image file.
    """
    if not bundle_root.exists():
        raise FileNotFoundError(f"Bundle root '{bundle_root}' does not exist.")

    entries: List[BundleEntry] = []

    def _collect_from_folder(folder: Path) -> None:
        prompt_path = folder / "prompt.txt"
        if not prompt_path.is_file():
            return
        bundle_image = _find_bundle_image(folder)
        if bundle_image is None:
            raise FileNotFoundError(f"No bundle image found inside {folder}. "
                                    f"Expected a file such as bundle.png.")
        prompt_text = prompt_path.read_text(encoding="utf-8").strip()
        if not prompt_text:
            raise ValueError(f"Prompt in {prompt_path} is empty.")
        entries.append(BundleEntry(folder.name, bundle_image, prompt_text))

    # Support a single folder that directly contains bundle/prompt.
    _collect_from_folder(bundle_root)

    for child in sorted(bundle_root.iterdir()):
        if child.is_dir():
            _collect_from_folder(child)

    if not entries:
        raise ValueError(f"No bundle entries with prompt.txt found under {bundle_root}.")
    return entries


def load_bundle_images(entries: Sequence[BundleEntry]) -> List[Image.Image]:
    """Load bundles as RGB PIL images (closed file handles)."""
    images: List[Image.Image] = []
    for entry in entries:
        with Image.open(entry.image_path) as img:
            images.append(img.convert("RGB").copy())
    return images


def build_param_grid(
    gammas: Sequence[float],
    etas: Sequence[float],
    stops: Sequence[float],
    guidances: Sequence[float],
) -> List[Tuple[float, float, float, float]]:
    return list(itertools.product(gammas, etas, stops, guidances))


def ensure_output_dir(base: Path) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"rf_batch_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch RF inversion/edit from bundle images.")
    parser.add_argument(
        "--bundle-root",
        default="./test_rf",
        type=Path,
        help="Directory that contains bundle folders with bundle image(s) and prompt.txt files.",
    )
    parser.add_argument(
        "--config",
        default="./pipeline/pipeline_config/default.yaml",
        type=Path,
        help="Pipeline config used to initialize the Kiss3D wrapper.",
    )
    parser.add_argument(
        "--output-dir",
        default="./outputs/rf_batch_results",
        type=Path,
        help="Base directory to store RF edit images.",
    )
    parser.add_argument(
        "--num-steps",
        default=28,
        type=int,
        help="Number of diffusion steps for RF edit.",
    )
    parser.add_argument(
        "--rf-gamma",
        nargs="+",
        type=float,
        default=[0.95],
        help="List of RF gamma values.",
    )
    parser.add_argument(
        "--rf-eta",
        nargs="+",
        type=float,
        default=[0.6],
        help="List of RF eta values.",
    )
    parser.add_argument(
        "--rf-stop",
        nargs="+",
        type=float,
        default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        help="List of RF stop ratios.",
    )
    parser.add_argument(
        "--guidance-scale",
        nargs="+",
        type=float,
        default=[2.0],
        help="List of guidance scales.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle_root: Path = args.bundle_root

    entries = discover_bundle_entries(bundle_root)
    logger.info(f"Discovered {len(entries)} bundle(s) under {bundle_root}.")
    for entry in entries:
        logger.info(" - %s (%s)", entry.name, entry.image_path)

    bundle_images = load_bundle_images(entries)
    prompts = [entry.prompt for entry in entries]

    param_grid = build_param_grid(args.rf_gamma, args.rf_eta, args.rf_stop, args.guidance_scale)
    logger.info("Running %d hyperparameter configuration(s).", len(param_grid))

    k3d_wrapper = init_minimum_wrapper_from_config(str(args.config))

    output_base = ensure_output_dir(args.output_dir)
    logger.info("Saving outputs to %s", output_base)

    for gamma, eta, stop, guidance in param_grid:
        combo_name = "_".join(
            [
                f"gamma{_fmt_float(gamma)}",
                f"eta{_fmt_float(eta)}",
                f"stop{_fmt_float(stop)}",
                f"guidance{_fmt_float(guidance)}",
            ]
        )
        combo_dir = output_base / combo_name
        combo_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Running RF edit with gamma=%s, eta=%s, stop=%s, guidance=%s",
            gamma,
            eta,
            stop,
            guidance,
        )

        _, _, _, save_path_tgt = run_edit_3d_bundle_rf(
            k3d_wrapper,
            bundle_img=bundle_images,
            prompt_tgt=prompts,
            rf_gamma=gamma,
            rf_eta=eta,
            rf_stop=stop,
            num_steps=args.num_steps,
            guidance_scale=guidance,
        )

        if isinstance(save_path_tgt, str):
            save_paths = [save_path_tgt]
        else:
            save_paths = list(save_path_tgt)

        for entry, tmp_path in zip(entries, save_paths):
            sample_dir = combo_dir / entry.name
            sample_dir.mkdir(parents=True, exist_ok=True)
            dst_path = sample_dir / f"{entry.name}_rf_bundle.png"
            shutil.copy2(tmp_path, dst_path)
            logger.info("Saved RF result to %s", dst_path)

        meta = {
            "rf_gamma": gamma,
            "rf_eta": eta,
            "rf_stop": stop,
            "guidance_scale": guidance,
            "num_steps": args.num_steps,
            "bundles": [str(entry.image_path) for entry in entries],
        }
        with open(combo_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
