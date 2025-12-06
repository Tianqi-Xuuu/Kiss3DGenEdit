#!/usr/bin/env python3
"""
Batch ControlNet editing utility.

Usage example:
    python batch_controlnet_edit.py \
            --control-mode tile blur \
            --control-guidance-end 0.65 0.65 \
            --control-guidance-end 0.35 0.9 \
            --control-scale 0.6 0.9 \
            --strength 0.9 0.95

The script expects `--bundle-root` to contain folders with a bundle image and a
`prompt.txt`. Each discovered entry is processed across all requested
hyperparameter combinations, and the results are saved into dedicated folders.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
import torchvision.transforms.functional as TF
import torchvision.utils as vutils
from PIL import Image

from pipeline.kiss3d_wrapper import (
    init_minimum_control_net_wrapper_from_config,
    seed_everything,
)
from pipeline.utils import logger


IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass
class BundleEntry:
    name: str
    image_path: Path
    prompt: str


@dataclass
class BundlePayload:
    entry: BundleEntry
    tensor: torch.Tensor
    control_images: List[Image.Image]


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
            raise FileNotFoundError(
                f"No bundle image found inside {folder}. Expected a file such as bundle.png."
            )
        prompt_text = prompt_path.read_text(encoding="utf-8").strip()
        if not prompt_text:
            raise ValueError(f"Prompt in {prompt_path} is empty.")
        entries.append(BundleEntry(folder.name, bundle_image, prompt_text))

    _collect_from_folder(bundle_root)

    for child in sorted(bundle_root.iterdir()):
        if child.is_dir():
            _collect_from_folder(child)

    if not entries:
        raise ValueError(f"No bundle entries with prompt.txt found under {bundle_root}.")
    return entries


def _load_bundle_tensor(image_path: Path) -> torch.Tensor:
    with Image.open(image_path) as img:
        bundle = img.convert("RGB")
    if bundle.size != (2048, 1024):
        logger.warning("Resizing bundle image %s from %s to 2048x1024.", image_path, bundle.size)
        bundle = bundle.resize((2048, 1024), Image.BICUBIC)
    return TF.to_tensor(bundle)


def _prepare_control_images(
    k3d_wrapper,
    bundle_tensor: torch.Tensor,
    control_modes: Sequence[str],
    down_scale: int,
    blur_kernel: int,
    blur_sigma: float,
    save_cond: bool,
) -> List[Image.Image]:
    control_images: List[Image.Image] = []
    for mode in control_modes:
        kwargs = {}
        if mode in {"tile", "lq"}:
            kwargs["down_scale"] = down_scale
        elif mode == "blur":
            kwargs["kernel_size"] = blur_kernel
            kwargs["sigma"] = blur_sigma
        control_image = k3d_wrapper.preprocess_controlnet_cond_image(
            bundle_tensor, mode, save_intermediate_results=save_cond, **kwargs
        )
        control_images.append(control_image.copy())
    return control_images


def load_bundle_payloads(
    entries: Sequence[BundleEntry],
    k3d_wrapper,
    control_modes: Sequence[str],
    down_scale: int,
    blur_kernel: int,
    blur_sigma: float,
    save_cond: bool,
) -> List[BundlePayload]:
    payloads: List[BundlePayload] = []
    for entry in entries:
        tensor = _load_bundle_tensor(entry.image_path)
        control_images = _prepare_control_images(
            k3d_wrapper,
            tensor,
            control_modes,
            down_scale,
            blur_kernel,
            blur_sigma,
            save_cond,
        )
        payloads.append(BundlePayload(entry=entry, tensor=tensor, control_images=control_images))
    return payloads


def build_param_grid(
    strengths: Sequence[float],
    steps: Sequence[int],
    control_scales: Sequence[float],
    seeds: Sequence[int],
) -> List[Tuple[float, int, float, int]]:
    return list(itertools.product(strengths, steps, control_scales, seeds))


def ensure_output_dir(base: Path) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"controlnet_batch_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _broadcast(values: Sequence[float] | None, target_len: int, label: str, default: float) -> List[float]:
    if target_len == 0:
        return []
    data = list(values) if values is not None else [default]
    if len(data) == 1:
        return data * target_len
    if len(data) != target_len:
        raise ValueError(f"{label} expects 1 or {target_len} values, got {len(data)}.")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch ControlNet editing from bundle images.")
    parser.add_argument(
        "--bundle-root",
        default=Path("./test_rf"),
        type=Path,
        help="Directory containing bundle folders with prompt.txt files.",
    )
    parser.add_argument(
        "--config",
        default=Path("./pipeline/pipeline_config/default.yaml"),
        type=Path,
        help="Pipeline config used to initialize the Kiss3D ControlNet wrapper.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("./outputs/controlnet_batch_results"),
        type=Path,
        help="Base directory to store ControlNet edit images.",
    )
    parser.add_argument(
        "--control-mode",
        nargs="+",
        default=["tile"],
        choices=["tile", "lq", "blur"],
        help="ControlNet mode(s) to enable. Multiple values will be applied jointly.",
    )
    parser.add_argument(
        "--control-guidance-start",
        nargs="+",
        type=float,
        default=[0.0],
        help="Start ratio(s) for each control mode. Provide one value to broadcast.",
    )
    parser.add_argument(
        "--control-guidance-end",
        nargs="+",
        type=float,
        action="append",
        default=[[0.1],
                 [0.3],
                 [0.5]],
        help="End ratio list(s) for each control mode. Repeat the flag to sweep multiple lists.",
    )
    parser.add_argument(
        "--control-mode-scale",
        nargs="+",
        type=float,
        default=[1],
        help="Optional per-mode multipliers applied to the global --control-scale value.",
    )
    parser.add_argument(
        "--control-scale",
        nargs="+",
        type=float,
        default=[1.0],
        help="Global ControlNet conditioning scale(s) to sweep.",
    )
    parser.add_argument(
        "--strength",
        nargs="+",
        type=float,
        default=[0.5, 0.7, 0.9],
        help="Flux img2img strength value(s).",
    )
    parser.add_argument(
        "--num-steps",
        nargs="+",
        type=int,
        default=[20],
        help="Number of diffusion steps for each run.",
    )
    parser.add_argument(
        "--seed",
        nargs="+",
        type=int,
        default=[42],
        help="Random seed(s) applied per configuration.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=3.5,
        help="Classifier-free guidance scale for Flux.",
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=1.0,
        help="LoRA scale applied through joint_attention_kwargs.",
    )
    parser.add_argument(
        "--downscale",
        type=int,
        default=4,
        help="Downscale factor used for tile/lq control preprocessing.",
    )
    parser.add_argument(
        "--blur-kernel",
        type=int,
        default=51,
        help="Kernel size used when control_mode=blur.",
    )
    parser.add_argument(
        "--blur-sigma",
        type=float,
        default=2.0,
        help="Sigma used when control_mode=blur.",
    )
    parser.add_argument(
        "--save-cond-images",
        action="store_true",
        help="Keep intermediate control condition visualizations in TMP_DIR.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle_root: Path = args.bundle_root

    entries = discover_bundle_entries(bundle_root)
    logger.info("Discovered %d bundle(s) under %s.", len(entries), bundle_root)
    for entry in entries:
        logger.info(" - %s (%s)", entry.name, entry.image_path)

    control_modes = args.control_mode
    control_guidance_start = _broadcast(
        args.control_guidance_start, len(control_modes), "control-guidance-start", default=0.0
    )
    end_arg_lists = args.control_guidance_end or [[0.65]]
    control_guidance_end_sets = [
        _broadcast(end_list, len(control_modes), "control-guidance-end", default=0.65)
        for end_list in end_arg_lists
    ]

    control_mode_scale = _broadcast(
        args.control_mode_scale, len(control_modes), "control-mode-scale", default=1.0
    )

    param_grid = build_param_grid(args.strength, args.num_steps, args.control_scale, args.seed)
    logger.info("Running %d hyperparameter configuration(s).", len(param_grid))

    k3d_wrapper = init_minimum_control_net_wrapper_from_config(str(args.config))

    payloads = load_bundle_payloads(
        entries,
        k3d_wrapper,
        control_modes,
        args.downscale,
        args.blur_kernel,
        args.blur_sigma,
        args.save_cond_images,
    )

    output_base = ensure_output_dir(args.output_dir)
    flux_device = k3d_wrapper.config["flux"].get("device", "cpu")

    for end_idx, control_guidance_end in enumerate(control_guidance_end_sets):
        end_tag = "ce" + "-".join(_fmt_float(val) for val in control_guidance_end)
        for strength, steps, ctrl_scale, seed in param_grid:
            ctrl_scale_vector = [ctrl_scale * mode_scale for mode_scale in control_mode_scale]
            combo_name = "_".join(
                [
                    f"str{_fmt_float(strength)}",
                    f"steps{steps}",
                    f"ctrl{_fmt_float(ctrl_scale)}",
                    f"seed{seed}",
                    end_tag,
                ]
            )
            combo_dir = output_base / combo_name
            combo_dir.mkdir(parents=True, exist_ok=True)

            logger.info(
                "Running ControlNet edit with strength=%.3f, steps=%d, ctrl_scale=%.3f, seed=%d, guidance_end=%s",
                strength,
                steps,
                ctrl_scale,
                seed,
                control_guidance_end,
            )

            seed_everything(seed)

            for payload in payloads:
                k3d_wrapper.renew_uuid()
                entry = payload.entry
                logger.info("   -> %s", entry.name)

                bundle_tensor = payload.tensor.unsqueeze(0).to(flux_device)

                gen_bundle = k3d_wrapper.generate_3d_bundle_image_controlnet(
                    prompt=entry.prompt,
                    image=bundle_tensor,
                    strength=strength,
                    control_image=payload.control_images,
                    control_mode=control_modes,
                    control_guidance_start=control_guidance_start,
                    control_guidance_end=control_guidance_end,
                    controlnet_conditioning_scale=ctrl_scale_vector,
                    num_inference_steps=steps,
                    guidance_scale=args.guidance_scale,
                    lora_scale=args.lora_scale,
                    seed=seed,
                    save_intermediate_results=False,
                )

                sample_dir = combo_dir / entry.name
                sample_dir.mkdir(parents=True, exist_ok=True)

                dst_path = sample_dir / f"{entry.name}_controlnet.png"
                vutils.save_image(gen_bundle.clamp(0.0, 1.0), dst_path)
                logger.info("Saved ControlNet edit to %s", dst_path)

            meta = {
                "strength": strength,
                "num_steps": steps,
                "control_mode": control_modes,
                "control_guidance_start": control_guidance_start,
                "control_guidance_end": control_guidance_end,
                "control_scale": ctrl_scale_vector,
                "guidance_scale": args.guidance_scale,
                "lora_scale": args.lora_scale,
                "downscale": args.downscale,
                "blur_kernel": args.blur_kernel,
                "blur_sigma": args.blur_sigma,
                "seed": seed,
                "entries": [
                    {
                        "name": payload.entry.name,
                        "image_path": str(payload.entry.image_path),
                        "prompt": payload.entry.prompt,
                    }
                    for payload in payloads
                ],
            }
            with open(combo_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
