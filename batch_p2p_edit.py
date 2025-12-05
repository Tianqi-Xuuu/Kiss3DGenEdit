#!/usr/bin/env python3
"""
Batch Prompt-to-Prompt editing utility.

Usage example:
    python batch_p2p_edit.py \
        --prompt-root ./test_p2p \
        --p2p-tau 0.2 0.4 \
        --p2p-edit-mode qk_img qk_attn

The script expects `--prompt-root` to contain one or more folders. Each folder
must provide both a `src_prompt.txt` and `tgt_prompt.txt` (alternative file name
variants such as `prompt_src.txt` are also accepted). Every discovered entry is
processed across all requested hyperparameter combinations and saved to
dedicated sub-folders.
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

from pipeline.kiss3d_wrapper import init_minimum_wrapper_from_config, run_edit_3d_bundle_p2p
from pipeline.utils import logger


SRC_PROMPT_NAMES = ("src_prompt.txt", "prompt_src.txt", "src.txt")
TGT_PROMPT_NAMES = ("tgt_prompt.txt", "prompt_tgt.txt", "tgt.txt")


@dataclass
class PromptEntry:
    name: str
    src_prompt_path: Path
    tgt_prompt_path: Path
    src_prompt: str
    tgt_prompt: str


def _fmt_float(value: float) -> str:
    """Create a readable float string for folder names."""
    if float(int(value)) == float(value):
        return str(int(value))
    text = f"{value:.4f}".rstrip("0")
    return text.rstrip(".")


def _find_prompt_file(folder: Path, candidate_names: Iterable[str]) -> Path | None:
    for name in candidate_names:
        candidate = folder / name
        if candidate.is_file():
            return candidate
    return None


def _read_prompt(path: Path, role: str) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{role} prompt inside {path} is empty.")
    return text


def discover_prompt_entries(prompt_root: Path) -> List[PromptEntry]:
    """
    Discover Prompt-to-Prompt entries inside prompt_root.

    Every valid entry is either prompt_root itself or one of its direct child
    folders that contains both a src_prompt.txt and tgt_prompt.txt file.
    """
    if not prompt_root.exists():
        raise FileNotFoundError(f"Prompt root '{prompt_root}' does not exist.")

    entries: List[PromptEntry] = []

    def _collect_from_folder(folder: Path) -> None:
        src_path = _find_prompt_file(folder, SRC_PROMPT_NAMES)
        tgt_path = _find_prompt_file(folder, TGT_PROMPT_NAMES)
        if src_path is None or tgt_path is None:
            return

        src_prompt = _read_prompt(src_path, "Source")
        tgt_prompt = _read_prompt(tgt_path, "Target")
        entries.append(
            PromptEntry(
                name=folder.name,
                src_prompt_path=src_path,
                tgt_prompt_path=tgt_path,
                src_prompt=src_prompt,
                tgt_prompt=tgt_prompt,
            )
        )

    _collect_from_folder(prompt_root)
    for child in sorted(prompt_root.iterdir()):
        if child.is_dir():
            _collect_from_folder(child)

    if not entries:
        raise ValueError(
            f"No folders with both src/tgt prompt files were found under {prompt_root}."
        )
    return entries


def build_param_grid(edit_modes: Sequence[str], taus: Sequence[float]) -> List[Tuple[str, float]]:
    return list(itertools.product(edit_modes, taus))


def ensure_output_dir(base: Path) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"p2p_batch_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch Prompt-to-Prompt editing from text prompts.")
    parser.add_argument(
        "--prompt-root",
        default=Path("./test_p2p_short_identity"),
        type=Path,
        help="Directory containing folders with src_prompt.txt and tgt_prompt.txt files.",
    )
    parser.add_argument(
        "--config",
        default=Path("./pipeline/pipeline_config/default.yaml"),
        type=Path,
        help="Pipeline config used to initialize the Kiss3D wrapper.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("./outputs/p2p_batch_results"),
        type=Path,
        help="Base directory to store Prompt-to-Prompt images.",
    )
    parser.add_argument(
        "--p2p-tau",
        nargs="+",
        type=float,
        default=[0.15, 0.2],
        help="List of Prompt-to-Prompt tau values.",
    )
    parser.add_argument(
        "--p2p-edit-mode",
        nargs="+",
        type=str,
        default=["qk_img", "qk_full" , "full"],
        help="List of Prompt-to-Prompt edit modes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt_root: Path = args.prompt_root

    entries = discover_prompt_entries(prompt_root)
    logger.info("Discovered %d Prompt-to-Prompt job(s) under %s.", len(entries), prompt_root)
    for entry in entries:
        logger.info(" - %s", entry.name)

    param_grid = build_param_grid(args.p2p_edit_mode, args.p2p_tau)
    logger.info("Running %d hyperparameter configuration(s).", len(param_grid))

    k3d_wrapper = init_minimum_wrapper_from_config(str(args.config))

    output_base = ensure_output_dir(args.output_dir)
    logger.info("Saving outputs to %s", output_base)

    for edit_mode, tau in param_grid:
        sanitized_mode = edit_mode.replace("/", "-")
        combo_name = "_".join(
            [
                f"mode_{sanitized_mode}",
                f"tau{_fmt_float(tau)}",
            ]
        )
        combo_dir = output_base / combo_name
        combo_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Running Prompt-to-Prompt edit with mode=%s, tau=%s", edit_mode, tau)

        for entry in entries:
            logger.info("   -> %s", entry.name)
            _, _, save_path_src, save_path_tgt = run_edit_3d_bundle_p2p(
                k3d_wrapper,
                prompt_src=entry.src_prompt,
                prompt_tgt=entry.tgt_prompt,
                p2p_edit_mode=edit_mode,
                p2p_tau=tau,
            )

            sample_dir = combo_dir / entry.name
            sample_dir.mkdir(parents=True, exist_ok=True)

            dst_src = sample_dir / f"{entry.name}_src_bundle.png"
            dst_tgt = sample_dir / f"{entry.name}_tgt_bundle.png"
            shutil.copy2(save_path_src, dst_src)
            shutil.copy2(save_path_tgt, dst_tgt)

            logger.info("Saved results to %s and %s", dst_src, dst_tgt)

        meta = {
            "p2p_tau": tau,
            "p2p_edit_mode": edit_mode,
            "entries": [
                {
                    "name": entry.name,
                    "src_prompt_path": str(entry.src_prompt_path),
                    "tgt_prompt_path": str(entry.tgt_prompt_path),
                    "src_prompt": entry.src_prompt,
                    "tgt_prompt": entry.tgt_prompt,
                }
                for entry in entries
            ],
        }
        with open(combo_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
