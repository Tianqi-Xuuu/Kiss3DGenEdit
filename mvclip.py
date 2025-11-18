import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Union, Tuple

import matplotlib.pyplot as plt
from PIL import Image
import ImageReward as RM  # pip install ImageReward


def split_bundle_image(bundle_img: Image.Image, rows: int, cols: int) -> List[Image.Image]:
    """Cut a bundle image into rows x cols tiles (views)."""
    w, h = bundle_img.size
    tile_w = w // cols
    tile_h = h // rows

    views = []
    for r in range(rows):
        for c in range(cols):
            left = c * tile_w
            upper = r * tile_h
            right = left + tile_w
            lower = upper + tile_h
            tile = bundle_img.crop((left, upper, right, lower))
            views.append(tile)
    return views


def default_view_suffixes_8() -> List[str]:
    """Default 8-view suffixes: 4 natural-color + 4 geometry-only."""
    return [
        # 4 natural-color / cartoon-like views
        "Cartoon-styled rendering, left side view, on a clean white background.",
        "Cartoon-styled rendering, back view, on a clean white background.",
        "Cartoon-styled rendering, right side view, on a clean white background.",
        "Cartoon-styled rendering, front view, on a clean white background.",

        # 4 geometry-only pseudo-color views
        "Geometry-only rendering with smooth rainbow colors, left side view, on a clean white background.",
        "Geometry-only rendering with smooth rainbow colors, back view, on a clean white background.",
        "Geometry-only rendering with smooth rainbow colors, right side view, on a clean white background.",
        "Geometry-only rendering with smooth rainbow colors, front view, on a clean white background.",
    ]


class MVClipEvaluator:
    """
    Multi-view scoring wrapper using ImageReward models (ImageReward / CLIP / BLIP / Aesthetic).

    Parameters
    ----------
    model_type : {"ImageReward", "CLIP", "BLIP", "Aesthetic"}, default "ImageReward"
        Which scoring model to load from ImageReward.
        - "ImageReward": preference RM, expects PIL.Image
        - "CLIP", "BLIP", "Aesthetic": score models, expect image paths.
    rows : int, default 2
        Number of rows in the bundle grid.
    cols : int, default 4
        Number of columns in the bundle grid.
    show_plot_default : bool, default False
        Default whether to show matplotlib plots in `evaluate_bundle`.
    save_dir : str or None, default None
        If not None, save the visualization figures into this directory.

    Attributes
    ----------
    model : object
        Underlying scoring model instance with a `.score(prompt, image_or_path)` method.
    """

    def __init__(
        self,
        model_type: str = "ImageReward",
        rows: int = 2,
        cols: int = 4,
        save_dir: str = "MVClip_Eval_Results",
    ):
        self.model_type = model_type
        self.rows = rows
        self.cols = cols
        self.save_dir = save_dir

        # Load model once
        if model_type == "ImageReward":
            print("Loading ImageReward-v1.0 (BLIP-based reward model)...")
            self.model = RM.load("ImageReward-v1.0")
        elif model_type == "CLIP":
            print("Loading CLIPScore model...")
            self.model = RM.load_score("CLIP")
        elif model_type == "BLIP":
            print("Loading BLIPScore model...")
            self.model = RM.load_score("BLIP")
        elif model_type == "Aesthetic":
            print("Loading AestheticScore model...")
            self.model = RM.load_score("Aesthetic")
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        
        print("Model loaded.")

        # Ensure save_dir exists if provided
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

    def evaluate_bundle(
        self,
        bundle: Union[str, Path, Image.Image],
        base_prompt: str,
        view_suffixes: List[str] | None = None,
        top_k : int | None = None,
        verbose_scores: bool = False,
        save_plot: bool = False,
        figure_name: str | None = None,
    ) -> Tuple[float, List[float]]:
        """
        Parameters
        ----------
        bundle : str or pathlib.Path or PIL.Image.Image
            Input bundle image. Can be a file path or an in-memory PIL image.
        base_prompt : str
            Base text description (e.g., "A dog. "). Per-view suffixes will be
            appended to this string to form the final prompts.
        top_k : int, optional
            If specified, average only the top_k highest view scores.
        verbose_scores : bool, optional
            If True, print per-view scores during evaluation.
        view_suffixes : List[str], optional
            List of textual suffixes, one per view, appended to `base_prompt`
            to form the per-view prompts. If None, a default set will be used
            when the number of views is supported (e.g., 8).
        show_plot : bool, optional
            If True, display a matplotlib figure showing the grid of views
            and their scores. If None, falls back to `self.show_plot_default`.
        figure_name : str, optional
            If not None and `self.save_dir` is set, save the figure as this name
            (e.g., "sample_001.png") under `self.save_dir`.

        Returns
        -------
        avg_score : float
            The average score across all views.
        scores : List[float]
            A list of scores, one per view, in the same order as the views.
        """

        # 1. Load bundle image
        if isinstance(bundle, (str, Path)):
            bundle_img = Image.open(bundle).convert("RGB")
        elif isinstance(bundle, Image.Image):
            bundle_img = bundle.convert("RGB")
        else:
            raise TypeError("bundle must be str, Path, or PIL.Image.Image")

        # 2. Split into views
        views = split_bundle_image(bundle_img, rows=self.rows, cols=self.cols)
        n_views = len(views)

        # 3. Prepare per-view prompts
        if view_suffixes is None:
            if n_views == 8:
                view_suffixes = default_view_suffixes_8()
            else:
                view_suffixes = ["" for _ in range(n_views)]

        if len(view_suffixes) != n_views:
            raise ValueError(
                f"view_suffixes length {len(view_suffixes)} does not match number of views {n_views}"
            )

        prompts = [base_prompt + s for s in view_suffixes]

        scores: List[float] = []

        # 4. Scoring logic depends on model_type
        if self.model_type == "ImageReward":
            # ImageReward expects PIL.Image directly
            for i, (prompt, img) in enumerate(zip(prompts, views)):
                score = self.model.score(prompt, img)
                scores.append(score)
                if verbose_scores:
                    print(f"[ImageReward] View {i:02d} score: {score:.4f}")
        else:
            # CLIP / BLIP / Aesthetic expect image paths
            tmp_dir = tempfile.mkdtemp(prefix="mvclip_views_")
            try:
                view_paths = []
                for i, img in enumerate(views):
                    path = os.path.join(tmp_dir, f"view_{i:02d}.png")
                    img.save(path)
                    view_paths.append(path)

                for i, (prompt, path) in enumerate(zip(prompts, view_paths)):
                    score = self.model.score(prompt, path)
                    scores.append(score)
                    if verbose_scores:
                        print(f"[{self.model_type}] View {i:02d} score: {score:.4f}")
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        # 5. Average score
        if top_k is not None and 1 <= top_k < len(scores):
            top_scores = sorted(scores, reverse=True)[:top_k]
            avg_score = sum(top_scores) / top_k
        else:
            avg_score = sum(scores) / len(scores) if scores else float("nan")

        print("-" * 40)
        print(f"Base prompt: {base_prompt}")
        print(f"Average {self.model_type} score across {len(scores)} views: {avg_score:.4f}")

        # 6. Visualization
        if  self.save_dir is not None and save_plot:

            if figure_name is None:
                import time
                figure_name = f"mvclip_eval_{int(time.time())}.png"

            fig, axes = plt.subplots(self.rows, self.cols, figsize=(4 * self.cols, 4 * self.rows))
            axes = axes.flatten() if self.rows * self.cols > 1 else [axes]

            for i, ax in enumerate(axes[:n_views]):
                ax.imshow(views[i])
                ax.axis("off")
                ax.set_title(f"{self.model_type} score: {scores[i]:.2f}", fontsize=10)

            fig.suptitle(base_prompt, fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.93])

            save_path = os.path.join(self.save_dir, figure_name)
            fig.savefig(save_path, dpi=150)
            print(f"Figure saved to: {save_path}")

        return avg_score, scores

if __name__ == "__main__":
    # Example usage
    evaluator = MVClipEvaluator(model_type="ImageReward")

    bundle_path = "examples/dog_3d_bundle_image.png" 
    base_prompt = "A cartoon dog standing on four legs, smiling with its mouth open."

    avg_score, scores = evaluator.evaluate_bundle(
        bundle=bundle_path,
        base_prompt=base_prompt,
        save_plot=False,
        top_k=6
    )

    print(f"Final average score: {avg_score:.4f}")