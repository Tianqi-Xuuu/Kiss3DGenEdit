from pathlib import Path
from typing import List
from PIL import Image


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


def save_bundle_views(image_path: str, rows: int = 2, cols: int = 4) -> None:
    """
    Load a bundle image, split into rows x cols views,
    and save them into a folder named after the image file.
    """
    img_path = Path(image_path)
    bundle_img = Image.open(img_path)

    # Folder name = file stem (no extension), next to the original image
    out_dir = img_path.parent / img_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    views = split_bundle_image(bundle_img, rows, cols)

    for i, view in enumerate(views):
        out_file = out_dir / f"{img_path.stem}_view_{i:02d}.png"
        view.save(out_file)
        print(f"Saved view {i} to {out_file}")



save_bundle_views("examples/midway_edit_3d/cat2corgi/cat2corgi_mode_qk_img_tau0.3_tgt_3d_bundle_1764021554.png", rows=2, cols=4)
