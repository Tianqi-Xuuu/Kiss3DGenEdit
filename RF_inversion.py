import torch
from diffusers import FluxPipeline
import requests
import PIL
from io import BytesIO
import os
from pipeline.utils import lrm_reconstruct, isomer_reconstruct, preprocess_input_image

# from pipeline.utils import logger, TMP_DIR, OUT_DIR
from PIL import Image
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


input_image_path = './examples_RF/cartoon_dinosaur.png'
prompt = "A 3D girl with red hat and blue dress"

TMP_DIR = './outputs/tmp/guidance_experiment_rf_inversion'

os.makedirs(TMP_DIR, exist_ok=True)

input_image = preprocess_input_image(Image.open(input_image_path))

# use RF inversion to get latents
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
    custom_pipeline="pipeline_flux_rf_inversion")

pipe.to("cuda")

inverted_latents, image_latents, latent_image_ids = pipe.invert(
image=input_image, 
num_inversion_steps=30, 
gamma=0.5
)

# eta_values = [0.5, 0.65, 0.75, 0.85, 0.9]
guidance_scale = [1, 2, 3, 4, 5]
# start with an empty list for generated PIL images
images_list = []

# record the prompt and some metadata to a text file for reproducibility
prompt_path = os.path.join(TMP_DIR, 'prompt.txt')
with open(prompt_path, 'w', encoding='utf-8') as pf:
    pf.write(f'prompt: {prompt}\n')
    pf.write(f'guidance_scale: {guidance_scale}\n')
    pf.write('method: RF inversion via FluxPipeline\n')
    pf.write('\n')
    pf.write('NOTE: This file records the prompt used to generate the images.\n')

for guidance in guidance_scale:
    img = pipe(
        prompt=prompt,
        inverted_latents=inverted_latents,
        image_latents=image_latents,
        latent_image_ids=latent_image_ids,
        start_timestep=0,
        stop_timestep=7/30,
        num_inference_steps=30,
        guidance_scale=guidance,
        eta=0.7,
    ).images[0]

    # save individual image (keeps original behavior)
    img.save(os.path.join(TMP_DIR, f'{guidance}_3D_input_image.png'))
    images_list.append(img)

# save the original input image separately and include it in the combined view
input_save_path = os.path.join(TMP_DIR, 'input_image.png')
try:
    # input_image is preprocessed; convert back to PIL.Image if needed
    if isinstance(input_image, Image.Image):
        input_image.save(input_save_path)
        images_list.insert(0, input_image)
    else:
        # if preprocess_input_image returned a tensor/array, try converting
        pil_in = Image.fromarray(np.asarray(input_image))
        pil_in.save(input_save_path)
        images_list.insert(0, pil_in)
except Exception:
    # non-fatal: continue without adding input image
    pass

# Save all generated images into a single combined figure (grid)
if len(images_list) > 0:
    n = len(images_list)
    # Force a single-row layout: all subfigures in one horizontal line
    cols = max(1, n)
    rows = 1

    # choose a reasonable figsize (3 inches per image)
    # Warning: very large n will create a wide figure; adjust multiplier if needed
    fig_w = cols * 3
    fig_h = 3
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))

    # flatten axes for easy indexing (handles single-row/single-col cases)
    if isinstance(axes, np.ndarray):
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    # add the prompt as a suptitle, leaving room at top of the figure
    plt.suptitle(prompt, fontsize=10, wrap=True)

    for i in range(rows * cols):
        ax = axes_flat[i]
        if i < n:
            # convert PIL image to numpy array for imshow
            ax.imshow(np.asarray(images_list[i]))
            # label first image as input, remaining with guidance values
            if i == 0:
                ax.set_title('input image')
            else:
                # guidance index is i-1 because input occupies index 0
                gidx = i - 1
                if gidx < len(guidance_scale):
                    ax.set_title(f'guidance: {guidance_scale[gidx]}')
                else:
                    ax.set_title('')
            ax.axis('off')
        else:
            ax.axis('off')
    # leave space at top for the suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    combined_path = os.path.join(TMP_DIR, 'combined_guidance.png')
    plt.savefig(combined_path, dpi=150)
    plt.close(fig)

    # Also save each image as a page in a multipage PDF for convenience
    pdf_path = os.path.join(TMP_DIR, 'all_figures.pdf')
    try:
        with PdfPages(pdf_path) as pdf:
            for i, im in enumerate(images_list):
                fig_page = plt.figure(figsize=(8, 8))
                plt.imshow(np.asarray(im))
                plt.axis('off')
                title = 'input image' if i == 0 else f'guidance: {guidance_scale[i-1] if i-1 < len(guidance_scale) else ""}'
                plt.title(title)
                pdf.savefig(fig_page, bbox_inches='tight')
                plt.close(fig_page)
    except Exception:
        # if matplotlib PDF backend not available, skip silently
        pass

# input_image = pipe(
#     prompt=prompt,
#     inverted_latents=inverted_latents,
#     image_latents=image_latents,
#     latent_image_ids=latent_image_ids,
#     start_timestep=0,
#     stop_timestep=15/30,
#     num_inference_steps=30,
#     guidance_scale=5,
#     eta=0.9,
# ).images[0]

# input_image.save(os.path.join(TMP_DIR, f'0.6_3D_input_image.png'))
