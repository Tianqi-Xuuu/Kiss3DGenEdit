import os
import glob
import argparse
import shutil
import torch
import torchvision
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from models.lrm.online_render.render_single import render_mesh
from pipeline.kiss3d_wrapper import init_wrapper_from_config, run_text_to_3d, run_image_to_3d
import matplotlib.pyplot as plt
from diffusers import FluxPipeline
import numpy as np

os.environ["IMAGEIO_USERDIR"] = "/ocean/projects/cis250190p/stang6/imageio"

# from pipeline.utils import render_3d_bundle_image_from_mesh  # Unused import removed for cleanliness


def find_mesh_in_folder(folder):
    """
    Find the first mesh file in the given folder.
    Supported formats: .obj, .glb, .ply
    """
    patterns = ["**/*.obj", "**/*.glb", "**/*.ply"]
    for pat in patterns:
        files = glob.glob(os.path.join(folder, pat), recursive=True)
        if files:
            return files[0]
    return None

def save_combined_figure(images_list, scale_list, save_path, title, prompt):
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
                    gidx = i - 1
                    if gidx < len(scale_list):
                        ax.set_title(f'{title}: {scale_list[gidx]}')
                    else:
                        ax.set_title('')
                ax.axis('off')
            else:
                ax.axis('off')
        # leave space at top for the suptitle
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gso_root",
        default="./GSO",
        type=str,
        help="GSO directory root (ignored for hardcoded mesh_paths unless you modify mesh_paths).",
    )
    parser.add_argument(
        "--out_root",
        default="./GSO_bundles_RF",
        type=str,
        help="Bundle output directory, e.g., ./GSO_bundles_RF",
    )
    args = parser.parse_args()

    enable_redux = True
    use_mv_rgb = True
    use_controlnet = True

    os.makedirs(args.out_root, exist_ok=True)

    pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
    custom_pipeline="pipeline_flux_rf_inversion")

    # Move pipeline to GPU if available; otherwise stay on CPU (will be slower)
    if torch.cuda.is_available():
        pipe.to("cuda")
    else:
        print("[WARN] CUDA not available; running on CPU.")

    prompts = {
        'school_bus': ["A 3D green school bus", "A 3D blue taxi",],
        'shark': ["A 3D great pink shark", "A 3D orca",]
    }

    mesh_paths = {
        'school_bus': './GSO/Sonny_School_Bus/',
        'shark': './GSO/Weisshai_Great_White_Shark/',
    }

    eta_values = [0.5, 0.65, 0.75, 0.85, 0.9]
    guidance_scale = [1, 2, 3, 4, 5]

    print("Initializing Kiss3D wrapper...")
    k3d_wrapper = init_wrapper_from_config('./pipeline/pipeline_config/default.yaml')

    # for name in sorted(os.listdir(args.gso_root)):
    for name, folder in mesh_paths.items():

        mesh_path = folder+'/meshes/object.obj'

        renderings = render_mesh(mesh_path, save_dir=None)
        rgbs = renderings['rgb'][...,:3].cpu().permute(0, 3, 1, 2)
        input_image = rgbs[0].unsqueeze(0)

        count = 0

        for prompt in prompts[name]:
            TMP_DIR = os.path.join(args.out_root, name, f"prompt_{count}")
            os.makedirs(TMP_DIR, exist_ok=True)

            # prepare guidance/eta subfolders
            guidance_dir = os.path.join(TMP_DIR, 'guidance')
            eta_dir = os.path.join(TMP_DIR, 'eta')
            os.makedirs(guidance_dir, exist_ok=True)
            os.makedirs(eta_dir, exist_ok=True)

            inverted_latents, image_latents, latent_image_ids = pipe.invert(
                image=input_image, 
                num_inversion_steps=30, 
                gamma=0.5
            )

            # Convert first rendering to PIL and pre-seed lists
            pil_input_image = to_pil_image(rgbs[0])  # rgbs[0] shape (3,H,W)
            guidance_list = [pil_input_image]
            eta_list = [pil_input_image]

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
                img.save(os.path.join(guidance_dir, f'{guidance}_3D_input_image.png'))
                guidance_list.append(img)
            
            for eta in eta_values:
                img = pipe(
                    prompt=prompt,
                    inverted_latents=inverted_latents,
                    image_latents=image_latents,
                    latent_image_ids=latent_image_ids,
                    start_timestep=0,
                    stop_timestep=7/30,
                    num_inference_steps=30,
                    guidance_scale=3,
                    eta=eta,
                ).images[0]

                img.save(os.path.join(eta_dir, f'{eta}_3D_input_image.png'))
                eta_list.append(img)

            # Save the original input view for reference
            input_save_path = os.path.join(TMP_DIR, 'input_image.png')
            pil_input_image.save(input_save_path)

            # save the combined guidance scale figur
            save_combined_figure(
                guidance_list,
                guidance_scale,
                os.path.join(TMP_DIR, 'combined_guidance.png'),
                title='guidance',
                prompt=prompt
            )
            # save the combined eta figure
            save_combined_figure(
                eta_list,
                eta_values,
                os.path.join(TMP_DIR, 'combined_eta.png'),
                title='eta',
                prompt=prompt
            )

            # read one for bundle generation
            bundle_image_path = os.path.join(guidance_dir, f'{guidance_scale[2]}_3D_input_image.png')

            gen_save_path, recon_mesh_path = run_image_to_3d(
                k3d_wrapper,
                bundle_image_path,
                enable_redux,
                use_mv_rgb,
                use_controlnet
            )
            image_to_3d_dir = os.path.join(TMP_DIR, 'image_to_3d')
            os.makedirs(image_to_3d_dir, exist_ok=True)
            # Copy generated artifacts
            shutil.copy2(gen_save_path, os.path.join(image_to_3d_dir, f'{name}_{count}_3d_bundle.png'))
            shutil.copy2(recon_mesh_path, os.path.join(image_to_3d_dir, f'{name}_{count}.glb'))

            count += 1

    print("[DONE] All meshes processed.")


if __name__ == "__main__":
    main()
