from pipeline.kiss3d_wrapper import init_wrapper_from_config, run_edit_3d_bundle_p2p, run_edit_3d_bundle_rf, init_minimum_wrapper_from_config
import os
from pipeline.utils import logger, TMP_DIR, OUT_DIR
import time
import shutil

if __name__ == "__main__":
    os.makedirs(os.path.join(OUT_DIR, 'text_to_3d'), exist_ok=True)
    k3d_wrapper = init_minimum_wrapper_from_config('./pipeline/pipeline_config/default.yaml')
    src_prompt = 'A charming 3D doll of a young girl styled as a Hogwarts student, ' \
                'with a whimsical, magical flair. Front view features a radiant smile, ' \
                'large, twinkling eyes, and a Gryffindor scarf with a cozy, fluffy texture; ' \
                'left side view showcases a miniature wand tucked behind her ear, ' \
                'with a tiny golden snitch hanging from a string; ' \
                'rear view presents a backpack adorned with house badges and a small owl figurine; ' \
                'right side view displays a playful broomstick with feathers fluttering gently, ' \
                'alongside a tiny house elf mascot. No background. ' \
                'Arranged in a 2x4 grid with RGB images on top and normal maps below.'
    tgt_prompt = 'A charming 3D doll of a young girl styled as a Hogwarts student, ' \
                'with a whimsical, magical flair. Front view features a sad expression, ' \
                'large, twinkling eyes, and a Gryffindor scarf with a cozy, fluffy texture; ' \
                'left side view showcases a miniature wand tucked behind her ear, ' \
                'with a tiny golden snitch hanging from a string; ' \
                'rear view presents a backpack adorned with house badges and a small owl figurine; ' \
                'right side view displays a playful broomstick with feathers fluttering gently, ' \
                'alongside a tiny house elf mascot. No background.' \
                'Arranged in a 2x4 grid with RGB images on top and normal maps below.'
    name = "doll_girl"
    os.system(f'rm -rf {TMP_DIR}/*')
    end = time.time()
    p2p_tau = 0.2
    # 创建 token heatmap 保存目录
    token_heatmap_dir = os.path.join(OUT_DIR, 'token_heatmaps', f'{name}_tau{p2p_tau}')
    os.makedirs(token_heatmap_dir, exist_ok=True)

    # 创建 layer heatmap 保存目录
    layer_heatmap_dir = os.path.join(OUT_DIR, 'layer_heatmaps', f'{name}_tau{p2p_tau}')
    os.makedirs(layer_heatmap_dir, exist_ok=True)

    # 使用 T2I mask 进行编辑
    result = run_edit_3d_bundle_p2p(
        k3d_wrapper,
        prompt_src=src_prompt,
        prompt_tgt=tgt_prompt,
        p2p_tau=p2p_tau,
        # T2I Mask 参数
        use_t2i_mask=False,         # 设为 True 则启用局部混合
        t2i_mask_blocks=[0, 1, 2, 3, 4],  # 使用前5个 MMDiT 块
        t2i_mask_threshold=0.5,
        t2i_mask_sigma=2.0,
        return_mask=True,           # 返回 mask 用于可视化
        per_view_mask=False,
        # Per-token heatmap 参数
        return_per_token_heatmaps=True,  # 为每个 token 生成热力图
        per_token_heatmap_dir=token_heatmap_dir,  # 保存目录
        # Per-layer heatmap 参数
        return_per_layer_heatmaps=True,  # 为每一层生成热力图（每隔3层 + 全局平均）
        per_layer_heatmap_dir=layer_heatmap_dir,  # 保存目录
        layer_heatmap_interval=3,  # 每隔3层绘制（0,3,6,9,12,15,18）
    )

    # 解析返回值
    if len(result) == 6:
        src_img, tgt_img, src_save_path, tgt_save_path, t2i_mask, mask_save_path = result
        print(f"T2I mask saved to: {mask_save_path}")
        print(f"T2I mask shape: {t2i_mask.shape if t2i_mask is not None else 'None'}")
    else:
        src_img, tgt_img, src_save_path, tgt_save_path = result

    print(f"P2P edit_3d_bundle time: {time.time() - end}")
    print(f"Token heatmaps saved to: {token_heatmap_dir}")
    print(f"Layer heatmaps saved to: {layer_heatmap_dir}")

    save_dir = os.path.join("examples", 'final_edit_3d')
    os.makedirs(save_dir, exist_ok=True)

    timestamp = int(time.time())

    src_dst = os.path.join(
        save_dir, f"{name}_tau{p2p_tau}_src_3d_bundle_{timestamp}.png"
    )
    tgt_dst = os.path.join(
        save_dir, f"{name}_tau{p2p_tau}_tgt_3d_bundle_{timestamp}.png"
    )
    shutil.copyfile(src_save_path, src_dst)
    shutil.copyfile(tgt_save_path, tgt_dst)

    # 保存 mask
    if len(result) == 6:
        mask_dst = os.path.join(
            save_dir, f"{name}_tau{p2p_tau}_t2i_mask_{timestamp}.png"
        )
        shutil.copyfile(mask_save_path, mask_dst)

    end = time.time()

    src_img, tgt_img, src_save_path, tgt_save_path = run_edit_3d_bundle_rf(k3d_wrapper,
                                                        bundle_img=src_img,
                                                        prompt_tgt=tgt_prompt,
                                                        rf_gamma=0.6,
                                                        rf_eta=0.8,
                                                        rf_stop=0.8,
                                                        num_steps=20,
                                                        guidance_scale=2.0,
                                                    )

    print(f"RF edit_3d_bundle time: {time.time() - end}")

    src_dst = os.path.join(
        save_dir, f"{name}_rf_gamma0.6_eta0.8_stop0.8_tgt_3d_bundle_{timestamp}.png"
    )
    shutil.copyfile(tgt_save_path, src_dst)

    print("Saved edited 3D bundle images to:", save_dir)
