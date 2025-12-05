"""
T2I Mask 生成工具函数

基于 FluxEdit 论文 (arXiv 2508.07519) 实现：
- 从 T2I (text-to-image) 注意力图生成编辑 mask
- 支持阈值化和高斯平滑
- 支持 3D Bundle (2x4 多视角) 的统一/独立 mask
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def gaussian_blur_2d(
    x: torch.Tensor,
    kernel_size: int = 5,
    sigma: float = 2.0,
) -> torch.Tensor:
    """
    对 2D tensor 应用高斯模糊

    Args:
        x: [batch, channels, height, width] 的 tensor
        kernel_size: 高斯核大小（奇数）
        sigma: 高斯核标准差

    Returns:
        模糊后的 tensor
    """
    if kernel_size % 2 == 0:
        kernel_size += 1

    # 创建 1D 高斯核
    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - kernel_size // 2
    kernel_1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()

    # 创建 2D 高斯核
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
    kernel_2d = kernel_2d.expand(x.shape[1], 1, kernel_size, kernel_size)

    # 应用高斯模糊
    padding = kernel_size // 2
    x_blurred = F.conv2d(x, kernel_2d, padding=padding, groups=x.shape[1])

    return x_blurred


def generate_t2i_mask(
    t2i_attn_cache: Dict[str, torch.Tensor],
    selected_blocks: Optional[List[int]] = None,
    threshold: float = 0.5,
    gaussian_sigma: float = 2.0,
    gaussian_kernel_size: int = 5,
    height: int = 64,
    width: int = 128,
    per_view_mask: bool = False,
    num_views: int = 8,
) -> torch.Tensor:
    """
    从 T2I 注意力图生成编辑 mask

    基于 FluxEdit 论文，使用 q_image @ k_text.T 计算的注意力图
    来确定编辑区域。

    Args:
        t2i_attn_cache: 各层的 T2I 注意力图，key 格式为 "mmdit_{idx}"
                        value 形状为 [batch, heads, img_len, txt_len]
        selected_blocks: 要使用的 MMDiT 块索引列表，默认 [0,1,2,3,4]
        threshold: 二值化阈值，范围 [0, 1]
        gaussian_sigma: 高斯平滑的标准差
        gaussian_kernel_size: 高斯核大小
        height: 输出 mask 的高度（latent 空间）
        width: 输出 mask 的宽度（latent 空间）
        per_view_mask: 是否为每个视角独立计算 mask（用于 3D Bundle）
        num_views: 视角数量（用于 3D Bundle，默认 8 = 2x4）

    Returns:
        mask: [1, 1, height, width] 的 mask tensor，值域 [0, 1]
    """
    if selected_blocks is None:
        selected_blocks = [0, 1, 2, 3, 4]

    # 1. 收集选定块的注意力图
    attn_maps = []
    for idx in selected_blocks:
        key = f"mmdit_{idx}"
        if key in t2i_attn_cache:
            attn_maps.append(t2i_attn_cache[key])

    if len(attn_maps) == 0:
        # 如果没有找到任何注意力图，返回全1 mask
        print(f"Warning: No T2I attention maps found for blocks {selected_blocks}. Available keys: {list(t2i_attn_cache.keys())}")
        return torch.ones(1, 1, height, width)

    # 2. 堆叠并平均所有块的注意力
    # attn_maps[i]: [batch, heads, img_len, txt_len]
    stacked = torch.stack(attn_maps, dim=0)  # [num_blocks, batch, heads, img_len, txt_len]
    avg_attn = stacked.mean(dim=0)  # [batch, heads, img_len, txt_len]

    # 3. 平均所有 heads
    avg_attn = avg_attn.mean(dim=1)  # [batch, img_len, txt_len]

    # 4. 取每个 patch 对所有 text tokens 的最大注意力
    # 注意：不能用 sum，因为 softmax 后每行和=1，sum 会导致所有 patch 值相同
    patch_attn = avg_attn.max(dim=-1)[0]  # [batch, img_len]

    # 5. 计算 latent 空间的尺寸
    # Flux 使用 2x2 packing，所以实际 patch 数 = (height/2) * (width/2)
    latent_h = height // 2
    latent_w = width // 2
    expected_patches = latent_h * latent_w

    actual_patches = patch_attn.shape[-1]

    if actual_patches != expected_patches:
        print(f"Warning: patch count mismatch. Expected {expected_patches}, got {actual_patches}. "
              f"Attempting to infer correct dimensions...")
        # 尝试推断正确的尺寸
        latent_h = int(math.sqrt(actual_patches * height / width))
        latent_w = actual_patches // latent_h
        if latent_h * latent_w != actual_patches:
            # 如果无法整除，使用近似值
            latent_h = int(math.sqrt(actual_patches))
            latent_w = latent_h
            print(f"Using fallback dimensions: {latent_h} x {latent_w}")

    # 6. Reshape 到空间维度
    batch_size = patch_attn.shape[0]

    if per_view_mask and num_views > 1:
        # 3D Bundle 模式：为每个视角独立计算 mask
        # 假设 patch 按视角排列
        patches_per_view = actual_patches // num_views
        view_h = int(math.sqrt(patches_per_view * (height // 2) / (width // 4)))
        view_w = patches_per_view // view_h

        patch_attn = patch_attn.view(batch_size, num_views, patches_per_view)
        patch_attn = patch_attn.view(batch_size, num_views, view_h, view_w)

        # 归一化每个视角
        for v in range(num_views):
            view_attn = patch_attn[:, v]
            view_min = view_attn.min()
            view_max = view_attn.max()
            patch_attn[:, v] = (view_attn - view_min) / (view_max - view_min + 1e-8)

        # 重新组合成完整的 bundle
        # 假设 2x4 布局: 第一行 RGB (4个), 第二行 Normal (4个)
        row1 = torch.cat([patch_attn[:, i] for i in range(4)], dim=-1)  # [batch, view_h, view_w*4]
        row2 = torch.cat([patch_attn[:, i] for i in range(4, 8)], dim=-1)
        patch_attn = torch.cat([row1, row2], dim=-2)  # [batch, view_h*2, view_w*4]
        patch_attn = patch_attn.unsqueeze(1)  # [batch, 1, H, W]
    else:
        # 统一 mask 模式
        patch_attn = patch_attn.view(batch_size, 1, latent_h, latent_w)

        # 7. 归一化到 [0, 1]
        patch_min = patch_attn.min()
        patch_max = patch_attn.max()
        patch_attn = (patch_attn - patch_min) / (patch_max - patch_min + 1e-8)

    # 8. 上采样到目标分辨率
    patch_attn = F.interpolate(
        patch_attn,
        size=(height, width),
        mode='bilinear',
        align_corners=False
    )

    # 9. 阈值化生成二值 mask
    mask = (patch_attn > threshold).float()

    # 10. 高斯平滑以减少边界伪影
    if gaussian_sigma > 0:
        mask = gaussian_blur_2d(mask, kernel_size=gaussian_kernel_size, sigma=gaussian_sigma)
        # 重新归一化
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

    return mask


def apply_mask_blend(
    latents_src: torch.Tensor,
    latents_tgt: torch.Tensor,
    mask: torch.Tensor,
    blend_mode: str = "hard",
) -> torch.Tensor:
    """
    使用 mask 混合 source 和 target latents

    Args:
        latents_src: source latents [batch, seq_len, dim] 或 [batch, channels, h, w]
        latents_tgt: target latents，形状与 latents_src 相同
        mask: 编辑 mask [1, 1, h, w]，值域 [0, 1]
        blend_mode: 混合模式
            - "hard": 直接使用 mask (edited = mask * tgt + (1-mask) * src)
            - "soft": 使用平滑过渡

    Returns:
        混合后的 latents
    """
    # 确保 mask 与 latents 形状兼容
    if len(latents_src.shape) == 3:
        # packed latents: [batch, seq_len, dim]
        # 需要 unpack，应用 mask，再 pack 回去
        # 这种情况比较复杂，暂时直接使用 tgt
        print("Warning: Mask blending for packed latents not fully implemented. Using target latents directly.")
        return latents_tgt

    # latents: [batch, channels, h, w]
    if mask.shape[-2:] != latents_src.shape[-2:]:
        mask = F.interpolate(mask, size=latents_src.shape[-2:], mode='bilinear', align_corners=False)

    # 扩展 mask 到与 latents 相同的 channels
    mask = mask.expand_as(latents_src)

    if blend_mode == "hard":
        blended = mask * latents_tgt + (1 - mask) * latents_src
    elif blend_mode == "soft":
        # 使用 3 次方使过渡更平滑
        smooth_mask = mask ** 3 * (10 - 15 * mask + 6 * mask ** 2)
        blended = smooth_mask * latents_tgt + (1 - smooth_mask) * latents_src
    else:
        blended = latents_tgt

    return blended


def visualize_t2i_mask(
    mask: torch.Tensor,
    save_path: str,
    original_image: Optional[torch.Tensor] = None,
    overlay_alpha: float = 0.0,
) -> None:
    """
    可视化并保存 T2I mask（带 colorbar 图例）

    Args:
        mask: [1, 1, H, W] 的 mask tensor
        save_path: 保存路径
        original_image: 可选的原始图像，用于叠加显示
        overlay_alpha: 叠加透明度，0.0 表示只显示 mask，0.5 表示半透明叠加
    """
    # 转换为 numpy（bfloat16 需要先转为 float32）
    mask_np = mask.squeeze().float().cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 8))

    if overlay_alpha > 0 and original_image is not None:
        # 叠加到原始图像上
        if original_image.shape[0] == 1:
            original_image = original_image.squeeze(0)
        img_np = original_image.cpu().permute(1, 2, 0).numpy()

        # 调整 mask 尺寸以匹配原图
        if mask_np.shape != img_np.shape[:2]:
            from skimage.transform import resize
            mask_np = resize(mask_np, img_np.shape[:2], preserve_range=True)

        ax.imshow(img_np)
        im = ax.imshow(mask_np, cmap='jet', alpha=overlay_alpha, vmin=0, vmax=1)
    else:
        # 只显示热力图
        im = ax.imshow(mask_np, cmap='jet', vmin=0, vmax=1)

    # 添加 colorbar 图例
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attention', fontsize=12)

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"T2I mask saved to: {save_path}")


def generate_per_token_heatmaps(
    t2i_attn_cache: Dict[str, torch.Tensor],
    tokenizer,
    input_ids: torch.Tensor,
    save_dir: str,
    selected_blocks: Optional[List[int]] = None,
    gaussian_sigma: float = 2.0,
    gaussian_kernel_size: int = 5,
    height: int = 64,
    width: int = 128,
) -> List[str]:
    """
    为每个 text token 生成独立的热力图。

    基于 FluxEdit 论文，提取每个 text token 对图像各区域的注意力，
    生成 token-specific 的热力图并保存到指定文件夹。

    Args:
        t2i_attn_cache: 各层的 T2I 注意力图，key 格式为 "mmdit_{idx}"
                        value 形状为 [batch, heads, img_len, txt_len]
        tokenizer: T5TokenizerFast，用于解码 token ID 到文字
        input_ids: [1, txt_len] 的 tokenized prompt
        save_dir: 输出文件夹路径
        selected_blocks: 要使用的 MMDiT 块索引列表，默认 [0,1,2,3,4]
        gaussian_sigma: 高斯平滑的标准差
        gaussian_kernel_size: 高斯核大小
        height: 输出热力图的高度（latent 空间）
        width: 输出热力图的宽度（latent 空间）

    Returns:
        保存的文件路径列表
    """
    import os
    import re

    os.makedirs(save_dir, exist_ok=True)

    if selected_blocks is None:
        selected_blocks = [0, 1, 2, 3, 4]

    # 1. 收集并平均选定块的 attention
    attn_maps = []
    for idx in selected_blocks:
        key = f"mmdit_{idx}"
        if key in t2i_attn_cache:
            attn_maps.append(t2i_attn_cache[key])

    if len(attn_maps) == 0:
        print(f"Warning: No T2I attention maps found for blocks {selected_blocks}.")
        return []

    # 堆叠并平均
    stacked = torch.stack(attn_maps, dim=0)  # [num_blocks, B, H, img_len, txt_len]
    avg_attn = stacked.mean(dim=0)           # [B, H, img_len, txt_len]
    avg_attn = avg_attn.mean(dim=1)          # [B, img_len, txt_len]

    batch_size, img_len, txt_len = avg_attn.shape

    # 2. 计算空间维度 (Flux 使用 2x2 packing)
    latent_h = height // 2
    latent_w = width // 2
    expected_patches = latent_h * latent_w

    if img_len != expected_patches:
        # 尝试推断正确的尺寸
        latent_h = int(math.sqrt(img_len * height / width))
        latent_w = img_len // latent_h
        if latent_h * latent_w != img_len:
            latent_h = int(math.sqrt(img_len))
            latent_w = latent_h
        print(f"Inferred latent dimensions: {latent_h} x {latent_w}")

    # 3. 获取 token IDs
    tokens = input_ids[0].tolist() if input_ids.dim() > 1 else input_ids.tolist()

    saved_paths = []
    pad_token_id = getattr(tokenizer, 'pad_token_id', 0)
    eos_token_id = getattr(tokenizer, 'eos_token_id', 1)

    # 4. 为每个 token 生成热力图
    for token_idx in range(min(txt_len, len(tokens))):
        token_id = tokens[token_idx]

        # 跳过 padding 和 EOS tokens
        if token_id in [pad_token_id, eos_token_id]:
            continue

        # 解码 token 到文字
        try:
            word = tokenizer.decode([token_id], skip_special_tokens=True).strip()
        except Exception:
            word = ""

        if not word:
            word = f"special_{token_id}"

        # 清理文件名（移除特殊字符，限制长度）
        safe_word = re.sub(r'[^\w\-]', '_', word)[:20]

        # 提取该 token 的 attention: [B, img_len]
        token_attn = avg_attn[:, :, token_idx]  # [B, img_len]

        # Reshape 到空间维度
        token_attn = token_attn.view(batch_size, 1, latent_h, latent_w)

        # 归一化到 [0, 1]
        attn_min = token_attn.min()
        attn_max = token_attn.max()
        if attn_max - attn_min > 1e-8:
            token_attn = (token_attn - attn_min) / (attn_max - attn_min)
        else:
            token_attn = torch.zeros_like(token_attn)

        # 上采样到目标分辨率
        token_attn = F.interpolate(
            token_attn,
            size=(height, width),
            mode='bilinear',
            align_corners=False
        )

        # 高斯平滑
        if gaussian_sigma > 0:
            token_attn = gaussian_blur_2d(
                token_attn,
                kernel_size=gaussian_kernel_size,
                sigma=gaussian_sigma
            )
            # 重新归一化
            attn_min = token_attn.min()
            attn_max = token_attn.max()
            if attn_max - attn_min > 1e-8:
                token_attn = (token_attn - attn_min) / (attn_max - attn_min)

        # 使用 matplotlib 生成带 colorbar 的热力图
        # 注意：bfloat16 需要先转为 float32 才能转到 cpu
        mask_np = token_attn.squeeze().float().cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(mask_np, cmap='jet', vmin=0, vmax=1)

        # 添加 colorbar 图例
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Attention', fontsize=12)

        # 添加 token 作为标题
        ax.set_title(f'Token: "{word}"', fontsize=14)
        ax.axis('off')

        # 保存
        filename = f"token_{token_idx:03d}_{safe_word}.png"
        save_path = os.path.join(save_dir, filename)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_paths.append(save_path)

    print(f"Saved {len(saved_paths)} token heatmaps to {save_dir}")
    return saved_paths


def generate_per_layer_heatmaps(
    t2i_attn_cache_src: Dict[str, torch.Tensor],
    t2i_attn_cache_tgt: Optional[Dict[str, torch.Tensor]],
    save_dir: str,
    layer_interval: int = 3,
    total_layers: int = 19,
    gaussian_sigma: float = 2.0,
    gaussian_kernel_size: int = 5,
    height: int = 64,
    width: int = 128,
) -> List[str]:
    """
    为每一层（按间隔）生成独立的热力图，便于分析不同层的 attention 差异。
    同时绘制 source 和 target 分支的热图。

    Args:
        t2i_attn_cache_src: Source 分支各层的 T2I 注意力图
        t2i_attn_cache_tgt: Target 分支各层的 T2I 注意力图（可选）
        save_dir: 输出文件夹路径
        layer_interval: 层间隔，默认为 3（绘制 0, 3, 6, 9, 12, 15, 18 层）
        total_layers: 总层数，默认为 19（Flux MMDiT blocks）
        gaussian_sigma: 高斯平滑的标准差
        gaussian_kernel_size: 高斯核大小
        height: 输出热力图的高度（latent 空间）
        width: 输出热力图的宽度（latent 空间）

    Returns:
        保存的文件路径列表
    """
    import os
    import math

    saved_paths = []

    # 处理两个分支: src 和 tgt
    branches = [("src", t2i_attn_cache_src)]
    if t2i_attn_cache_tgt:
        branches.append(("tgt", t2i_attn_cache_tgt))

    for branch_name, t2i_attn_cache in branches:
        branch_dir = os.path.join(save_dir, branch_name)
        os.makedirs(branch_dir, exist_ok=True)

        # 计算要绘制的层索引
        layer_indices = list(range(0, total_layers, layer_interval))

        for layer_idx in layer_indices:
            key = f"mmdit_{layer_idx}"
            if key not in t2i_attn_cache:
                print(f"Warning: Layer {layer_idx} not found in {branch_name} cache (key: {key})")
                continue

            # 获取该层的 attention: [batch, heads, img_len, txt_len]
            attn = t2i_attn_cache[key]

            # 平均所有 heads: [batch, img_len, txt_len]
            attn = attn.mean(dim=1)

            # 对所有 tokens 取 max: [batch, img_len]
            patch_attn = attn.max(dim=-1)[0]

            batch_size = patch_attn.shape[0]
            img_len = patch_attn.shape[-1]

            # 计算空间维度 (Flux 使用 2x2 packing)
            latent_h = height // 2
            latent_w = width // 2
            expected_patches = latent_h * latent_w

            if img_len != expected_patches:
                # 尝试推断正确的尺寸
                latent_h = int(math.sqrt(img_len * height / width))
                latent_w = img_len // latent_h
                if latent_h * latent_w != img_len:
                    latent_h = int(math.sqrt(img_len))
                    latent_w = latent_h

            # Reshape 到空间维度
            patch_attn = patch_attn.view(batch_size, 1, latent_h, latent_w)

            # 归一化到 [0, 1]
            attn_min = patch_attn.min()
            attn_max = patch_attn.max()
            if attn_max - attn_min > 1e-8:
                patch_attn = (patch_attn - attn_min) / (attn_max - attn_min)
            else:
                patch_attn = torch.zeros_like(patch_attn)

            # 上采样到目标分辨率
            patch_attn = F.interpolate(
                patch_attn,
                size=(height, width),
                mode='bilinear',
                align_corners=False
            )

            # 高斯平滑
            if gaussian_sigma > 0:
                patch_attn = gaussian_blur_2d(
                    patch_attn,
                    kernel_size=gaussian_kernel_size,
                    sigma=gaussian_sigma
                )
                # 重新归一化
                attn_min = patch_attn.min()
                attn_max = patch_attn.max()
                if attn_max - attn_min > 1e-8:
                    patch_attn = (patch_attn - attn_min) / (attn_max - attn_min)

            # 使用 matplotlib 生成带 colorbar 的热力图
            # 注意：bfloat16 需要先转为 float32 才能转到 cpu
            mask_np = patch_attn.squeeze().float().cpu().numpy()

            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(mask_np, cmap='jet', vmin=0, vmax=1)

            # 添加 colorbar 图例
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Attention', fontsize=12)

            # 添加层号和分支名作为标题
            ax.set_title(f'[{branch_name.upper()}] Layer {layer_idx} / {total_layers - 1}', fontsize=14)
            ax.axis('off')

            # 保存
            filename = f"layer_{layer_idx:02d}.png"
            save_path = os.path.join(branch_dir, filename)
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            saved_paths.append(save_path)

        # 生成该分支的全局平均热图（所有层平均）
        all_attn_maps = []
        for layer_idx in range(total_layers):
            key = f"mmdit_{layer_idx}"
            if key in t2i_attn_cache:
                all_attn_maps.append(t2i_attn_cache[key])

        if all_attn_maps:
            # 堆叠并平均所有层
            stacked = torch.stack(all_attn_maps, dim=0)
            avg_attn = stacked.mean(dim=0)
            avg_attn = avg_attn.mean(dim=1)
            patch_attn = avg_attn.max(dim=-1)[0]

            batch_size = patch_attn.shape[0]
            img_len = patch_attn.shape[-1]

            # 计算空间维度
            latent_h = height // 2
            latent_w = width // 2
            if img_len != latent_h * latent_w:
                latent_h = int(math.sqrt(img_len * height / width))
                latent_w = img_len // latent_h
                if latent_h * latent_w != img_len:
                    latent_h = int(math.sqrt(img_len))
                    latent_w = latent_h

            patch_attn = patch_attn.view(batch_size, 1, latent_h, latent_w)

            # 归一化
            attn_min = patch_attn.min()
            attn_max = patch_attn.max()
            if attn_max - attn_min > 1e-8:
                patch_attn = (patch_attn - attn_min) / (attn_max - attn_min)

            # 上采样
            patch_attn = F.interpolate(patch_attn, size=(height, width), mode='bilinear', align_corners=False)

            # 高斯平滑
            if gaussian_sigma > 0:
                patch_attn = gaussian_blur_2d(patch_attn, kernel_size=gaussian_kernel_size, sigma=gaussian_sigma)
                attn_min = patch_attn.min()
                attn_max = patch_attn.max()
                if attn_max - attn_min > 1e-8:
                    patch_attn = (patch_attn - attn_min) / (attn_max - attn_min)

            # 绘制全局平均热图
            mask_np = patch_attn.squeeze().float().cpu().numpy()
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(mask_np, cmap='jet', vmin=0, vmax=1)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Attention', fontsize=12)
            ax.set_title(f'[{branch_name.upper()}] Global Average (All {len(all_attn_maps)} Layers)', fontsize=14)
            ax.axis('off')

            filename = "layer_avg_global.png"
            save_path = os.path.join(branch_dir, filename)
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            saved_paths.append(save_path)

        print(f"Saved {branch_name} heatmaps to {branch_dir}")

    print(f"Total saved: {len(saved_paths)} layer heatmaps")
    return saved_paths
