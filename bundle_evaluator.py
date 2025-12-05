"""
Bundle Evaluator - 统一的3D Bundle评估系统

整合 MVClipEvaluator、LPIPSCalculator、GeminiConsistencyEvaluator
提供 evaluate_bundle() 和 rank_bundles() 两个核心功能

Bundle格式：2048x1024 (宽x高)
- 上半部分(2048x512): 4个RGB视角（左、背、右、前）
- 下半部分(2048x512): 4个法线/几何视角
"""
"""
Usage Examples:
  # 评估单个bundle
  python bundle_evaluator.py -b bundle.png -p "A cartoon dog"

  # 评估并计算LPIPS
  python bundle_evaluator.py -b edited.png -p "A cat" -r original.png

  # Bundle排序
  python bundle_evaluator.py --rank -b v1.png v2.png v3.png v4.png -p "A red car"

  # 跳过Gemini（节省API费用）
  python bundle_evaluator.py -b bundle.png -p "A dog" --skip-gemini
"""

import os
import re
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

import requests
import torch
from PIL import Image

# 导入现有评估器
from mvclip import MVClipEvaluator, split_bundle_image, default_view_suffixes_8
from lpips_utils import LPIPSCalculator
from gemini_consistency_eval import (
    GeminiConsistencyEvaluator,
    extract_rgb_views,
    merge_views_to_grid,
    image_to_base64,
)

# 类型别名
BundleInput = Union[str, Path, Image.Image]


# ============================================================================
# 排序专用 Prompt
# ============================================================================

RANKING_PROMPT = """你是一个3D模型质量评估专家。

我将展示{n}个3D模型的多视角渲染图。每个模型显示4个不同视角（2x2网格排列）：
- 左上：左侧视图
- 右上：背面视图
- 左下：右侧视图
- 右下：正面视图

请根据以下提示词评估哪个模型生成得最好：

**提示词**: "{prompt}"

评估维度：
1. **语义匹配度**：模型是否准确表达了提示词描述的对象
2. **视角一致性**：4个视角是否展示同一个连贯的3D物体
3. **整体质量**：形状、纹理、颜色的合理性

请以JSON格式返回结果：
{{
    "ranking": [<最佳模型索引>, <次佳模型索引>, ...],
    "scores": [<模型0综合分数>, <模型1综合分数>, ...],
    "analysis": "<排序理由的简短分析>"
}}

注意：
- 模型索引从0开始
- 分数范围0-100
- ranking数组长度应等于模型数量
"""


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class BundleEvaluationResult:
    """单个Bundle的完整评估结果"""
    # MVClip/ImageReward 分数
    mvclip_score: float
    mvclip_scores_per_view: List[float]  # 8个视角的分数

    # Gemini 评估分数
    gemini_consistency_score: float       # 视角一致性 (0-100)
    gemini_semantic_score: float          # 语义匹配度 (0-100)
    gemini_consistency_analysis: str      # 一致性分析文本
    gemini_semantic_analysis: str         # 语义分析文本

    # LPIPS 距离（可选，需要reference_bundle）
    lpips_distance: Optional[float] = None
    lpips_distances_per_view: Optional[List[float]] = None  # 4个RGB视角

    # 元信息
    prompt: str = ""
    has_lpips: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "mvclip_score": self.mvclip_score,
            "mvclip_scores_per_view": self.mvclip_scores_per_view,
            "gemini_consistency_score": self.gemini_consistency_score,
            "gemini_semantic_score": self.gemini_semantic_score,
            "gemini_consistency_analysis": self.gemini_consistency_analysis,
            "gemini_semantic_analysis": self.gemini_semantic_analysis,
            "lpips_distance": self.lpips_distance,
            "lpips_distances_per_view": self.lpips_distances_per_view,
            "prompt": self.prompt,
            "has_lpips": self.has_lpips,
        }


@dataclass
class RankingResult:
    """Bundle排序结果"""
    # Gemini排序结果
    gemini_ranking: List[int]              # 按偏好排序的索引 [最优, 次优, ...]
    gemini_scores: List[float]             # 每个bundle的Gemini综合分数
    gemini_analysis: str                   # Gemini排序分析

    # ImageReward排序结果
    imagereward_ranking: List[int]         # 按分数排序的索引
    imagereward_scores: List[float]        # 每个bundle的ImageReward平均分

    # 元信息
    num_bundles: int = 0
    prompt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "gemini_ranking": self.gemini_ranking,
            "gemini_scores": self.gemini_scores,
            "gemini_analysis": self.gemini_analysis,
            "imagereward_ranking": self.imagereward_ranking,
            "imagereward_scores": self.imagereward_scores,
            "num_bundles": self.num_bundles,
            "prompt": self.prompt,
        }


# ============================================================================
# 工具函数
# ============================================================================

def load_bundle_image(bundle: BundleInput) -> Image.Image:
    """
    统一加载Bundle图片

    Args:
        bundle: 路径字符串、Path对象或PIL Image

    Returns:
        PIL Image (RGB模式)

    Raises:
        FileNotFoundError: 文件不存在
        TypeError: 不支持的输入类型
    """
    if isinstance(bundle, (str, Path)):
        path = Path(bundle)
        if not path.exists():
            raise FileNotFoundError(f"Bundle图片不存在: {bundle}")
        img = Image.open(path).convert("RGB")
    elif isinstance(bundle, Image.Image):
        img = bundle.convert("RGB")
    else:
        raise TypeError(f"不支持的bundle类型: {type(bundle)}")

    # 验证尺寸
    w, h = img.size
    if w != 2048 or h != 1024:
        warnings.warn(f"预期bundle尺寸为2048x1024，实际为{w}x{h}")

    return img


def extract_rgb_views_from_bundle(bundle_img: Image.Image) -> List[Image.Image]:
    """
    从Bundle中提取4个RGB视角（上半部分）

    Args:
        bundle_img: Bundle图片

    Returns:
        4个RGB视角图片列表 [左, 背, 右, 前]
    """
    w, h = bundle_img.size
    tile_w = w // 4
    tile_h = h // 2  # 只取上半部分

    views = []
    for i in range(4):
        left = i * tile_w
        view = bundle_img.crop((left, 0, left + tile_w, tile_h))
        views.append(view)
    return views


# ============================================================================
# 主评估类
# ============================================================================

class BundleEvaluator:
    """
    统一的3D Bundle评估器

    整合 MVClipEvaluator、LPIPSCalculator、GeminiConsistencyEvaluator
    提供一站式评估和排序功能

    Bundle格式：2048x1024 (宽x高)
    - 上半部分(2048x512): 4个RGB视角（左、背、右、前）
    - 下半部分(2048x512): 4个法线/几何视角
    """

    def __init__(
        self,
        device: Optional[str] = None,
        mvclip_model_type: str = "ImageReward",
        lpips_net: str = "alex",
        gemini_api_key: Optional[str] = None,
        gemini_model: str = "google/gemini-2.5-pro-preview-06-05",
        lazy_load: bool = True,
    ):
        """
        初始化评估器

        Args:
            device: 计算设备 ('cuda' 或 'cpu')，None表示自动检测
            mvclip_model_type: MVClip模型类型 ("ImageReward", "CLIP", "BLIP", "Aesthetic")
            lpips_net: LPIPS网络类型 ("alex" 或 "vgg")
            gemini_api_key: OpenRouter API Key，None则从环境变量读取
            gemini_model: Gemini模型名称
            lazy_load: 是否延迟加载模型（首次使用时加载）
        """
        # 设备设置
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # 配置存储
        self.mvclip_model_type = mvclip_model_type
        self.lpips_net = lpips_net
        self.gemini_api_key = gemini_api_key or os.environ.get("OPENROUTER_API_KEY")
        self.gemini_model = gemini_model
        self.lazy_load = lazy_load

        # 延迟加载的模型实例
        self._mvclip_evaluator: Optional[MVClipEvaluator] = None
        self._lpips_calculator: Optional[LPIPSCalculator] = None
        self._gemini_evaluator: Optional[GeminiConsistencyEvaluator] = None

        # 非延迟加载时立即初始化
        if not lazy_load:
            self._init_all_models()

    def _init_all_models(self) -> None:
        """初始化所有模型"""
        _ = self.mvclip_evaluator
        _ = self.lpips_calculator
        _ = self.gemini_evaluator

    @property
    def mvclip_evaluator(self) -> MVClipEvaluator:
        """获取MVClip评估器（延迟加载）"""
        if self._mvclip_evaluator is None:
            self._mvclip_evaluator = MVClipEvaluator(
                model_type=self.mvclip_model_type,
                rows=2,
                cols=4,
                save_dir=None,
            )
        return self._mvclip_evaluator

    @property
    def lpips_calculator(self) -> LPIPSCalculator:
        """获取LPIPS计算器（延迟加载）"""
        if self._lpips_calculator is None:
            self._lpips_calculator = LPIPSCalculator(
                net=self.lpips_net,
                device=self.device,
            )
        return self._lpips_calculator

    @property
    def gemini_evaluator(self) -> GeminiConsistencyEvaluator:
        """获取Gemini评估器（延迟加载）"""
        if self._gemini_evaluator is None:
            if not self.gemini_api_key:
                raise ValueError(
                    "Gemini API key未设置。请通过参数传入或设置OPENROUTER_API_KEY环境变量"
                )
            self._gemini_evaluator = GeminiConsistencyEvaluator(
                api_key=self.gemini_api_key,
                model=self.gemini_model,
            )
        return self._gemini_evaluator

    def evaluate_bundle(
        self,
        bundle: BundleInput,
        prompt: str,
        reference_bundle: Optional[BundleInput] = None,
        skip_gemini: bool = False,
        skip_mvclip: bool = False,
        mvclip_top_k: Optional[int] = None,
    ) -> BundleEvaluationResult:
        """
        评估单个Bundle图片

        Args:
            bundle: Bundle图片（路径或PIL Image）
            prompt: 文本描述
            reference_bundle: 可选，用于LPIPS对比的参考bundle
            skip_gemini: 是否跳过Gemini评估（节省API调用）
            skip_mvclip: 是否跳过MVClip评估
            mvclip_top_k: MVClip只取前k个最高分的视角计算平均分

        Returns:
            BundleEvaluationResult 包含所有评估指标
        """
        # 1. 加载Bundle图片
        bundle_img = load_bundle_image(bundle)

        # 2. MVClip/ImageReward 评估
        mvclip_score = 0.0
        mvclip_scores_per_view: List[float] = []

        if not skip_mvclip:
            try:
                avg_score, scores = self.mvclip_evaluator.evaluate_bundle(
                    bundle=bundle_img,
                    base_prompt=prompt,
                    top_k=mvclip_top_k,
                    verbose_scores=False,
                    save_plot=False,
                )
                mvclip_score = avg_score
                mvclip_scores_per_view = scores
            except Exception as e:
                warnings.warn(f"MVClip评估失败: {e}")
                mvclip_scores_per_view = [0.0] * 8

        # 3. Gemini 一致性评估
        gemini_consistency_score = 0.0
        gemini_semantic_score = 0.0
        gemini_consistency_analysis = ""
        gemini_semantic_analysis = ""

        if not skip_gemini:
            try:
                gemini_result = self.gemini_evaluator.evaluate(
                    bundle_image=bundle_img,
                    prompt=prompt,
                )
                gemini_consistency_score = gemini_result.get("consistency_score", 0.0)
                gemini_semantic_score = gemini_result.get("semantic_score", 0.0)
                gemini_consistency_analysis = gemini_result.get("consistency_analysis", "")
                gemini_semantic_analysis = gemini_result.get("semantic_analysis", "")
            except requests.exceptions.Timeout:
                gemini_consistency_analysis = "Gemini API超时"
            except requests.exceptions.RequestException as e:
                gemini_consistency_analysis = f"Gemini API错误: {e}"
            except Exception as e:
                gemini_consistency_analysis = f"Gemini评估失败: {e}"

        # 4. LPIPS 距离（可选）
        lpips_distance: Optional[float] = None
        lpips_distances_per_view: Optional[List[float]] = None
        has_lpips = False

        if reference_bundle is not None:
            try:
                ref_img = load_bundle_image(reference_bundle)

                # 提取两个bundle的RGB视角
                bundle_views = extract_rgb_views_from_bundle(bundle_img)
                ref_views = extract_rgb_views_from_bundle(ref_img)

                # 计算每个视角的LPIPS
                distances = []
                for bv, rv in zip(bundle_views, ref_views):
                    dist = self.lpips_calculator.calculate_distance(bv, rv)
                    distances.append(dist)

                lpips_distances_per_view = distances
                lpips_distance = sum(distances) / len(distances)
                has_lpips = True
            except Exception as e:
                warnings.warn(f"LPIPS计算失败: {e}")

        # 5. 构建返回结果
        return BundleEvaluationResult(
            mvclip_score=mvclip_score,
            mvclip_scores_per_view=mvclip_scores_per_view,
            gemini_consistency_score=gemini_consistency_score,
            gemini_semantic_score=gemini_semantic_score,
            gemini_consistency_analysis=gemini_consistency_analysis,
            gemini_semantic_analysis=gemini_semantic_analysis,
            lpips_distance=lpips_distance,
            lpips_distances_per_view=lpips_distances_per_view,
            prompt=prompt,
            has_lpips=has_lpips,
        )

    def rank_bundles(
        self,
        bundles: List[BundleInput],
        prompt: str,
    ) -> RankingResult:
        """
        对多个Bundle进行偏好排序

        Args:
            bundles: 1-4张bundle图片列表
            prompt: 文本描述

        Returns:
            RankingResult 包含两套独立排序结果

        Raises:
            ValueError: 如果bundles数量不在1-4范围内
        """
        n = len(bundles)
        if n < 1 or n > 4:
            raise ValueError(f"bundles数量必须在1-4范围内，当前: {n}")

        # 1. 加载所有Bundle
        bundle_imgs = [load_bundle_image(b) for b in bundles]

        # 2. ImageReward 评分
        imagereward_scores = []
        for img in bundle_imgs:
            try:
                avg_score, _ = self.mvclip_evaluator.evaluate_bundle(
                    bundle=img,
                    base_prompt=prompt,
                    verbose_scores=False,
                    save_plot=False,
                )
                imagereward_scores.append(avg_score)
            except Exception as e:
                warnings.warn(f"ImageReward评估失败: {e}")
                imagereward_scores.append(0.0)

        # 按分数降序排列的索引
        imagereward_ranking = sorted(
            range(n),
            key=lambda i: imagereward_scores[i],
            reverse=True,
        )

        # 3. Gemini 排序评估
        try:
            gemini_ranking, gemini_scores, gemini_analysis = self._gemini_rank(
                bundle_imgs, prompt
            )
        except Exception as e:
            warnings.warn(f"Gemini排序失败: {e}")
            gemini_ranking = list(range(n))
            gemini_scores = [50.0] * n
            gemini_analysis = f"Gemini排序失败: {e}"

        return RankingResult(
            gemini_ranking=gemini_ranking,
            gemini_scores=gemini_scores,
            gemini_analysis=gemini_analysis,
            imagereward_ranking=imagereward_ranking,
            imagereward_scores=imagereward_scores,
            num_bundles=n,
            prompt=prompt,
        )

    def _gemini_rank(
        self,
        bundle_imgs: List[Image.Image],
        prompt: str,
    ) -> Tuple[List[int], List[float], str]:
        """使用Gemini对多个Bundle进行排序"""
        n = len(bundle_imgs)

        # 为每个bundle提取RGB视角并创建2x2网格
        grids = []
        for img in bundle_imgs:
            views = extract_rgb_views_from_bundle(img)
            grid = merge_views_to_grid(views)
            grids.append(grid)

        # 构建多图消息
        content = [
            {"type": "text", "text": RANKING_PROMPT.format(n=n, prompt=prompt)}
        ]

        for i, grid in enumerate(grids):
            img_base64 = image_to_base64(grid)
            content.append({
                "type": "text",
                "text": f"\n--- 模型 {i} ---"
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_base64}"}
            })

        # 调用API
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.gemini_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.gemini_model,
                "messages": [{"role": "user", "content": content}],
            },
            timeout=120,
        )
        response.raise_for_status()

        # 解析结果
        return self._parse_ranking_response(response.json(), n)

    def _parse_ranking_response(
        self,
        response: dict,
        n: int,
    ) -> Tuple[List[int], List[float], str]:
        """解析Gemini排序响应"""
        try:
            content = response["choices"][0]["message"]["content"]

            # 提取JSON
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 尝试直接解析
                json_str = content

            data = json.loads(json_str)

            ranking = data.get("ranking", list(range(n)))
            scores = data.get("scores", [50.0] * n)
            analysis = data.get("analysis", "")

            # 验证ranking有效性
            if len(ranking) != n or set(ranking) != set(range(n)):
                ranking = list(range(n))

            # 确保scores长度正确
            if len(scores) != n:
                scores = [50.0] * n

            return ranking, [float(s) for s in scores], analysis

        except Exception as e:
            return list(range(n)), [50.0] * n, f"解析失败: {str(e)}"

    def release_models(self) -> None:
        """释放模型占用的内存"""
        if self._mvclip_evaluator is not None:
            if hasattr(self._mvclip_evaluator, 'model'):
                del self._mvclip_evaluator.model
            self._mvclip_evaluator = None

        if self._lpips_calculator is not None:
            if hasattr(self._lpips_calculator, 'loss_fn'):
                del self._lpips_calculator.loss_fn
            self._lpips_calculator = None

        self._gemini_evaluator = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================================
# 便捷函数
# ============================================================================

def evaluate_bundle(
    bundle: BundleInput,
    prompt: str,
    reference_bundle: Optional[BundleInput] = None,
    **kwargs,
) -> BundleEvaluationResult:
    """
    便捷函数：评估单个Bundle

    注意：每次调用会重新加载模型，批量评估请使用 BundleEvaluator 类

    Args:
        bundle: Bundle图片（路径或PIL Image）
        prompt: 文本描述
        reference_bundle: 可选，用于LPIPS对比的参考bundle
        **kwargs: 传递给 BundleEvaluator 的其他参数

    Returns:
        BundleEvaluationResult
    """
    evaluator = BundleEvaluator(**kwargs)
    return evaluator.evaluate_bundle(bundle, prompt, reference_bundle)


def rank_bundles(
    bundles: List[BundleInput],
    prompt: str,
    **kwargs,
) -> RankingResult:
    """
    便捷函数：对多个Bundle排序

    注意：每次调用会重新加载模型，多次调用请使用 BundleEvaluator 类

    Args:
        bundles: 1-4张bundle图片列表
        prompt: 文本描述
        **kwargs: 传递给 BundleEvaluator 的其他参数

    Returns:
        RankingResult
    """
    evaluator = BundleEvaluator(**kwargs)
    return evaluator.rank_bundles(bundles, prompt)


# ============================================================================
# 命令行入口
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Bundle Evaluator - 统一的3D Bundle评估系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 评估单个bundle
  python bundle_evaluator.py -b outputs/bundle.png -p "A cute cartoon cat"

  # 评估并计算与参考bundle的LPIPS距离
  python bundle_evaluator.py -b outputs/edited.png -p "A cat" -r outputs/original.png

  # 跳过Gemini评估（节省API费用）
  python bundle_evaluator.py -b outputs/bundle.png -p "A cat" --skip-gemini

  # Bundle排序模式
  python bundle_evaluator.py --rank -b v1.png v2.png v3.png -p "A red sports car"
        """
    )

    parser.add_argument(
        "--bundle", "-b",
        type=str,
        nargs="+",
        required=True,
        help="Bundle图片路径（排序模式下可传多个）"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        required=True,
        help="文本描述"
    )
    parser.add_argument(
        "--reference", "-r",
        type=str,
        default=None,
        help="参考Bundle路径（用于LPIPS计算，仅单bundle评估模式）"
    )
    parser.add_argument(
        "--rank",
        action="store_true",
        help="启用排序模式（对多个bundle排序）"
    )
    parser.add_argument(
        "--skip-gemini",
        action="store_true",
        help="跳过Gemini评估"
    )
    parser.add_argument(
        "--skip-mvclip",
        action="store_true",
        help="跳过MVClip评估"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出JSON文件路径"
    )

    args = parser.parse_args()

    evaluator = BundleEvaluator()

    if args.rank:
        # 排序模式
        if len(args.bundle) < 2:
            print("排序模式需要至少2个bundle")
            exit(1)
        if len(args.bundle) > 4:
            print("排序模式最多支持4个bundle")
            exit(1)

        print(f"排序 {len(args.bundle)} 个Bundle...")
        result = evaluator.rank_bundles(
            bundles=args.bundle,
            prompt=args.prompt,
        )

        print("\n" + "=" * 50)
        print(f"Prompt: {args.prompt}")
        print("=" * 50)
        print(f"\nImageReward 排序: {result.imagereward_ranking}")
        print(f"ImageReward 分数: {[f'{s:.4f}' for s in result.imagereward_scores]}")
        print(f"\nGemini 排序: {result.gemini_ranking}")
        print(f"Gemini 分数: {[f'{s:.1f}' for s in result.gemini_scores]}")
        print(f"Gemini 分析: {result.gemini_analysis}")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
            print(f"\n结果已保存到: {args.output}")

    else:
        # 单bundle评估模式
        if len(args.bundle) > 1:
            print("单bundle评估模式只接受1个bundle，如需排序请使用 --rank")
            exit(1)

        print(f"评估Bundle: {args.bundle[0]}")
        result = evaluator.evaluate_bundle(
            bundle=args.bundle[0],
            prompt=args.prompt,
            reference_bundle=args.reference,
            skip_gemini=args.skip_gemini,
            skip_mvclip=args.skip_mvclip,
        )

        print("\n" + "=" * 50)
        print(f"Prompt: {args.prompt}")
        print("=" * 50)
        print(f"MVClip Score: {result.mvclip_score:.4f}")
        print(f"Gemini Consistency: {result.gemini_consistency_score}/100")
        print(f"Gemini Semantic: {result.gemini_semantic_score}/100")
        if result.has_lpips:
            print(f"LPIPS Distance: {result.lpips_distance:.4f}")
        print("-" * 50)
        if result.gemini_consistency_analysis:
            print(f"一致性分析: {result.gemini_consistency_analysis}")
        if result.gemini_semantic_analysis:
            print(f"语义分析: {result.gemini_semantic_analysis}")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
            print(f"\n结果已保存到: {args.output}")

