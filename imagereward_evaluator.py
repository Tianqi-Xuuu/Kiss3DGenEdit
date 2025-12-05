"""
ENV: python=3.10 pip install clip image-reward matplotlib torch==2.1 numpy==1.26 transformers==4.40
ImageReward Evaluator - 独立的 ImageReward 评估模块

从 bundle_evaluator.py 中分离出来的 ImageReward 评估功能
只评估上半部分的4个RGB视角（不含法线图）
支持单个 Bundle 评估和多 Bundle 排序

Usage Examples:
  # 评估单个 bundle
  python imagereward_evaluator.py -b bundle.png -p "A cartoon dog"

  # Bundle 排序
  python imagereward_evaluator.py --rank -b v1.png v2.png v3.png -p "A red car"
"""

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from PIL import Image
import ImageReward as RM

# 类型别名
BundleInput = Union[str, Path, Image.Image]


# 4个RGB视角的 prompt 后缀
VIEW_SUFFIXES_4 = [
    " Cartoon-styled rendering, left side view, on a clean white background.",
    " Cartoon-styled rendering, back view, on a clean white background.",
    " Cartoon-styled rendering, right side view, on a clean white background.",
    " Cartoon-styled rendering, front view, on a clean white background.",
]


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class ImageRewardResult:
    """单个 Bundle 的 ImageReward 评估结果"""
    score: float                      # 平均分
    scores_per_view: List[float]      # 4个RGB视角的分数
    prompt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "score": self.score,
            "scores_per_view": self.scores_per_view,
            "prompt": self.prompt,
        }


@dataclass
class ImageRewardRankingResult:
    """多 Bundle 的 ImageReward 排序结果"""
    ranking: List[int]                 # 按分数降序排列的索引
    scores: List[float]                # 每个 bundle 的平均分
    num_bundles: int = 0
    prompt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "ranking": self.ranking,
            "scores": self.scores,
            "num_bundles": self.num_bundles,
            "prompt": self.prompt,
        }


# ============================================================================
# 工具函数
# ============================================================================

def load_bundle_image(bundle: BundleInput) -> Image.Image:
    """
    统一加载 Bundle 图片

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


def extract_rgb_views(bundle_img: Image.Image) -> List[Image.Image]:
    """
    从Bundle中提取上半部分的4个RGB视角

    Args:
        bundle_img: Bundle图片 (2048x1024)

    Returns:
        4个RGB视角图片列表 [左, 背, 右, 前]，每个512x512
    """
    w, h = bundle_img.size
    tile_w = w // 4      # 512
    tile_h = h // 2      # 512，只取上半部分

    views = []
    for i in range(4):
        left = i * tile_w
        view = bundle_img.crop((left, 0, left + tile_w, tile_h))
        views.append(view)
    return views


# ============================================================================
# 主评估类
# ============================================================================

class ImageRewardEvaluator:
    """
    独立的 ImageReward 评估器

    用于评估 3D Bundle 图片与文本提示的匹配程度
    只评估上半部分的4个RGB视角（不含法线图）

    Bundle格式：2048x1024 (宽x高)
    - 上半部分(2048x512): 4个RGB视角（左、背、右、前）← 只评估这部分
    - 下半部分(2048x512): 4个法线/几何视角（不评估）
    """

    def __init__(
        self,
        lazy_load: bool = True,
    ):
        """
        初始化评估器

        Args:
            lazy_load: 是否延迟加载模型（首次使用时加载）
        """
        self.lazy_load = lazy_load

        # 延迟加载的模型实例
        self._model = None

        # 非延迟加载时立即初始化
        if not lazy_load:
            _ = self.model

    @property
    def model(self):
        """获取 ImageReward 模型（延迟加载）"""
        if self._model is None:
            print("Loading ImageReward-v1.0...")
            self._model = RM.load("ImageReward-v1.0")
            print("Model loaded.")
        return self._model

    def evaluate(
        self,
        bundle: BundleInput,
        prompt: str,
        top_k: Optional[int] = None,
        verbose: bool = False,
    ) -> ImageRewardResult:
        """
        评估单个 Bundle 图片（只评估4个RGB视角）

        Args:
            bundle: Bundle图片（路径或PIL Image）
            prompt: 文本描述
            top_k: 只取前k个最高分的视角计算平均分（最大4）
            verbose: 是否打印详细分数

        Returns:
            ImageRewardResult 包含评估结果
        """
        bundle_img = load_bundle_image(bundle)
        views = extract_rgb_views(bundle_img)

        scores = []
        for i, view in enumerate(views):
            view_prompt = prompt + VIEW_SUFFIXES_4[i]
            score = self.model.score(view_prompt, view)
            scores.append(score)
            if verbose:
                print(f"[View {i}] score: {score:.4f}")

        # 计算平均分
        if top_k is not None and 1 <= top_k < len(scores):
            top_scores = sorted(scores, reverse=True)[:top_k]
            avg_score = sum(top_scores) / top_k
        else:
            avg_score = sum(scores) / len(scores)

        return ImageRewardResult(
            score=avg_score,
            scores_per_view=scores,
            prompt=prompt,
        )

    def rank(
        self,
        bundles: List[BundleInput],
        prompt: str,
        top_k: Optional[int] = None,
    ) -> ImageRewardRankingResult:
        """
        对多个 Bundle 进行排序

        Args:
            bundles: bundle图片列表
            prompt: 文本描述
            top_k: 只取前k个最高分的视角计算平均分

        Returns:
            ImageRewardRankingResult 包含排序结果
        """
        n = len(bundles)
        if n < 1:
            raise ValueError("bundles列表不能为空")

        # 评估每个 Bundle
        scores = []
        for bundle in bundles:
            result = self.evaluate(bundle, prompt, top_k=top_k, verbose=False)
            scores.append(result.score)

        # 按分数降序排列的索引
        ranking = sorted(range(n), key=lambda i: scores[i], reverse=True)

        return ImageRewardRankingResult(
            ranking=ranking,
            scores=scores,
            num_bundles=n,
            prompt=prompt,
        )

    def release_model(self) -> None:
        """释放模型占用的内存"""
        if self._model is not None:
            del self._model
            self._model = None

        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================================
# 便捷函数
# ============================================================================

def evaluate_imagereward(
    bundle: BundleInput,
    prompt: str,
    **kwargs,
) -> ImageRewardResult:
    """
    便捷函数：评估单个 Bundle

    注意：每次调用会重新加载模型，批量评估请使用 ImageRewardEvaluator 类

    Args:
        bundle: Bundle图片（路径或PIL Image）
        prompt: 文本描述
        **kwargs: 传递给 ImageRewardEvaluator 的其他参数

    Returns:
        ImageRewardResult
    """
    evaluator = ImageRewardEvaluator(**kwargs)
    return evaluator.evaluate(bundle, prompt)


def rank_imagereward(
    bundles: List[BundleInput],
    prompt: str,
    **kwargs,
) -> ImageRewardRankingResult:
    """
    便捷函数：对多个 Bundle 排序

    注意：每次调用会重新加载模型，多次调用请使用 ImageRewardEvaluator 类

    Args:
        bundles: bundle图片列表
        prompt: 文本描述
        **kwargs: 传递给 ImageRewardEvaluator 的其他参数

    Returns:
        ImageRewardRankingResult
    """
    evaluator = ImageRewardEvaluator(**kwargs)
    return evaluator.rank(bundles, prompt)


# ============================================================================
# 命令行入口
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ImageReward Evaluator - 只评估4个RGB视角",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 评估单个 bundle
  python imagereward_evaluator.py -b outputs/bundle.png -p "A cute cartoon cat"

  # Bundle 排序模式
  python imagereward_evaluator.py --rank -b v1.png v2.png v3.png -p "A red sports car"

  # 使用 top-k 评分（只取前3个最高分）
  python imagereward_evaluator.py -b bundle.png -p "A dog" --top-k 3
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
        "--rank",
        action="store_true",
        help="启用排序模式（对多个bundle排序）"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="只取前k个最高分的视角计算平均分（最大4）"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="打印每个视角的详细分数"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出JSON文件路径"
    )

    args = parser.parse_args()

    evaluator = ImageRewardEvaluator()

    if args.rank:
        # 排序模式
        if len(args.bundle) < 2:
            print("排序模式需要至少2个bundle")
            exit(1)

        print(f"排序 {len(args.bundle)} 个Bundle...")
        result = evaluator.rank(
            bundles=args.bundle,
            prompt=args.prompt,
            top_k=args.top_k,
        )

        print("\n" + "=" * 50)
        print(f"Prompt: {args.prompt}")
        print("=" * 50)
        print(f"\nImageReward 排序: {result.ranking}")
        print(f"ImageReward 分数: {[f'{s:.4f}' for s in result.scores]}")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
            print(f"\n结果已保存到: {args.output}")

    else:
        # 单 bundle 评估模式
        if len(args.bundle) > 1:
            print("单bundle评估模式只接受1个bundle，如需排序请使用 --rank")
            exit(1)

        print(f"评估Bundle: {args.bundle[0]}")
        result = evaluator.evaluate(
            bundle=args.bundle[0],
            prompt=args.prompt,
            top_k=args.top_k,
            verbose=args.verbose,
        )

        print("\n" + "=" * 50)
        print(f"Prompt: {args.prompt}")
        print("=" * 50)
        print(f"ImageReward Score: {result.score:.4f}")
        print(f"Per-view scores: {[f'{s:.4f}' for s in result.scores_per_view]}")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
            print(f"\n结果已保存到: {args.output}")
