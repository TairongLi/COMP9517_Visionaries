"""传统_B: 手工特征描述子。

规则:
  - 每个描述子一个独立函数, 不要抽公共基类。
  - 输入统一是 np.ndarray (H, W, 3) uint8 RGB, 由 io.load_split 提供。
  - 输出统一是 1-D float32 向量, 同一个描述子对所有图必须等长。
  - 不要在这里做分类, 分类在 run_traditional.py 里。
"""

import numpy as np

# 描述子注册表: run_traditional.py 通过名字取函数。加新描述子记得在这里登记。
DESCRIPTORS = {}


def register(name):
    def deco(fn):
        DESCRIPTORS[name] = fn
        return fn
    return deco


@register("color_hist")
def color_histogram(img, bins=32):
    """传统_B: 颜色直方图。
    建议: 转 HSV, 每通道 bins 个 bin, 分别归一化后 concat -> 3*bins 维。
    参考 cv2.calcHist 或 np.histogram。
    消融点: bins ∈ {16, 32, 64}, RGB vs HSV。
    """
    raise NotImplementedError("传统_B")


@register("color_moments")
def color_moments(img):
    """传统_B: 颜色矩。
    建议: 每通道取 mean / std / skewness -> 9 维。可先把图切成 2x2 或 3x3 网格
    再逐格计算, 保留粗略空间信息 (grid 是一个消融点)。
    """
    raise NotImplementedError("传统_B")


@register("lbp")
def lbp(img, P=8, R=1, method="uniform"):
    """传统_B: LBP 纹理特征。
    建议: skimage.feature.local_binary_pattern, 先转灰度, 输出直方图并归一化。
    uniform 模式下 P=8 是 10 维, 偏短 -> 可考虑多尺度 (R ∈ {1,2,3}) 拼接。
    消融点: P/R 组合, 是否多尺度。
    """
    raise NotImplementedError("传统_B")


@register("hog")
def hog(img, size=(128, 128), orientations=9, pixels_per_cell=(16, 16)):
    """传统_B: HOG 形状特征。
    建议: skimage.feature.hog, 必须先 resize 到固定尺寸否则维度不一致。
    注意 pixels_per_cell 太小会让维度爆炸 (500 类 * 30000 图会吃满内存), 先算一下维度。
    消融点: cell 大小, orientations。
    """
    raise NotImplementedError("传统_B")


@register("color_hist_lbp")
def color_hist_lbp(img):
    """传统_B: 早期融合示例 —— 颜色 + 纹理 concat。
    做完单个描述子后再做这个, 用来说明"互补性"。
    注意两段特征的数值尺度差异, concat 前各自 L2 归一化。
    """
    raise NotImplementedError("传统_B")


def extract_all(images, descriptor_name, **kwargs):
    """对一批图跑同一个描述子。不要改这个函数, B 只需要实现上面的描述子。"""
    fn = DESCRIPTORS[descriptor_name]
    feats = [np.asarray(fn(im, **kwargs), dtype=np.float32).ravel() for im in images]
    dims = {f.shape[0] for f in feats}
    assert len(dims) == 1, f"{descriptor_name} 输出维度不一致: {dims}"
    return np.stack(feats)
