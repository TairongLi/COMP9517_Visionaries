"""传统_C: BoVW (SIFT -> 码本 -> 直方图编码)。

流程: 只在 train split 上采样 SIFT 描述子 -> k-means 建码本 -> 所有 split 用同一个码本编码。
码本绝对不能用 val/test 的图训练, 那是 data leakage。
"""

import numpy as np

# ---------------------------------------------------------------- 局部特征


def sift_descriptors(img, max_kp=None):
    """传统_C: 提取单张图的 SIFT 描述子, 返回 (K, 128) float32; 没检测到关键点返回 (0, 128)。
    建议: cv2.SIFT_create(), 先转灰度。
    注意 opencv-python 4.4+ 才有免费的 SIFT, 老版本要装 opencv-contrib。
    可选消融: dense SIFT (固定网格采样) vs 关键点检测。
    """
    raise NotImplementedError("传统_C")


def sample_train_descriptors(train_images, per_image=100, seed=0):
    """传统_C: 从训练图里采样描述子用于建码本。
    500 类 * 40 图 * 每图几百个 SIFT = 上千万个 128 维向量, 全量 k-means 跑不动。
    建议每张图随机抽 per_image 个, 总量控制在 100 万以内。
    per_image 是一个消融点。
    """
    raise NotImplementedError("传统_C")


# ---------------------------------------------------------------- 码本


def build_codebook(descriptors, k=512, seed=0):
    """传统_C: k-means 建码本, 返回训练好的聚类器。
    建议 sklearn.cluster.MiniBatchKMeans (普通 KMeans 在这个数据量下太慢)。
    消融点: k ∈ {256, 512, 1024} —— 这是报告里必须有的一张表。
    记得计时, 归入 train_s。
    """
    raise NotImplementedError("传统_C")


def encode_bovw(img_descriptors, codebook, k, normalize="l2"):
    """传统_C: 把一张图的 SIFT 描述子编码成 k 维直方图。
    步骤: 每个描述子 predict 到最近的 visual word -> 计数 -> 归一化。
    建议先做 L1 归一化再开方 (Hellinger / power normalization), 通常比纯 L2 好, 可作为消融点。
    没有描述子的图返回全零向量, 不要报错。
    """
    raise NotImplementedError("传统_C")


def encode_split(images, codebook, k, **kwargs):
    """对一批图做编码。不要改这个函数。"""
    out = np.zeros((len(images), k), dtype=np.float32)
    for i, im in enumerate(images):
        d = sift_descriptors(im)
        if len(d) > 0:
            out[i] = encode_bovw(d, codebook, k, **kwargs)
    return out


# ---------------------------------------------------------------- 分类器


def build_classifier(name, seed=0):
    """传统_C: 返回一个 sklearn 分类器。
    必须实现 'linear_svm' 和 'random_forest' 两个 —— 这一维是"分类器对比"的依据。

    LinearSVC(C=..., max_iter=...)          # 500 类下比 SVC(kernel='rbf') 快得多
    RandomForestClassifier(n_estimators=..., n_jobs=-1)

    注意: 不要用 SVC(probability=True), Platt scaling 在 500 类下慢到不可接受。
    top-5 只需要排序, decision_function 的输出就够。
    """
    raise NotImplementedError("传统_C")


def scores_of(clf, X):
    """统一取分数矩阵 (N, C)。不要改这个函数。"""
    if hasattr(clf, "decision_function"):
        s = clf.decision_function(X)
        return s if s.ndim == 2 else np.stack([-s, s], axis=1)   # 二分类兜底
    return clf.predict_proba(X)
