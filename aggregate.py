"""汇总_D1: 把 results/*.npz 变成报告里的表和图。
所有图表由这一个文件产出, 五个人不用各画各的, 风格自动一致。
改配色/字号/选取的类别子集时只改这里一处。
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ..metrics import hardest_pairs, load_result, per_class_f1


def main_table(results_dir="results", out="report/main_table.csv"):
    """汇总_D1: 主表 —— 每个方法一行。
    列: method, top1, top5, balanced_acc, macro_P, macro_R, macro_F1, train_s, test_s, ms_per_image
    作业明确要求比较训练/测试时间 vs 性能, 时间列不要漏。
    """
    rows = []
    for p in sorted(Path(results_dir).glob("*.npz")):
        *_, method, scores = load_result(p)
        rows.append({"method": method, **scores})
    df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def plot_curves(runs_dir="runs", run_names=None, out="report/curves.png"):
    """汇总_D1: 训练动态图 —— 从 runs/<name>/curves.csv 画 train/val 的 loss 和 accuracy。
    建议 1x2 子图 (左 loss 右 acc), 每个 run 一条线, train 实线 val 虚线。
    中档评分要求"看得出收敛", 所以 x 轴要跑满 epoch, 别截断。
    """
    raise NotImplementedError("汇总_D1")


def plot_confusion(npz_path, out_full="report/cm_full.png", out_subset="report/cm_subset.png",
                   n_subset=20):
    """汇总_D1: 混淆矩阵。
    两张图: (1) 全类矩阵, 500x500 不标类名, 用 imshow + log 色标看整体结构;
            (2) 选定子集的细化矩阵, 标注类名和数值。
    子集怎么选: 建议取 per-class F1 最低的 n_subset 个类, 比随机取信息量大。
    """
    raise NotImplementedError("汇总_D1")


def confusion_pairs_table(npz_path, top=15, out="report/hard_pairs.csv"):
    """汇总_D1: 最易混淆的物种对。
    hardest_pairs 已经实现, 这里负责存表 + 挑出对应图片做定性分析。
    报告里要说明为什么像 —— 同属? 颜色接近? 姿态/背景相似?
    """
    y_true, y_pred, y_score, names, method, _ = load_result(npz_path)
    pairs = hardest_pairs(y_true, y_pred, names, top=top)
    df = pd.DataFrame(pairs, columns=["true_species", "predicted_as", "count"])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def sample_success_failure(npz_path, manifest, image_root, n=8, out_dir="report/examples"):
    """汇总_D1: 成功与失败样例。
    作业要求展示代表性的成功和失败案例并解释原因。
    建议: 成功取置信度最高的 n 张; 失败取"预测错且置信度还很高"的 n 张 —— 这类最有讨论价值。
    每张图标注 true / pred / score。
    """
    raise NotImplementedError("汇总_D1")


def per_class_report(npz_path, out="report/per_class.csv"):
    """汇总_D1: 每类 F1 + 支持数, 用来找长尾弱类。"""
    y_true, y_pred, _, names, _, _ = load_result(npz_path)
    f1, sup = per_class_f1(y_true, y_pred, names)
    df = pd.DataFrame({"species": names, "f1": f1, "support": sup}).sort_values("f1")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df
