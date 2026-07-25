"""深度组的实验矩阵。每个人只填自己那一段的 EXPERIMENTS 列表, 然后跑 run_all()。

variant() 会断言"只改一个字段", 所以这里天然强制单变量消融。
基准配置在 src/config.py 的 BASELINE, 不要各自改基准。
"""

from ..config import BASELINE, variant
from ..train import fit, test

# ------------------------------------------------------------------ 深度_D2
# 负责人: D2 —— 从零训练 + 架构对比
# 目标: 证明模型确实收敛 (curves.csv 走平), 并给出至少两种架构的对比。
# 提醒: from scratch 收敛慢, epochs 可能要显著大于 BASELINE, 尽早起跑。

D2_EXPERIMENTS = [
    variant("scratch_r18"),                          # 基准: resnet18, pretrained=False
    # 深度_D2: 换架构 —— variant("scratch_r50", arch="resnet50")
    # 深度_D2: 换架构 —— variant("scratch_effb0", arch="efficientnet_b0")
    # 深度_D2: 优化器消融 —— variant("scratch_sgd", optimizer="sgd")
    # 深度_D2: 学习率消融 —— 至少两个点, 用来说明基准 lr 是选过的不是拍的
]

# ------------------------------------------------------------------ 深度_D3
# 负责人: D3 —— 预训练迁移 + 数据增强消融
# 目标: 与 D2 的 scratch_r18 构成"同架构、只差预训练权重"的干净对比 (作业硬性要求)。
# 提醒: 微调的 lr 通常要比 from scratch 小一个量级, 但那样就变成两个变量了。
#       处理办法: 主表保持单变量, 另开一张小表单独报告 lr 敏感性, 并在报告里说明。

D3_EXPERIMENTS = [
    variant("pretrained_r18", pretrained=True),      # 与 scratch_r18 唯一差别: 预训练权重
    # 深度_D3: 增强消融 —— variant("pretrained_noaug", augment="none")   (基于预训练基准)
    # 深度_D3: 增强消融 —— variant("pretrained_strongaug", augment="strong")
    # 深度_D3: 架构 x 预训练 —— variant("pretrained_r50", arch="resnet50")
]

# ------------------------------------------------------------------ 深度_D1 (框架维护者)
# 负责人: 你 —— 框架写完后接手这一段
# 核心对比表: 同架构下 scratch vs pretrained, 这是报告里最重要的一张表。
# 另外负责: 汇总所有 results/*.npz, 出主表格 + 混淆矩阵 + 最易混淆物种对。

D1_EXPERIMENTS = [
    # 深度_D1: 图像尺寸消融 —— variant("size_160", img_size=160)
    # 深度_D1: label smoothing —— variant("ls_01", label_smoothing=0.1)
]


def run_all(experiments, dry_run=False):
    """按顺序跑一组实验。断线后重跑同一份列表会自动 resume, 已完成的很快跳过。"""
    done = []
    for cfg in experiments:
        print(f"\n=== {cfg.run_name} ===")
        if dry_run:
            print(cfg)
            continue
        model, names, train_s, run = fit(cfg)
        path, scores = test(cfg, model, names, train_s, run)
        done.append((cfg.run_name, scores))
    return done


if __name__ == "__main__":
    run_all(D2_EXPERIMENTS, dry_run=True)   # 先 dry_run 检查配置, 确认无误再关掉
