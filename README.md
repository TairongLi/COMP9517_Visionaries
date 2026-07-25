# 任务分工

每个人只改自己名下的文件。带 `raise NotImplementedError("<你的标记>")` 的函数就是你要填的。
标记规则: `传统_B` / `传统_C` / `深度_D2` / `深度_D3` / `深度_D1` / `汇总_D1`。
全局搜索自己的标记就能列出待办。

| 标记 | 负责人 | 文件 | 内容 |
|---|---|---|---|
| — | A | (数据管线) | 产出 `manifest.csv` + 图片目录 |
| `传统_B` | B | `src/traditional/features.py` | 颜色直方图/颜色矩/LBP/HOG 及融合 |
| `传统_C` | C | `src/traditional/bovw.py` | SIFT、k-means 码本、BoVW 编码、两种分类器 |
| `深度_D2` | D2 | `src/deep/experiments.py` | 从零训练 + 架构对比 |
| `深度_D3` | D3 | `src/deep/experiments.py` | 预训练迁移 + 增强消融 |
| `深度_D1` / `汇总_D1` | 你 | `src/config.py` `src/data.py` `src/metrics.py` `src/train.py` `src/report/aggregate.py` | 框架 + 核心对比 + 图表汇总 |

## 不要改的文件

`src/metrics.py` 是全组共用的评分口径。要改先在群里说,因为改了所有人都得重跑。

`src/config.py` 的 `BASELINE` 同理 —— 所有消融都是相对它的单变量变化,基准一动,整张消融表失效。

## 谁在等谁

```
A (manifest) ──┬── B, C  传统流水线
               └── D1 框架 ── D2, D3 深度实验 ── D1 汇总出图
```

A 交付前,所有人都用 `python tools/make_fake_data.py --classes 20 --out data_fake` 造的假数据先把代码跑通。
接口不变,A 一交付换个路径就能上真数据。

## 提交约定

每个方法跑完必须落一个 `results/<method>.npz`,通过 `save_result()` 产出。
报告里所有表格和混淆矩阵都从这些文件生成,不要自己画图往报告里贴。

`clf.decision_function(X)` 就能满足 top-5,**不要用 `SVC(probability=True)`**,500 类下慢到不可接受。
