# Phase 1C：H1 / H2 / H3 监督对照

本轮只比较有标签监督工程参照。H1 是共享头，H2 是 PS/FH 独立头，H3 在 H2 基础上增加 PS 与 FH 专业增强模块；三者都不是完整半监督 GeoEqui-LD。误差越低越好。

## selected best

| 模型 | epoch | PS1 | PS2 | FH1 | MRE_ALL | AoP MAE |
|---|---:|---:|---:|---:|---:|---:|
| H1 共享头 | 3 | 22.483 | 27.854 | 46.837 | 32.391 | 12.130 |
| H2 独立头 | 3 | 17.564 | 24.193 | 51.797 | 31.185 | 13.563 |
| H3 专业增强 | 14 | 12.426 | 21.446 | 40.831 | 24.901 | 10.289 |

H3 的选择节点是 epoch 14，H1 与 H2 都是 epoch 3。H3 相对 H2 的差值为 PS1 -5.138 px、PS2 -2.747 px、FH1 -10.967 px、MRE_ALL -6.284 px、AoP MAE -3.274°。这一组数字全部更低，但节点轮次不同，且来自同一 validation 的选择与汇报，只能作为单次描述。

## matched epoch 3

| 模型 | epoch | PS1 | PS2 | FH1 | MRE_ALL | AoP MAE |
|---|---:|---:|---:|---:|---:|---:|
| H1 共享头 | 3 | 22.483 | 27.854 | 46.837 | 32.391 | 12.130 |
| H2 独立头 | 3 | 17.564 | 24.193 | 51.797 | 31.185 | 13.563 |
| H3 专业增强 | 3 | 13.981 | 26.938 | 49.239 | 30.053 | 11.421 |

H3 相对 H2：PS1 -3.583 px，PS2 +2.745 px，FH1 -2.559 px，MRE_ALL -1.132 px，AoP MAE -2.142°。这里 PS2 反而更高。

## matched epoch 16

| 模型 | epoch | PS1 | PS2 | FH1 | MRE_ALL | AoP MAE |
|---|---:|---:|---:|---:|---:|---:|
| H1 共享头 | 16 | 13.077 | 27.981 | 46.004 | 29.021 | 13.931 |
| H2 独立头 | 16 | 17.407 | 31.110 | 36.739 | 28.419 | 13.933 |
| H3 专业增强 | 16 | 13.904 | 23.274 | 40.631 | 25.936 | 14.279 |

H3 相对 H2：PS1 -3.503 px，PS2 -7.836 px，FH1 +3.891 px，MRE_ALL -2.483 px，AoP MAE +0.346°。这里 FH1 与 AoP MAE 反而更高。

## 逐点回答

- PS1：H3 在 selected best、epoch 3、epoch 16 都低于 H2；但 H3 同时加入两个增强分支，不能把差值直接解释成 PS 模块的独立因果收益。
- PS2：selected best 与 epoch 16 更低，epoch 3 更高，方向不统一。
- FH1：selected best 与 epoch 3 更低，epoch 16 更高，方向不统一。
- MRE_ALL：三组对照均低于 H2，是这次运行最稳定的正向信号。
- AoP MAE：selected best 与 epoch 3 更低，epoch 16 更高，不能称为全程改善。
- train–validation gap：H3 的 MRE_ALL gap 相对 H2 在 epoch 3 扩大 +4.142 px，在 epoch 16 缩小 -0.906 px；没有统一收窄。

所以准确结论是：H3 的 selected best 和 MRE_ALL 结果支持继续研究，但三个关键点没有在所有对齐轮次上一致改善。当前只有 seed 42，不做显著性、稳定胜出或 SOTA 声明。

曲线见 [validation_metrics.png](curves/validation_metrics.png) 与 [h3_train_validation_gap.png](curves/h3_train_validation_gap.png)。旧 U-Net B2 仅保留为历史量级参照，架构与预算不同，不用于证明 HRNet 或 H3 的结构优劣。
