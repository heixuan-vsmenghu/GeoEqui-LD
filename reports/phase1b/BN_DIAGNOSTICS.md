# Phase 1B：BatchNorm 短诊断

这次只做了一个很窄的检查：固定 H1 的 best（epoch 3）和 last（epoch 20）权重，不训练参数，只用 300 张 train 图像重新累计一次 BatchNorm 运行统计，然后重新看 validation。原 checkpoint 没有被覆盖，重估结果也没有参与选模。

| checkpoint | BN 状态 | PS1 MRE | PS2 MRE | FH1 MRE | MRE_ALL | AoP MAE |
|---|---|---:|---:|---:|---:|---:|
| best / e3 | 原 BN | 22.483 | 27.854 | 46.837 | 32.391 | 12.130 |
| best / e3 | train 图像重估 | 35.332 | 43.668 | 42.345 | 40.448 | 16.142 |
| last / e20 | 原 BN | 20.148 | 54.646 | 44.131 | 39.642 | 23.109 |
| last / e20 | train 图像重估 | 17.309 | 32.553 | 39.969 | 29.944 | 14.820 |

## 原 BN 下的 validation–train 差距

| checkpoint | split | PS1 MRE | PS2 MRE | FH1 MRE | MRE_ALL | AoP MAE |
|---|---|---:|---:|---:|---:|---:|
| best / e3 | train | 14.512 | 18.822 | 38.710 | 24.015 | 9.829 |
| best / e3 | validation | 22.483 | 27.854 | 46.837 | 32.391 | 12.130 |
| best / e3 | validation − train | 7.970 | 9.032 | 8.127 | 8.377 | 2.301 |
| last / e20 | train | 13.367 | 30.055 | 27.451 | 23.624 | 15.299 |
| last / e20 | validation | 20.148 | 54.646 | 44.131 | 39.642 | 23.109 |
| last / e20 | validation − train | 6.782 | 24.591 | 16.680 | 16.018 | 7.811 |

e3 的 validation−train 差距在 PS2/FH1 上分别是 9.032/8.127 px；到 e20 扩大为 24.591/16.680 px。AoP 差距也从 2.301° 增至 7.811°。这描述了后期泛化差距，但仍不能单独确定成因。

数值方向并不一致。best checkpoint 重估后，PS2 从 27.854 变成 43.668 px，变差 15.814 px；FH1 反而改善 4.493 px。last checkpoint 的 PS2 和 FH1 则分别改善 22.093 和 4.162 px，整体 MRE 与 AoP 也下降。

所以目前比较稳妥的说法是：batch size 1 下，BN 运行统计确实会影响 validation 表现，可能是波动来源之一；但一次重估既能改善某个端点，也会损害另一个端点，不能据此把 H1 的全部波动都归因于 BN。参数审计确认权重未变，重估阶段只有 BN 运行统计发生变化。

这项检查只使用 train 图像更新统计并在 validation 上汇总，testing 没有读取或评估。
