# Phase 0 监督 baseline 结果

运行日期：2026-08-28

这是一条用来验证数据和训练闭环的最小监督基线，不是完整 GeoEqui-LD。训练只使用 300 张 labeled train，checkpoint 只由官方 100 张 validation 选择；方案冻结后，testing 仅评估一次。

## 1. 实验设置

| 项目 | 设置 |
|---|---|
| 输入 / 输出 | `1×512×512` 灰度图 / `3×256×256` 热图 |
| 关键点顺序 | `PS1, PS2, FH1` |
| 模型 | 小型 U-Net，GroupNorm，484,171 个可训练参数 |
| 解码 | spatial softmax，temperature 0.05，DSNT 连续坐标 |
| 监督损失 | `MSE + 10×SmoothL1(coord) + JS(probability, Gaussian)` |
| Gaussian | sigma = 4 heatmap px |
| 优化器 | Adam，lr = 0.001，weight decay = 0.0001 |
| batch / epoch | 1 / 20 |
| 随机种子 | 42 |
| 选模指标 | validation `aop_mae_deg`，越小越好 |
| 训练代码版本 | `d421cab667319070a44d5155c6abc0153925d6b3` |

本轮没有使用 testing 选 epoch、损失权重或阈值，也没有加入 HRNet、EMA、伪标签、无标签一致性或水平翻转。

## 2. Validation 结果

best checkpoint 出现在 epoch 15：

| 指标 | 结果 |
|---|---:|
| MRE_PS1 | 13.417 px |
| MRE_PS2 | 20.357 px |
| MRE_FH1 | 40.564 px |
| MRE_ALL | 24.779 px |
| AoP MAE | 8.514° |
| 有效 AoP | 100 / 100 |

最后一轮的 MRE_ALL 为 25.886 px、AoP MAE 为 10.562°，所以结果使用 validation 选出的 epoch 15，而不是 last checkpoint。

## 3. 冻结后的单次 Testing 结果

| 指标 | 结果 |
|---|---:|
| MRE_PS1 | 11.078 px |
| MRE_PS2 | 13.855 px |
| MRE_FH1 | 34.858 px |
| MRE_ALL | 19.930 px |
| AoP MAE | 7.553° |
| 有效 AoP | 501 / 501 |

这组 testing 数值只是冻结方案的一次结果，后续不能再用它挑参数。testing 的 501 条记录只对应 493 种唯一图像内容，且缺少患者/视频分组字段；因此它不能被解释为 501 个患者级独立样本。完整风险见 [DATA_AUDIT.md](DATA_AUDIT.md)。

## 4. 观察

- PS1 和 PS2 已能得到较稳定的局部概率峰；
- FH1 仍是误差最大的点，也是下一阶段最值得针对的部分；
- 训练损失持续下降，但 validation 在 epoch 15 后波动，没有理由用最后一轮替代 best；
- 4 样本 tiny-overfit 达到 1.049 px，并不意味着全量 validation 也能接近该数字，二者分别检验可学习性与泛化；
- 当前结果足以证明 Phase 0 监督链路可运行，但不能证明几何一致性或半监督策略有效。

## 5. 本地产物

完整运行产物位于 Git 忽略目录 `runs/phase0/baseline_20e_20260828/`，包括：

```text
config.yaml
environment.txt
train_log.csv
metrics.json
test_metrics_frozen_once.json
best.pt
last.pt
curves/
predictions/
```

checkpoint、真实图像叠加图和逐样本预测未进入公开仓库。是否允许公开这些衍生产物，需要导师或数据所有者确认。
