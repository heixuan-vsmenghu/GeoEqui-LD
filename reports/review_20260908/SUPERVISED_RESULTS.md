# 监督训练结果摘录

本文依据既有实验记录整理，供查阅当前监督结果；不是新实验、原始日志或完整训练曲线。[当前模型和损失核心](../../src/geoequi_ld/models/README.md)已收录，strict 原报告和完整训练运行系统尚未公开。本文未重新评价任何数据，也未改动原记录。

## 当前主结果：200 轮已完成，但未稳定收敛

strict GeoEqui-LD 使用 300 张有标签 train、100 张 validation，完成总计 200 轮纯监督预热。结构为单通道 512×512 输入、HRNet-W32、PS/FH 专业增强与独立解码器、3×256×256 热图输出。物理 batch=2，每轮完整遍历 300 张图、150 次 Adam 更新；seed=42、lr=0.0001、weight_decay=0.0001、clip=5、sigma=4 高斯 raw heatmap MSE。没有 scheduler、额外数据增强、EMA、几何项或伪标签项。

200 轮记录的接续部分从 epoch 30 完整训练状态恢复至 epoch 200。主选模指标为标准 `model.eval()` 下、自然累积 raw-BN 统计对应的 validation hard-argmax MRE_ALL；可恢复候选范围为 epoch 30 起点及 epoch 31–200。以下每行均来自该行同一 checkpoint，未跨 epoch 拼接：

| checkpoint / 评价 | PS1 px | PS2 px | FH1 px | MRE_ALL px | AoP MAE | AoP 数值可计算 | validation heatmap MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| epoch 39 selected，raw-BN hard argmax | 18.728 | 26.443 | 48.006 | 31.059 | 12.312° | 100/100 | 0.000591 |
| epoch 200 final，raw-BN hard argmax | 91.449 | 104.913 | 82.204 | 92.855 | 50.086° | 97/100 | 0.000865 |

这里 px 均为 512×512 输入坐标系的像素距离；AoP 数值可计算只表示已有有限值/非零射线规则通过，不等于几何结构可靠。selected 的 FH1 仍是同轮最大误差来源。final 明显差于 selected，不能把“训练满 200 轮”写成稳定收敛或完整半监督成功。31.059 px 是 validation argmax 结果，不是 Testing、DSNT 或半监督终态结果。

来源：仅本地、未提交的 `reports/strict_model_stage1/STRICT_STAGE1_200E.md` 第 1、3–6 节；第 2 节保存 epoch 31–200 逐轮聚合表。本摘录不包含该完整逐轮表。原报告尚无可引用的 Git 提交，不为其补造历史版本。

## 同一 selected 的读出差异

同一 epoch 39、同一 100 张 validation、同一 raw-BN 状态，普通空间 Softmax DSNT 的 MRE_ALL=153.767 px、AoP MAE=37.293°、归一化熵=0.999992；相对 argmax 的 MRE_ALL 差值为 122.708 px。隔离 BN 校准副本的 argmax ALL=29.399 px 只作辅助诊断，不替换 raw selected，也不与 raw 指标拼成一套成绩。

来源：上述原报告第 7–8 节。公式、人工高斯检查和后续归因边界见[热图与 DSNT 摘录](HEATMAP_DSNT_CHECK.md)。

## 官方 T10：流程完成，论文数值未完全复现

官方实现来源为 [IUGC2025 固定提交 `bc8fce2032c000c2569e916268ab918c0905ab4e`](https://github.com/0oTyTo0/IUGC2025/tree/bc8fce2032c000c2569e916268ab918c0905ab4e)。其官方 `HeatmapUNet` 使用 RGB 512×512 输入、64×64 热图、sigma=2、hard argmax、batch=4、Adam 和 StepLR，完成 150 轮。它既不是仓库早期的小型 U-Net，也不是上述 strict 模型。

本地 T10 复现预先固定 epoch 150 `final_model.pth` 为正式评价 checkpoint，而非查看 validation 后再选择最优轮。既有评价顺序为先 100 张 validation，再对 501 张 Testing 进行冻结评价；以下只是既有聚合记录的摘录，本轮没有读取 Testing 或重新评价。

| 来源 / checkpoint | split | MRE_ALL px | AoP 误差 |
|---|---|---:|---:|
| 官方论文表 2，T10 | Validation | 27.35 | 10.47° |
| 本地复现，epoch 150 final | Validation | 31.6133 | 15.7964° |
| 官方论文表 2，T10 | Testing | 21.83 | 8.37° |
| 本地复现，epoch 150 final | Testing | 23.7044 | 10.4706° |

本地 validation 中 99/100 个预测可计算 AoP，valid-only MAE=14.1377°；表中的 15.7964° 对一个退化预测计入 180° 惩罚，保留 100 张分母。这是本地既有保守聚合选择，不是论文明确规定。Testing 为 501/501 数值可计算。论文没有明确表 2 的 checkpoint 选择规则，因此“运行链路已复现”不等于论文数值或选择协议已完全复现。

来源：[本仓库官方 baseline 报告固定版本](https://github.com/heixuan-vsmenghu/GeoEqui-LD/blob/e3af9e2a2c29e5de3beab912fe1ecf7592300f8b/reports/baseline_reproduction/BASELINE_REPRODUCTION.md)，原路径 `reports/baseline_reproduction/BASELINE_REPRODUCTION.md`，第 1、3、5–9 节。该报告位于上述既有提交，不假定它存在于当前 main 的相对目录。

## 可比较范围

- T10 的评价前固定 final 与 strict 的 validation-selected 不是同一选模协议；validation 与 Testing 也不是同一 split。仅作量级对照，不据 31.059 与 31.6133 的差值宣称 strict 超过 baseline。
- strict 主结果来自 seed=42 的单次运行，没有在本摘录中增加多种子统计或补做实验。
- 原 Stage 2 从 epoch 39 接续后，student raw-BN validation argmax ALL 在 epoch 1/2 退化到 267.830/304.203 px，实际在 epoch 2 停止。完成监督预热不意味着联合训练已成功；后续短程对照的限制见[热图与 DSNT 摘录](HEATMAP_DSNT_CHECK.md)。

[返回当前进展](../../docs/CURRENT_PROGRESS.md) · [导师查阅索引](../../docs/ADVISOR_REVIEW_INDEX.md)
