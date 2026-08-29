# GeoEqui-LD

GeoEqui-LD 是一个面向产时超声三关键点检测的毕业设计工程。项目最终希望研究几何等变一致性能否帮助模型利用无标签图像；Phase 0 基础闭环已经冻结，Phase 0.5/0.6 的监督损失审计和 Phase 1A 的 B0 异常诊断、HRNet 监督参考接入也已完成。

目前没有把完整半监督方案包装成“已经实现”：本地尚未取得完整无标签池，因此 EMA 教师、伪标签筛选和无标签几何一致性训练都还没有开始。

## 当前任务

模型输入一张经会阴超声灰度图，按固定顺序预测三个点：

```text
KEYPOINT_ORDER = (PS1, PS2, FH1)
```

- `PS1`、`PS2`：耻骨联合两个端点；
- `FH1`：胎头切点；
- `AoP`：以 PS1 为顶点，由 `PS1→PS2` 与 `PS1→FH1` 构成的无向夹角。

默认张量约定：

```text
输入图像     [B, 1, 512, 512]
预测热图     [B, 3, 256, 256]
DSNT 坐标    [B, 3, 2]，顺序为 [x, y]，范围为 [-1, 1]
```

完整坐标说明见 [docs/COORDINATE_CONVENTION.md](docs/COORDINATE_CONVENTION.md)。

## 已经完成的部分

- 有标签数据只读审计与脱敏统计；
- 显式的 `[x, y]` 像素坐标和 DSNT 归一化坐标转换；
- 关键点高斯热图生成，`sigma=4` 表示热图像素；
- 相似变换的正变换、逆变换和跨视图点映射；
- 稳定的 AoP 和 MRE 计算；
- 可微 DSNT；
- DSNT 概率图的 Gaussian 分布约束，避免只优化坐标却得到不可解释的响应图；
- 一个输入单通道、输出三通道半分辨率热图的小型 U-Net；
- 坐标、热图、DSNT、AoP、变换、指标、访问策略、模型、监督训练和结果汇总均有自动化测试；
- 4 张真实训练图上的 tiny-overfit 门槛已经通过。
- 300/100 监督 baseline 已完成，并在方案冻结后对 testing 评估一次。

这些内容说明基础工程可以继续做监督阶段验证，不代表半监督方法或最终论文实验已经完成。

## Phase 0.5 监督消融

Phase 0.5 固定 Phase 0 的数据、轻量 U-Net、随机种子规则和 20 轮预算，只比较三种损失：

| 方案 | 损失 |
|---|---|
| B0 | heatmap MSE |
| B1 | heatmap MSE + coordinate SmoothL1 |
| B2 | heatmap MSE + coordinate SmoothL1 + distribution JS |

三组统一使用 validation 上的 DSNT AoP MAE 选择 checkpoint；B0 还会在同一个 checkpoint 上比较 argmax 和 DSNT。首轮 seed=42，按预先固定的 AoP MAE、MRE_ALL、复杂度顺序留下两个方案，再用全新的 seeds 43/44/45 做确认，避免把筛选用的 seed 42 混进稳定性均值。Phase 0.5 的入口没有 testing 参数，并通过本地数据指纹和访问策略拒绝 testing。

公开协议是 [configs/phase05_ablation.yaml](configs/phase05_ablation.yaml)。本地运行时复制 [configs/phase05_local.example.yaml](configs/phase05_local.example.yaml) 为被 Git 忽略的 `configs/phase05_local.yaml`，再运行：

```powershell
python scripts/train_phase05.py --protocol configs/phase05_ablation.yaml `
  --local-config configs/phase05_local.yaml --variant B0 --seed 42 `
  --output-dir runs/phase05/B0/seed_42
```

实验输出保存在被 Git 忽略的 `runs/`；公开仓库只接收脱敏聚合指标和曲线。

三个新种子（43/44/45）的确认结果如下；seed 42 只用于筛选，没有混入均值：

| 方案 | MRE_ALL（均值 ± 样本 SD） | AoP MAE（均值 ± 样本 SD） |
|---|---:|---:|
| B1 | 39.896 ± 1.170 px | 12.975 ± 0.768° |
| B2 | 24.973 ± 2.282 px | 9.251 ± 2.075° |
| B2 − B1（同种子配对） | −14.923 ± 2.953 px | −3.724 ± 2.810° |

逐种子结果、B0 的 DSNT/argmax 检查、运行时间、显存和限制见
[Phase 0.5 小结](reports/phase05/PHASE05_SUMMARY.md)与
[完整监督消融记录](reports/phase05/SUPERVISED_ABLATION.md)。这里只作 3 个确认种子的 validation 描述性复核，不声称统计显著或测试集结论。

Phase 0 原始 20 轮运行对应的脱敏快照保存在 [configs/phase0_frozen_20e.yaml](configs/phase0_frozen_20e.yaml)，不再用早期的 150 轮通用模板代替实际运行配置。

## Phase 0.6 长预算检查

Phase 0.6 没有改模型或调损失权重，只把 seed 42 下的 B0/B1/B2 用完全相同的
初始化、数据顺序和优化设置跑满 200 轮，用来判断 20 轮结果是不是单纯由收敛速度
造成。checkpoint 仍只按 validation AoP MAE、MRE_ALL、较早 epoch 依次选择，
testing 全程冻结。

| 方案 | best epoch | MRE_ALL | AoP MAE |
|---|---:|---:|---:|
| B0：MSE | 120 | 148.471 px | 20.551° |
| B1：MSE + coordinate SmoothL1 | 194 | 27.084 px | 8.767° |
| B2：B1 + JS | 15 | 24.779 px | 8.514° |

B0 在 20 轮后确实还有改善，但没有追上 B1/B2，并在 epoch 187–200 连续出现
0 个有效 AoP 预测。B1 的优势一直保持到 epoch 200；B2 很早取得更好的 selected
best，但 epoch 200 的两项主指标反而略差于 B1，因此 JS 在这次单 seed 检查中主要
体现为加快收敛，不能据此声称改善长期稳定性。

完整数字和边界见 [Phase 0.6 小结](reports/phase06/PHASE06_SUMMARY.md)、
[200 轮逐节点对照](reports/phase06/LONG_BUDGET_COMPARISON.md)和
[validation 曲线](reports/phase06/curves/validation_metrics.png)。这里只是增强监督
基线审计，不包含 HRNet、PS/FH 解耦、EMA、伪标签或半监督损失。

## Phase 1A：B0 诊断与 HRNet 监督参考

Phase 1A 先回看纯 MSE 的异常，再按老师资料里的主干约定接入 HRNet-W32。B0
第 200 轮的三张 raw heatmap 已经变成空间常数，DSNT 因此把三个点都解到中心，
有效 AoP 降到 0/100。合成高斯检查同时确认：当前 DSNT 计算和坐标接口本身能工作，
但热图振幅过低或过平时，大面积背景概率会把期望坐标拉向图像中心。由于没有保存
崩溃转折区间的 checkpoint，这里只描述端点现象，不把原因写死。

HRNet 使用 `timm==1.0.28`、单通道输入和 stage4 最终融合的高分辨率分支，后接
共享三通道热图头。结构探针在 512×512、batch 1、FP32 下通过，峰值 reserved
显存为 1.22 GiB；固定四样本跑满 500 步后，eval MRE_ALL 为 4.609 px，4/4 AoP
有效，叠加图人工检查也没有发现坐标错位。

20 轮监督参考完整跑完，validation checkpoint 仍按 AoP MAE、MRE_ALL、较早 epoch
依次选择：

| checkpoint | epoch | MRE_ALL | AoP MAE | 有效 AoP |
|---|---:|---:|---:|---:|
| best | 3 | 32.391 px | 12.130° | 100/100 |
| last | 20 | 39.642 px | 23.109° | 100/100 |

第 3 轮之后 validation 波动明显。batch size 1 下的 BatchNorm 是一个需要留意的
风险，但现有结果没有证明它就是波动原因。旧 U-Net B2 的 24.779 px / 8.514°只作
量级参考；两者架构不同，不据此作因果比较。详细过程见
[Phase 1A 小结](reports/phase1a/PHASE1A_SUMMARY.md)、
[HRNet 接入记录](reports/phase1a/HRNET_IMPLEMENTATION.md)和
[validation 曲线](reports/phase1a/curves/validation_metrics.png)。目前仍未实现
PS/FH 解耦、EMA、伪标签或半监督损失。

## 数据现状

当前可核验的监督部分是官方公开划分：

| 分区 | 图像数 | 状态 |
|---|---:|---|
| Train labeled | 300 | 可用于监督训练 |
| Validation | 100 | 官方公开验证集 |
| Testing | 501 | 仅用于冻结方案后的最终评估 |
| Unlabeled pool | 0（当前本地可用） | 缺失，阻塞半监督训练 |

监督数据本身已通过文件、标签、解码、边界和内容哈希检查。更完整的公开安全统计见：

- [reports/phase0/DATA_AUDIT.md](reports/phase0/DATA_AUDIT.md)
- [reports/phase0/dataset_statistics.json](reports/phase0/dataset_statistics.json)
- [reports/phase0/duplicate_report.csv](reports/phase0/duplicate_report.csv)

无标签数据和分组元数据的需求见 [reports/phase0/DATA_REQUIRED.md](reports/phase0/DATA_REQUIRED.md)。

## 当前监督结果

| Split | MRE_ALL | AoP MAE |
|---|---:|---:|
| Validation best（epoch 15） | 24.779 px | 8.514° |
| Testing frozen once | 19.930 px | 7.553° |

三点和风险说明见 [BASELINE_REPORT.md](reports/phase0/BASELINE_REPORT.md)，完整阶段总结见 [PHASE0_SUMMARY.md](reports/phase0/PHASE0_SUMMARY.md)。testing 结果没有用于调参；其 501 条记录只对应 493 种唯一图像内容，且患者/视频独立性尚未确认。

## 数据不会进入公开仓库

IUGC 数据包内协议与公开页面的许可表述并不完全一致，而且协议对数据复制、衍生内容和签署流程有限制。因此本仓库只保存代码、合成测试和脱敏聚合结果，不包含：

- 原始图像、标签 CSV 或数据压缩包；
- 带真实图像的可视化；
- 逐样本预测、伪标签或可还原个体数据的缓存；
- 任何本机绝对数据路径。

真实数据应存放在仓库之外，并通过运行时参数传入。公开材料前仍需由导师或数据所有者确认许可范围。

## 运行测试

在 Python 3.11 环境中安装 PyTorch、NumPy、Pandas、Pillow、PyYAML 和 pytest 后运行：

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q -p no:cacheprovider
```

测试数量会随阶段增加，以本地 `pytest` 与 GitHub Actions 的当次输出为准。

## 当前边界

- 尚未完成完整无标签数据审计；
- tiny-overfit 和最小监督 baseline 已完成；完整半监督方法尚未开始；
- 尚未实现 EMA 教师、伪标签或几何一致性损失；
- 标签中没有患者、病例或视频分组字段，不能证明各 split 在患者级独立；
- 当前任务是关键点检测和 AoP 定量，不是直接预测分娩结局。

Phase 0 的工程决策记录在 [reports/phase0/DECISIONS.md](reports/phase0/DECISIONS.md)。

## 参考来源

- IUGC 2025 官方代码：https://github.com/0oTyTo0/IUGC2025
- IUGC 2025 数据页：https://www.kaggle.com/datasets/aspirexxx/iugc-ultrasound-dataset-miccai-2025
- 官方监督 baseline 论文：https://openreview.net/attachment?id=hj7hmvKc2r&name=pdf
