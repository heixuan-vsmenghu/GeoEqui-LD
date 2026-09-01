# IUGC 2025 官方 UNet Heatmap baseline 复现

> 状态：正式训练与冻结评价均已完成。本文只记录官方 T10 baseline，不把仓库中早期的小型 U-Net、H1/H2/H3 或损失消融混入本次结果。

## 1. 官方来源与 commit

- 官方代码：[0oTyTo0/IUGC2025](https://github.com/0oTyTo0/IUGC2025)，固定在 commit `bc8fce2032c000c2569e916268ab918c0905ab4e`。
- 官方论文：[IUGC: A Benchmark of Landmark Detection in End-to-End Intrapartum Ultrasound Biometry](https://openreview.net/pdf?id=hj7hmvKc2r)。
- 导师指定数据：[Kaggle - IUGC Ultrasound Dataset (MICCAI 2025)](https://www.kaggle.com/datasets/aspirexxx/iugc-ultrasound-dataset-miccai-2025/data?select=Dataset)。
- 当前仓库通过 `third_party/IUGC2025` 子模块引用官方代码，没有把仓库中已有的自定义 U-Net 当作官方 baseline。

## 2. 数据划分

数据由 Kaggle CLI 从导师给出的页面正常下载为 ZIP，并使用 Windows `Expand-Archive` 正常解压。本次实际读到的目录如下：

| 划分 | 当前平台包中的图像数 | 本轮用途 |
|---|---:|---|
| Training / Labeled cases | 300 | 正式训练 |
| Training / Unlabeled cases | 33,466 | 不使用 |
| Validation | 100 | 训练结束后评价 |
| Testing | 501 | checkpoint 冻结后只评价一次 |

训练标注文件为 `Training/Labeled cases/label.csv`；validation 和 testing 各自包含 `landmarks_data.csv` 与 `aop_results.csv`。测试标注在正式训练完成且 checkpoint 口径冻结前不读取。

## 3. 官方配置

本次完整训练直接运行官方 `heatmap_train_only.py`，配置保持为：

| 项目 | 设置 |
|---|---|
| 模型 | 官方 `HeatmapUNet`，RGB 输入，3 通道热图输出 |
| 输入 | Resize 到 512×512，`ToTensor()`，不做均值方差归一化 |
| 热图 | 64×64，Gaussian sigma = 2.0 |
| 关键点顺序 | PS1、PS2、FH1 |
| 损失 | 纯 heatmap MSE |
| 解码 | 每个热图 hard argmax，按 64→512 比例映射 |
| batch size | 4 |
| epochs | 150 |
| optimizer | Adam |
| learning rate | 1e-4 |
| weight decay | 1e-4 |
| scheduler | StepLR，step=15，gamma=0.5 |
| seed | 42（沿用官方脚本的 PyTorch、NumPy 设置） |

没有加入 coordinate SmoothL1、JS、DSNT、PS/FH 解耦、注意力、几何损失、EMA 或伪标签。

## 4. 实际运行环境

- Windows，Python 3.11.4
- PyTorch 2.5.1+cu121，torchvision 0.20.1
- NVIDIA GeForce GTX 1650，4 GB 显存
- 本机完整 batch 4 会使用 WDDM 共享显存，速度明显慢于论文使用的 RTX 2080 Ti；这只影响运行时间，不改训练配置。

## 5. 训练是否完成

已完成 150/150 轮正式训练。日志中的 epoch 连续为 1—150，完整一轮均为 75 个 batch；纯训练进度合计约 41.41 小时，端到端墙钟时间约 42 小时 20 分。单轮平均约 16 分 34 秒，最慢一轮 18 分 38 秒，均低于预先规定的 20 分钟速度上限。日志中没有 traceback、NaN 或 Inf。

官方仓库没有提供 validation checkpoint 选择规则，只保存训练 loss 最优、训练坐标距离最优、每 50 轮和最终模型。为避免看过 validation 或 testing 后再挑模型，本次在评价前固定使用 `final_model.pth`（epoch 150）作为正式复现 checkpoint。它与 `model_epoch_150.pth` 字节完全一致，SHA-256 为 `FC1C5CDAE752EE5E1A1BE5980885209E03C12B06A40AF7C2561A7A6046CCEE44`。

评价顺序也在打开 testing 前冻结：先完整评价 100 张 validation，再对 501 张 testing 只运行一次。testing 的一次性本地锁和逐样本输出均保留在忽略目录中，不进入 Git。

## 6. 官方报告指标

论文表 2 报告的 UNet Heatmap / T10 结果如下（像素距离越低越好，AoP 单位为度）：

| 划分 | MRE_PS1 | MRE_PS2 | MRE_FH1 | MRE_ALL | AoP absolute error |
|---|---:|---:|---:|---:|---:|
| Validation | 12.3408 | 21.5383 | 48.1807 | 27.35 | 10.47 |
| Testing | 10.6720 | 15.6234 | 39.1866 | 21.83 | 8.37 |

命名映射：PS1 对应 PSR（耻骨联合右端点），PS2 对应 PSL（另一端点），FH1 对应 FHT（胎头切点）。

## 7. 本次复现指标

| 划分 | MRE_PS1 | MRE_PS2 | MRE_FH1 | MRE_ALL | AoP absolute error |
|---|---:|---:|---:|---:|---:|
| Validation | 11.4266 | 22.8057 | 60.6077 | 31.6133 | 15.7964° |
| Testing | 10.6306 | 15.7768 | 44.7056 | 23.7044 | 10.4706° |

Validation 中有 99/100 个预测可计算 AoP，1 个预测的 PS1 顶点射线退化；99 个有效预测的 AoP MAE 为 14.1377°。Testing 为 501/501 有效。官方论文只给出 AoP 与绝对误差公式，没有规定零长度射线的聚合办法。本次沿用仓库既定的保守口径：无效预测对全 split 分数贡献 180°，同时保留有效样本均值、有效数和无效数。因此表中的 validation `15.7964°` 是保持 100 张分母的保守惩罚分数，不把 99 张的 valid-only 均值冒充完整 validation 指标，也不宣称 180° 是官方规定。

## 8. 与官方结果的差值

下表为“本次复现值 − 官方报告值”；正数表示误差更大，负数表示误差更小。

| 划分 | ΔMRE_PS1 | ΔMRE_PS2 | ΔMRE_FH1 | ΔMRE_ALL | ΔAoP absolute error |
|---|---:|---:|---:|---:|---:|
| Validation | -0.9142 | +1.2674 | +12.4270 | +4.2633 | +5.3264° |
| Testing | -0.0414 | +0.1534 | +5.5190 | +1.8744 | +2.1006° |

PS1 已接近或略低于官方表值，但总体 MRE 与 AoP 均未追上，差距主要集中在 FH1。**运行链路已复现，数值尚未复现。**

## 9. 当前能确认的差异来源

1. 官方材料没有说明表 2 使用 final、训练 loss best、训练坐标 best 或其他 validation 选择 checkpoint；本次为避免事后挑选，预先固定 epoch 150 final。
2. Validation 出现 1 个 argmax 后的退化 AoP；论文没有给出这种情况的计分规则。本次 180° 是仓库既定的保守惩罚，不是从官方材料补造的规则。
3. 数值差距集中在 FH1：validation 与 testing 分别比官方高 12.4270 px 和 5.5190 px；PS1 基本追平，说明不是三个通道等幅偏移。
4. 论文使用 RTX 2080 Ti；本次是 Windows/WDDM 下的 GTX 1650 4 GB，PyTorch/CUDA 版本也未由官方完整冻结。它们不改变配置，但单次随机训练的数值路径不能假定逐位一致。
5. 当前 Kaggle 包的 train/validation/testing 数量与任务划分吻合，但上游没有发布可用于逐文件比对的 split 指纹，因此无法证明数据字节与论文运行时版本完全相同。

## 10. 下一步

本轮 baseline 已关闭：epoch 150 checkpoint、validation/testing 聚合指标、逐样本预测、好/中/差三例可视化以及单页导师 PDF 均已在本地生成。真实医学图像、checkpoint、逐样本预测、三例图和包含真实图像的 PDF 只保存在忽略目录，不提交公开仓库。

后续 HRNet 与半监督研究作为独立阶段开展，不回写或改造本次官方 T10 baseline 的模型、损失、解码和结果口径。
