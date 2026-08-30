# IUGC 2025 官方 UNet Heatmap baseline 复现

> 状态：正式训练进行中。本文只记录官方 T10 baseline，不把仓库中早期的小型 U-Net、H1/H2/H3 或损失消融混入本次结果。

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

用当前 Kaggle 解压数据完成了一次真实样本 smoke test：数据读取、前向、反向、Adam 更新及 checkpoint 保存均成功，loss 为有限值。150 轮正式训练正在运行。

官方仓库没有提供 validation checkpoint 选择规则，只保存训练 loss 最优、训练坐标距离最优、每 50 轮和最终模型。为避免看过 validation 或 testing 后再挑模型，本次在评价前固定使用 `final_model.pth`（epoch 150）作为正式复现 checkpoint。

## 6. 官方报告指标

论文表 2 报告的 UNet Heatmap / T10 结果如下（像素距离越低越好，AoP 单位为度）：

| 划分 | MRE_PS1 | MRE_PS2 | MRE_FH1 | MRE_ALL | AoP absolute error |
|---|---:|---:|---:|---:|---:|
| Validation | 12.3408 | 21.5383 | 48.1807 | 27.35 | 10.47 |
| Testing | 10.6720 | 15.6234 | 39.1866 | 21.83 | 8.37 |

命名映射：PS1 对应 PSR（耻骨联合右端点），PS2 对应 PSL（另一端点），FH1 对应 FHT（胎头切点）。

## 7. 本次复现指标

待正式训练完成后填写。

## 8. 与官方结果的差值

待正式训练完成后填写。

## 9. 当前能确认的差异来源

待数值结果出来后，只保留 3—5 个最具体、能由运行证据支持的原因。

## 10. 下一步

完成 150 轮训练后，用预先固定的 epoch 150 checkpoint 评价 validation；随后对 testing 只评价一次，生成好、中、差三例本地可视化和一页导师汇报 PDF。真实医学图像、逐样本预测和 checkpoint 只保存在本地 `runs/`，不提交公开仓库。
