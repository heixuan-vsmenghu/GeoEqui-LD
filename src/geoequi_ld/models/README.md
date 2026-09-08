# 模型与读出版本

当前首先查阅 [strict.py](strict.py) 中的 `StrictGeoEquiLDModel` 和 `StandardDSNT`，以及[严格热图 MSE](../training/strict_losses.py)。本目录收录真实核心源码及其 [HRNet/解码器](hrnet.py)、[PS/FH 增强](specialized.py)依赖。研究结果见[监督结果摘录](../../../reports/review_20260908/SUPERVISED_RESULTS.md)和[热图与 DSNT 核对](../../../reports/review_20260908/HEATMAP_DSNT_CHECK.md)。

| 路线 | 代码位置 | 实际读出 |
|---|---|---|
| 当前 strict 核心 | [strict.py](strict.py) | `StandardDSNT.forward` 调用 `standard_spatial_softmax`，直接 `F.softmax`，再调用 `spatial_expectation`；没有 temperature 参数 |
| 早期监督探索 | 本目录 [dsnt.py](dsnt.py)、[unet.py](unet.py) 及旧训练脚本 | `DSNT` 类默认 temperature=0.05，forward 使用温度缩放；这是历史实现 |

strict 复用 `dsnt.py` 中的坐标期望函数，不意味着调用了旧 `DSNT` 类。底层 `spatial_softmax` 函数自身默认 temperature=1.0，亦需与 `DSNT` 类的 0.05 默认值区分。旧函数签名、参数和计算均保留，供[历史追溯](../../../docs/history/RESEARCH_INDEX.md)。

HRNet 和 specialized 文件同时保留原有历史对照类，当前 strict 直接复用其中的主干、独立解码器和增强模块，不通过删旧类制造新版本。这里收录的是当前工作区源码，不证明与 epoch 39 或 Stage 2 两轮的历史运行字节逐位一致。

本次仅集成模型、损失原语和必要依赖，不含长训练启动器、数据清单、权重或完整实验恢复环境。模型构造使用 `pretrained=False`；所附测试只使用合成输入，不是训练或真实数据评价。

完整当前调用关系见[当前进展](../../../docs/CURRENT_PROGRESS.md#读出版本与实际调用)，配置来源见[配置说明](../../../configs/README.md)。
