# 源码阅读导航

先看[本阶段两页报告](../reports/progress_20261021/REPORT.pdf)和[实验协议](../reports/progress_20261021/PROTOCOLS.md)。下面区分已公开的原 strict 核心与早期训练工程；本轮 V2、结构/损失消融和 STLD/PIP 适配的完整生产入口尚未全部公开。

| 要查看的内容 | 实际源码入口 | 能核查的内容 |
|---|---|---|
| 原 strict 模型与普通 DSNT | [strict.py](../src/geoequi_ld/models/strict.py) | 灰度 512 输入、三通道 256 热图，独立 PS/FH 增强及解码分支；普通空间 Softmax 后取坐标期望 |
| HRNet 与两种解码器 | [hrnet.py](../src/geoequi_ld/models/hrnet.py) | timm HRNet-W32 最终融合高分辨率分支、共享/独立解码器；构造为 pretrained=False |
| PS/FH 专业增强 | [specialized.py](../src/geoequi_ld/models/specialized.py) | PS 调制变形卷积和空间注意力，FH ASPP-lite/SE；通道 LayerNorm 与残差连接 |
| 原 strict 损失原语 | [strict_losses.py](../src/geoequi_ld/training/strict_losses.py) | 有效点监督热图 MSE，以及已接纳点的对齐伪热图 MSE；不是当前 V2 的完整训练连接 |
| 坐标、变换与评价原语 | [coordinates.py](../src/geoequi_ld/geometry/coordinates.py)、[transforms.py](../src/geoequi_ld/geometry/transforms.py)、[aop.py](../src/geoequi_ld/geometry/aop.py)、[keypoints.py](../src/geoequi_ld/metrics/keypoints.py) | 归一化/像素坐标、相似变换、AoP 与点误差的实现；当前数字仍以阶段协议和聚合记录为准 |
| 原 strict 合成检查 | [test_strict_model.py](../tests/test_strict_model.py)、[test_strict_core.py](../tests/test_strict_core.py) | 分支/输出形状、普通 DSNT、MSE 与小合成 HRNet 特征检查；不等同真实医学数据评价 |
| 早期监督训练工程 | [train_baseline.py](../scripts/train_baseline.py)、[train_phase05.py](../scripts/train_phase05.py)、[train_phase06.py](../scripts/train_phase06.py) | 历史自定义 U-Net 与温度 DSNT 的基线、损失对照和长预算流程；不是本轮 Geo V2 或 STLD/PIP 的启动器 |

本轮[完整结果](../reports/progress_20261021/RESULTS.md)、[聚合数据](../reports/progress_20261021/metrics.json)及[协议](../reports/progress_20261021/PROTOCOLS.md)已公开；医学数据、逐例资料、模型权重、真实采样计划与未公开生产入口不在此导航中。[来源与许可范围](ATTRIBUTION_AND_RELEASE_SCOPE.md)单独说明。
