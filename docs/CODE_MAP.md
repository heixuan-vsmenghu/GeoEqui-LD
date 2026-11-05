# 源码阅读导航

先看[11月5日阶段报告](../reports/progress_20261105/REPORT.md)、[关键对照表](../reports/progress_20261105/KEY_COMPARISONS.md)和[方法与局限](../reports/progress_20261105/METHODS_AND_LIMITS.md)。[证据状态](../reports/progress_20261105/EVIDENCE_STATUS.json)区分主实验完整、网格部分和未启动任务。本页把公开源码能解释的计算与仍缺的完整执行材料分开，不能据此一键复现全部报告数字。

## 11月固定B组合的精确数组函数

[fh_radius_postprocess.py](../src/geoequi_ld/fh_radius_postprocess.py)公开固定B对称组合的`combine(points)`，直接保留已执行实现的函数签名、函数体、`EPS=1e-8`与四端点标识。它只接收同一原图坐标口径的两份Geo和两份STLD三点数组，形成Geo坐标集成、STLD坐标集成及候选输出；候选PSR/PSL保持Geo坐标集成值，FH保留Geo射线并采用两个STLD到同一Geo PSR的距离均值。退化射线或半径按原规则回退Geo集成。

[test_fh_radius_postprocess.py](../tests/test_fh_radius_postprocess.py)用人工数组核对半径计算、PS点不变、输入不变和退化回退；可用`python -m pytest tests/test_fh_radius_postprocess.py`运行。这些检查没有医学输入或模型调用。该函数不是图像到坐标的完整推理器，也不提供原生前处理、四模型权重、清洁身份验收或报告全部指标的再评价；输入有限性及来源一致性仍由调用方负责。公开组合函数不改变原生Geo与STLD各自的模型身份，也不证明逐例不退步。

## 已公开的原strict核心与历史工程

| 要查看的内容 | 实际源码入口 | 能核查的内容 |
|---|---|---|
| 原 strict 模型与普通 DSNT | [strict.py](../src/geoequi_ld/models/strict.py) | 灰度 512 输入、三通道 256 热图，独立 PS/FH 增强及解码分支；普通空间 Softmax 后取坐标期望 |
| HRNet 与两种解码器 | [hrnet.py](../src/geoequi_ld/models/hrnet.py) | timm HRNet-W32 最终融合高分辨率分支、共享/独立解码器；构造为 pretrained=False |
| PS/FH 专业增强 | [specialized.py](../src/geoequi_ld/models/specialized.py) | PS 调制变形卷积和空间注意力，FH ASPP-lite/SE；通道 LayerNorm 与残差连接 |
| 原 strict 损失原语 | [strict_losses.py](../src/geoequi_ld/training/strict_losses.py) | 有效点监督热图 MSE，以及已接纳点的对齐伪热图 MSE；不是当前 V2 的完整训练连接 |
| 坐标、变换与评价原语 | [coordinates.py](../src/geoequi_ld/geometry/coordinates.py)、[transforms.py](../src/geoequi_ld/geometry/transforms.py)、[aop.py](../src/geoequi_ld/geometry/aop.py)、[keypoints.py](../src/geoequi_ld/metrics/keypoints.py) | 归一化/像素坐标、相似变换、AoP 与点误差的实现；当前数字仍以阶段协议和聚合记录为准 |
| 原 strict 合成检查 | [test_strict_model.py](../tests/test_strict_model.py)、[test_strict_core.py](../tests/test_strict_core.py) | 分支/输出形状、普通 DSNT、MSE 与小合成 HRNet 特征检查；不等同真实医学数据评价 |
| 早期监督训练工程 | [train_baseline.py](../scripts/train_baseline.py)、[train_phase05.py](../scripts/train_phase05.py)、[train_phase06.py](../scripts/train_phase06.py) | 历史自定义 U-Net 与温度 DSNT 的基线、损失对照和长预算流程；不是本轮 Geo V2 或 STLD/PIP 的启动器 |

## 11月结果与这些源码的对应边界

原生Geo由自身热图及普通DSNT得到三点；11月半径组合还需要固定STLD输出。A先分别形成两路配对组合再平均误差，B先平均两个Geo的坐标，再用两个STLD相对同一Geo锚点的平均半径形成一个输出。这是额外的推理组合，不是改写原生Geo训练成绩，也不是将STLD误差均值当作坐标集成。公式、全部对照及回退口径见[方法与局限](../reports/progress_20261105/METHODS_AND_LIMITS.md)。

上表的strict模型、MSE原语与早期脚本可用于阅读结构和基础计算，但不等同11月所有训练目标、初始化、冻结范围、端点装载、原生Geo/STLD前处理与完整评价执行器。公开源码中的结构存在，不代表所有内部算子均完成独立有效性验证。FULL/NO_FH从头课程、同权重临时旁路和短程适配的研究问题也不同；当前保留FULL，详见关键对照表。

合成检查验证函数、张量或几何约定，不读取医学图像，也不能替代固定权重、真实数据或独立采集评价。没有权重和经授权输入时，不能从文档数字反推逐例预测或声称重新得到了所有报告结果。

## 数据可用性与来源

11月公开交付是[聚合报告及对照表](../reports/progress_20261105/KEY_COMPARISONS.md)、方法说明和状态记录。医学原图、病例标识、逐例标签/预测、模型权重、真实采样与RNG、完整私有日志不公开；数据和权重访问资格不能由本仓可见性或代码许可推定。有限的重复开发集扰动不构成独立患者或跨设备验证。

HRNet由`timm`提供，变形卷积使用`torchvision.ops.DeformConv2d`；STLD与PIPNet的研究及作者代码归属见[公开引用](../reports/progress_20261021/REFERENCES.md)。本仓[MIT LICENSE](../LICENSE)覆盖本仓原创源码，不重新许可第三方实现、医学数据、标注或预训练权重。具体归属、直接依赖及发布边界见[来源与发布范围](ATTRIBUTION_AND_RELEASE_SCOPE.md)；本次导航不复制第三方模型或改变既有许可。

10月阶段的[完整结果](../reports/progress_20261021/RESULTS.md)、[聚合数据](../reports/progress_20261021/metrics.json)、[协议](../reports/progress_20261021/PROTOCOLS.md)和[旧Testing501主表](../reports/model_comparison_20261006/RESULTS.md)继续按原口径保存，不用新开发集或混合读出覆盖历史数字。
