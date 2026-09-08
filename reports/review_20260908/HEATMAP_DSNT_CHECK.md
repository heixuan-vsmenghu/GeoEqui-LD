# 热图监督与普通 Softmax DSNT 核对摘录

本文依据既有实验记录和当前实现整理，不是新实验或原始日志。当前待核对的是目标热图、监督归约与 DSNT 输入的数值约定。监督续训退化与同一 checkpoint 的 DSNT 读出失准分开陈述，不预设导师方案错误，也不宣称某种后续改法已经有效。

## 当前证据：同一状态，两种读出

以下来自 strict Stage 1 的同一 epoch 39 raw-best、同一 raw-BN 状态、同一 100 张 validation。距离均为 **512×512 输入像素**，不是热图像素：

| 读出 | MRE_ALL px | AoP MAE | 归一化熵 |
|---|---:|---:|---:|
| hard argmax | 31.059 | 12.312° | 不适用 |
| 普通空间 Softmax DSNT | 153.767 | 37.293° | 0.999992 |

DSNT 与 argmax 的 MRE_ALL 差为 122.708 px。这里未使用 temperature，也未用 calibrated-BN 结果替换 raw-BN。完整同轮关键点指标及 epoch 200 的退步见[监督训练摘录](SUPERVISED_RESULTS.md)。

来源：仅本地、未提交的 `reports/strict_model_stage1/STRICT_STAGE1_200E.md` 第 3、7–8 节。该原报告没有可引用的 Git 提交；本页是聚合证据摘录，不冒充原始报告已上线。

## 当前计算链与历史实现的区别

设有效关键点集合为 V，热图大小 H=W=256，热图像素位置为 (x,y)，目标中心为 (x*,y*)。当前目标和监督项为：

```text
G(x,y) = exp(-((x-x*)^2 + (y-y*)^2) / (2 * 4^2))
Lsup = (1 / |V|) * sum_(b,k in V) [(1 / (H*W)) * sum_(x,y) (H_pred - G)^2]
```

高斯的连续幅度为 1，未作概率归一化；中心落在亚像素位置时，离散网格最大值可以小于 1。模型末端为线性卷积和插值，**没有 sigmoid 把 H_pred 强制限制在 `[0,1]`**。目标幅度与模型输出范围不是同一件事。

普通空间 Softmax 与期望读出为：

```text
P(x,y) = exp(H_pred(x,y)) / sum_(u,v) exp(H_pred(u,v))
x_norm(x) = 2*x/(W-1) - 1; y_norm(y) = 2*y/(H-1) - 1
(x_hat_norm, y_hat_norm) = sum_(x,y) P(x,y) * (x_norm(x), y_norm(y))
(x_hat_input, y_hat_input) = ((x_hat_norm+1)/2 * 511, (y_hat_norm+1)/2 * 511)
```

实际 Softmax 使用 PyTorch 实现，公式描述其数值含义；期望网格采用 `align_corners=True`。当前 `models/strict.py::StandardDSNT` 调用 `standard_spatial_softmax`，只复用旧 `models/dsnt.py::spatial_expectation`，**不调用旧 `DSNT` 类的 temperature=0.05 分支**。旧 Phase 0 `DSNT` 类及其参数属于历史实现，不代表上述 strict 读出。

代码可直接查阅：[高斯目标](../../src/geoequi_ld/data/heatmaps.py)的 `generate_gaussian_heatmaps`、[严格 MSE](../../src/geoequi_ld/training/strict_losses.py)的 `strict_supervised_heatmap_mse`、[解码器](../../src/geoequi_ld/models/hrnet.py)、[StandardDSNT](../../src/geoequi_ld/models/strict.py)、[坐标期望](../../src/geoequi_ld/models/dsnt.py)及[坐标转换](../../src/geoequi_ld/geometry/coordinates.py)。这是当前核心调用链，不包含完整训练运行系统，也不保证后续修改过的文件与历史运行逐字节相同。

## 独立合成检查：单位不能混用

既有人工高斯检查使用 **256×256 热图、sigma=4、中心 `[64,192]`、峰值 1**。该中心是人工构造点，不是真实样本坐标。

| 同一人工高斯的读出 | 坐标误差，256×256 热图像素 |
|---|---:|
| hard argmax | 0.000000 |
| 普通空间 Softmax DSNT | 90.329816 |

该合成高斯的普通 Softmax 归一化熵为 0.999944916。它说明，即使峰值位置完全正确，这一高斯数值直接经过 65,536 个空间位置上的普通 Softmax，也可以得到接近均匀的分布和错误的期望位置。这里是合成热图上的具体证据，不是患者数据结果，也不能将 90.329816 个热图像素与前表 153.767 个输入像素直接作同单位比较。

来源：仅本地、未提交的 `reports/strict_model_pilot/STAGE1_COMPATIBILITY.md` 的“合成兼容性”节。该早期报告包含其他历史实验与当时的阶段判断；本页只摘录上述既有合成检查，不把早期判断替代当前已完成的 200 轮监督结果，也不重跑任何温度诊断。

## Stage 2 退化与后续对照的边界

原 Stage 2 试跑从同一 epoch 39 完整训练状态初始化 student，EMA teacher 从 student 复制。student 标准 raw-BN validation argmax MRE_ALL 为：

| 时点 | MRE_ALL，输入像素 |
|---|---:|
| step 0 | 31.059 |
| epoch 1 | 267.830 |
| epoch 2 | 304.203 |

原计划最多 5 轮，实际在 epoch 2 停止；student/teacher 并未获得有效的联合训练增益。原归因审计中，固定首 batch 的加权角度项占 Ltotal 99.869526%，而 epoch 1/2 的平均角度贡献为 75.5624%/57.1844%，不能用一个 batch 的比例代表整轮。BN-only 回放未单独复现灾难性退化，原两轮加权伪标签贡献极小；这些是该次审计范围内的证据，不是对所有后续退化的唯一解释。

随后 25 步对照中，A 保留无标签前向但只用 Lsup，B 仅用单张有标签监督，C 为既有监督加坐标等变配置。三者 step 25 raw MRE_ALL 分别为 82.446、88.359、81.522 px；A/B 的 calibrated 值分别为 64.076、63.329 px，C 未记录 calibrated 值。B 在没有几何项和无标签前向时仍退步，因此不能将所有监督退化唯一归于 DSNT 几何梯度，亦尚未证明 batch=1 是唯一原因。

原生 Stage 1 参照复用原训练计算，25 次物理 batch=2 更新后，raw ALL 从 31.059 变为 44.503 px，calibrated 从 29.399 变为 41.520 px；未发现可直接修补的意外状态或张量接入差异。该组看 50 张次，B 组看 25 张次，样本顺序也不同，终点差异不是纯 batch 大小因果效应。与此同时，其 DSNT ALL 从 153.767 变为 153.786 px，读出差异仍独立存在。

上述来源均为仅本地、未提交原报告：`reports/strict_model_stage2/STRICT_STAGE2_ORIGINAL_5E.md` 第 1–7 节；`reports/strict_model_stage2/STAGE2_COLLAPSE_ATTRIBUTION.md` 第 3、6、8–9 节；`reports/strict_model_stage2/STAGE2_25STEP_PAIRED_CONTROLS.md` 的“同轮validation”“对照判读”节；`reports/strict_model_stage1/STAGE1_NATIVE_RESUME_25STEP.md` 的“结论边界修订”“节点结果”“回答与限制”节。此处不提供不存在的历史提交链接。

## 材料描述与实现选择

| 项目 | 区分 |
|---|---|
| 目标热图 | 材料给出 sigma=4；本地实现采用上式连续幅度 1 的高斯，不把每张离散目标的峰值都称为恰好 1。 |
| MSE 归约 | 材料 §2.5 写 MSE，公式为样本/关键点上的平方 L2 范数和，未显式列空间 H×W 平均；本地实现先空间 mean，再平均有效样本和关键点。不能将这一归约选择说成材料已唯一规定。 |
| DSNT | 材料的普通空间 Softmax 和期望链路与当前 strict 形式对应；H_pred 的数值约定仍需核对。历史 temperature=0.05 不是 strict 的隐含参数。 |
| 原 Stage 2 角度 | 计算角度时先用 `atan2(abs(cross),dot)` 得到弧度，再转成度参与角度绝对差；1e-8 射线门槛是当时实现选择，材料未明确训练角度单位和近零射线阈值。 |
| 后续保护版 | 弧度角度损失和至少 1 个热图像素射线条件属于后续诊断选择，不是原 epoch 1/2 的运行配置，也不是已获导师确认的默认口径。 |
| c_geo 与有效性 | c_geo 使用度制差值 `exp(-abs(theta1_deg-theta2_deg)/5)`；角度数值可计算与射线长度达到几何条件分别统计，有限数字不等于有效几何。 |

材料/实现区分沿用[当前进展说明](../../docs/CURRENT_PROGRESS.md)的核对记录，未在本页复制导师原材料。待确认的问题限于：**连续幅度 1、sigma=4 的目标热图，采用当前空间 mean 监督后，将同一 H_pred 直接送入普通 Softmax DSNT，是否符合设计预期的目标、归约与读出输入约定？** 当前证据不支持把新 activation、temperature 或损失写成已验证解决方案。

[返回监督结果](SUPERVISED_RESULTS.md) · [导师查阅索引](../../docs/ADVISOR_REVIEW_INDEX.md)
