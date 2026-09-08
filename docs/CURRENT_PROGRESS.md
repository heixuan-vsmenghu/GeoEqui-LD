# 当前结果与实现说明

更新于 2026-09-08。本文为公开的阶段结果摘要，依据已保存的本地实验报告和实现记录整理，未重新运行实验；完整训练运行系统与原始记录尚未随本文发布。

本次在已发布摘要基础上接入[当前核心源码](../src/geoequi_ld/models/README.md)，并补充[监督结果摘录](../reports/review_20260908/SUPERVISED_RESULTS.md)和[热图与 DSNT 核对](../reports/review_20260908/HEATMAP_DSNT_CHECK.md)。当前核心可导入核读，但不代表完整训练运行系统已经公开。

## 官方 T10 与 Stage 1

官方 `UNet Heatmap / T10` 的 150 轮训练和冻结评价已完成。本地复现 validation MRE_ALL 为 31.6133 px，Testing 为 23.7044 px、AoP 10.4706°；论文对应 Testing 为 21.83 px、8.37°。**运行链路已复现，论文数值尚未完全复现。** 结果引自[已有报告](https://github.com/heixuan-vsmenghu/GeoEqui-LD/blob/e3af9e2a2c29e5de3beab912fe1ecf7592300f8b/reports/baseline_reproduction/BASELINE_REPRODUCTION.md)，本次整理未重评 Testing。

strict Stage 1 使用 300 张 labeled train 和 100 张 validation，总计完成 200 轮纯监督预热。物理 batch=2，每轮 150 次 Adam 更新；seed=42、lr=0.0001、weight_decay=0.0001、clip=5，sigma=4 高斯热图 MSE，无 scheduler、额外增强、EMA、几何项或伪标签项。

以 raw-BN validation hard-argmax MRE_ALL 选模，完整可恢复的 selected 为 epoch 39。下表每行指标均来自该行同一存档；strict 主结果为 seed=42 的单次运行，表中 px 均为输入图像像素。

| checkpoint / 读出 | PS1 px | PS2 px | FH1 px | ALL px | AoP MAE | AoP 数值可计算 | val heatmap MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| epoch 39 selected，raw-BN argmax | 18.728 | 26.443 | 48.006 | 31.059 | 12.312° | 100/100 | 0.000591 |
| epoch 200 final，raw-BN argmax | 91.449 | 104.913 | 82.204 | 92.855 | 50.086° | 97/100 | 0.000865 |

同一 epoch 39、同一 validation、同一 raw BN 的普通 Softmax DSNT：MRE_ALL=153.767 px、AoP MAE=37.293°、归一化熵=0.999992；与 argmax 相差 122.708 px。隔离 BN 校准副本的 argmax ALL=29.399 px，仅为辅助诊断，不替换 raw selected。

FH1 是 selected 同轮三个点中最大的误差来源。final 明显差于 selected，所以不能称 200 轮已经稳定收敛。T10 的评价前固定 final 与 strict 的 validation-selected 不是同选模协议，AoP 退化预测的聚合方式也需区分，不据此宣称超过 baseline。31.059 px 不属于 Testing、DSNT 或最终半监督结果。

来源：本地未提交的 `reports/strict_model_stage1/STRICT_STAGE1_200E.md`，尤其第 3–8 节。本文的 AoP“数值可计算”对应已有记录的有限值/非零射线判定，不等于几何结构可靠或满足后续 1 像素射线条件。

## Stage 2 与监督接续

按材料搭建的 Stage 2 当前实现从上述 epoch 39 开始，计划最多 5 轮，实际在 epoch 2 停止。student validation argmax ALL 在 epoch 1/2 为 267.830/304.203 px。EMA、无标签前向和伪标签流程已实际运行，但联合训练尚未取得有效增益；具体数值约定列在“实现口径”中。

既有归因记录显示，固定首 batch 的加权角度项占 Ltotal 99.87%，epoch 1/2 的平均角度贡献约 75.56%/57.18%；两种统计不能混称。短射线条件下角度梯度可数值有限却异常大，有限 AoP 不等于有效几何。该记录不意味着所有后续退化都由同一原因造成。

补充短程对照显示，移除几何损失和无标签前向后，单张监督续训仍明显退步。原生 Stage 1 参照直接复用原模型、Adam、数据排列和训练计算，未发现可直接修补的意外接入差异；25 次物理 batch=2 更新后仍出现以下短期变化：

| 更新次数 | 累计训练张次 | raw argmax ALL px | calibrated argmax ALL px |
|---:|---:|---:|---:|
| 0 | 0 | 31.059 | 29.399 |
| 10 | 20 | 34.609 | 29.637 |
| 25 | 50 | 44.503 | 41.520 |

其普通 DSNT ALL 从 153.767 变为 153.786 px，终点熵仍近 1。本组看 50 张次，原 batch=1 对照看 25 张次，顺序也不同；终点差异不是纯 batch 大小因果效应。25 次更新不代表完整收敛结果，batch、样本顺序、数据暴露与 Adam 历史的相对作用仍未确定。

**监督续训波动与同一 checkpoint 的 DSNT 读出失准需要分开判断。** 现有对照尚不能确定续训退步的唯一原因；DSNT 读出差异在继续训练前已存在。目前训练暂停，保留已有状态与记录。

来源均为本地未提交报告：`reports/strict_model_stage2/STRICT_STAGE2_ORIGINAL_5E.md`、`reports/strict_model_stage2/STAGE2_COLLAPSE_ATTRIBUTION.md`、`reports/strict_model_stage2/STAGE2_25STEP_PAIRED_CONTROLS.md`，以及 `reports/strict_model_stage1/STAGE1_NATIVE_RESUME_25STEP.md`。

## 读出版本与实际调用

当前 strict 使用 [strict.py](../src/geoequi_ld/models/strict.py)：`StandardDSNT.forward` → `standard_spatial_softmax` → `F.softmax`，再调用 `models/dsnt.py` 的 `spatial_expectation`。Softmax 前没有除以 temperature；复用期望函数不等于实例化旧 DSNT 类。对应监督归约可直接查看 [strict_losses.py](../src/geoequi_ld/training/strict_losses.py)。

Stage 1 的本地 `scripts/run_strict_stage1_200e.py` 复用 batch2 监督更新及 `training/strict_runner.py` 的只读评价路径；Stage 2 的本地 `scripts/run_strict_stage2_original_5e.py` 经 `run_strict_logical_step` 使用 `StandardDSNT`，无标签诊断也直接实例化它。这里只静态核对当前文件与已保存记录的对应；后续扩展过的文件不冒充历史运行的逐字原件。

既有 [models/dsnt.py](../src/geoequi_ld/models/dsnt.py) 的 `DSNT` 类属于早期探索，默认 temperature=0.05；旧 `train_baseline.py`、`train_phase05.py`、`train_phase06.py` 从各自配置传入该温度。底层 `spatial_softmax` 函数自身默认 1.0，与类默认值不同。旧参数和运算保持原样，仅新增版本说明；当前 strict 核心已收录，完整训练运行系统仍未公开。

配置来源见[配置目录说明](../configs/README.md)，源码目录入口见[模型版本说明](../src/geoequi_ld/models/README.md)。历史配置与旧源码用于追溯，不作为当前实现的推荐入口。

## 实现口径

以下列出材料描述、当前实现及需要核对的具体约定。

| 项目 | 原材料与实际实现的区别 |
|---|---|
| 高斯目标 | 导师材料给出 sigma=4。当前连续高斯幅度为 1；中心落在亚像素位置时，离散采样最大值可以小于 1，并非每张目标图都恰有一个值为 1。 |
| 模型输出 | 当前末端是线性卷积及插值，没有 sigmoid 将预测限制在 `[0,1]`。目标幅度与模型输出范围是不同的约定。 |
| MSE reduction | 材料 §2.5 写 MSE，公式为样本/点上的平方 L2 范数和，未显式写空间 `H×W` 平均。实现先对 H/W 平均，再平均有效样本和关键点；这是当前采用的空间归约方式，仍需核对。 |
| DSNT | 当前对原始预测热图做普通空间 Softmax，再计算坐标期望。没有 temperature；MSE 拟合高斯热图不自动保证该读出准确。 |
| Lgeo 角度与射线 | 符号结构与材料一致，但训练角度单位和近零射线有效门槛未明确。早期度制/1e-8 及后续弧度/1 热图像素均记录为实现或诊断选择，尚未经导师确认。 |
| AoP 与 c_geo | `c_geo=exp(-abs(theta1_deg-theta2_deg)/5)` 使用度。数值可计算率表示得到有限角度，射线长度有效率另计；前者并不说明几何结构可靠。 |

本地代码定位：`src/geoequi_ld/data/heatmaps.py`、`models/hrnet.py`、`models/strict.py`、`training/strict_losses.py`、`geometry/aop.py`，后四项均相对于 `src/geoequi_ld/`。上述约定以本次运行记录为准。

目前希望确认：按连续幅度 1、sigma=4 构造目标，对空间像素误差取平均，再将同一预测热图 `H_pred` 直接送入普通 Softmax DSNT，是否符合设计预期。本文不预设新的 activation、temperature 或监督损失。

## 数据数量来源

导师材料的无标签数量记为 **31,121 张**，现有运行记录采用 **31,421 张**。本文保留两项来源及其差异，未据此修改数据清单或历史结果；原因尚未解释。

- 31,121：导师原 DOCX《GeoMatch 几何一致性引导的半监督结构化关键点检测》§2.6、§4.1。
- 31,421：已有 strict 数据接入和 Stage 2 运行报告采用的 `Training/Unlabeled cases` 子池口径；该口径另排除根目录额外 2,045 张。

两项记录相差 300 张，原因尚无充分证据解释，运行清单与历史结果均未调整。Stage 1 仅使用有标签训练图，同一 validation 上的 argmax/DSNT 结果保留原有统计口径；无标签数量差异另列为待解释的数据来源事项。

本文仅使用聚合记录；实际数据位置用 `DATA_ROOT`、运行输出用 `RUN_DIR` 表示，不附患者标识、逐图文件名、私人通信或原始导师文档。
