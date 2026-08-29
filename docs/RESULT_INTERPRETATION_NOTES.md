# 结果解释与勘误

这份说明不改动原始指标、日志或结果 JSON，只补充查阅时容易混淆的边界。

## 纯 MSE 的异常

Phase 0.6 和 Phase 1A 中，B0 在当时的小型 U-Net、热图生成、DSNT 解码和优化设置下出现了响应图趋于常数、三个点重合的退化现象。这个结果说明该次具体配置失败，不能推导为“MSE 对关键点检测普遍无效”，也不能据此否定导师材料中的完整方案。

相关记录：[长预算对照](../reports/phase06/LONG_BUDGET_COMPARISON.md)、[B0 诊断](../reports/phase1a/B0_DIAGNOSTICS.md)。

## H3 的结果范围

H3 的 16 轮结果来自 seed 42，并由同一 validation 选择存档和汇报指标。selected epoch 14 的 `MRE_ALL=24.901 px`、`AoP MAE=10.289°` 支持继续研究，但不是统计显著结果、临床可用结论或稳定胜出证明。

H3 selected best 低于 H1/H2；对齐到 epoch 16 时，H3 相对 H2 的 MRE_ALL 更低，AoP MAE 反而更高。H3 也尚未在这两个汇总指标上超过旧 U-Net。完整数字见[专业增强对照](../reports/phase1c/SPECIALIZED_COMPARISON.md)。

## 几何接口不等于性能改善

Phase 2A 已确认双视图预测能够逆变换到共同坐标系，坐标项和角度项可计算，两条预测路径都能获得梯度。该检查没有更新模型权重，也没有产生新的 validation 或 testing 指标，因此只能说明接口接通，不能写成几何一致性已经提高定位精度。

当前 `tx=0.1` 使用 `[-1,1]` 网格单位。在 `align_corners=True`、图宽 512 时，它对应 `0.1×(512-1)/2=25.55` 像素，不是图宽的 10%。详见[几何接口说明](../reports/phase2a/GEOMETRY_CONTRACT.md)。

## 视觉检查的执行者

Phase 1A、1B、1C 的历史聚合文件和部分报告使用了 `manual_visual_review`、“人工检查”或类似字段。现有私有 review JSON 保存了 PASS 决定和叠加图绑定，但没有记录独立审阅者身份、签名或医学复核人；本项目执行过程是 AI 辅助的。因此查阅版统一解释为 **AI 辅助视觉检查**，只能说明叠加图中未观察到明显坐标或通道错位，不能作为独立人工复核或医学评价。

历史 JSON 的字段名为保持原记录不变而保留，不补造审阅者身份。相关文字说明已在 [Phase 1A 小结](../reports/phase1a/PHASE1A_SUMMARY.md)、[HRNet 接入记录](../reports/phase1a/HRNET_IMPLEMENTATION.md)和[解码器对照](../reports/phase1b/DECODER_COMPARISON.md)中增加查阅注记。

## Testing 的实际边界

Phase 0 的 U-Net 方案冻结后曾对 501 条 testing 记录评估一次。Phase 0.5 以后没有重新用 testing 选择模型、损失或参数。Phase 2A 的一次归档诊断曾把部分 testing 成员的文件名元数据输出到本地终端，但没有读取图像内容或标签，也没有运行 testing 推理。这三件事应分别记录，不能概括成“整个项目从未接触 testing”。

数据重复和分区限制见[数据审计](../reports/phase0/DATA_AUDIT.md)，Phase 2A 事件见[数据接入记录](../reports/phase2a/DATA_INTAKE.md)。
