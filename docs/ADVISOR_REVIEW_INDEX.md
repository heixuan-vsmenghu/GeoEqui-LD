# 导师查阅索引

建议按下面顺序查看。仓库目前是 private，GitHub 链接只有获得仓库权限的账号才能打开；没有仓库权限时，先看单独提供的阶段进展 PDF 和数据包核查 PDF。

## 1. 当前状态与结果

- [README：任务、已有实现、结果表和未完成部分](../README.md)
- [结果解释与勘误](RESULT_INTERPRETATION_NOTES.md)

## 2. 最新架构对照及限制

- [PS/FH 专业增强结构](../reports/phase1c/SPECIALIZED_ARCHITECTURE.md)
- [H1/H2/H3 逐点与对齐轮次对照](../reports/phase1c/SPECIALIZED_COMPARISON.md)
- [Phase 1C 小结](../reports/phase1c/PHASE1C_SUMMARY.md)

## 3. 几何一致性接口

- [坐标、角度、可见性和梯度定义](../reports/phase2a/GEOMETRY_CONTRACT.md)
- [Phase 2A 小结](../reports/phase2a/PHASE2A_SUMMARY.md)

这里的结果只证明计算和梯度路径可以运行，没有新增模型精度结论。

## 4. 无标签数据包核查

- [一页归档问题说明](../reports/phase2a_closeout/ARCHIVE_ISSUE_BRIEF.md)
- [获得完整归档后的接收检查清单](../reports/phase2a_closeout/APPROVED_ARCHIVE_ACCEPTANCE_CHECKLIST.md)
- [完整数据接入记录](../reports/phase2a/DATA_INTAKE.md)

当前正式验收候选为 0，状态为 `BLOCKED_ACCESS + BLOCKED_INTEGRITY`。

## 5. 历史实验与负结果

- [Phase 0：早期 U-Net 监督结果和一次冻结 testing 评估](../reports/phase0/BASELINE_REPORT.md)
- [Phase 0.5：监督辅助项消融](../reports/phase05/SUPERVISED_ABLATION.md)
- [Phase 0.6：200 轮长预算对照，包含纯 MSE 失败结果](../reports/phase06/LONG_BUDGET_COMPARISON.md)
- [Phase 1A：B0 退化诊断与 HRNet 共享头](../reports/phase1a/PHASE1A_SUMMARY.md)
- [Phase 1B：BN 诊断与独立解码器对照](../reports/phase1b/PHASE1B_SUMMARY.md)
- [Phase 1C：专业增强监督对照](../reports/phase1c/PHASE1C_SUMMARY.md)
- [Phase 2A：数据与几何接口](../reports/phase2a/PHASE2A_SUMMARY.md)

各阶段的 `aggregate_results.json` 保留机器可读聚合值。训练权重、原始医学图像、逐样本预测和私人通信不在本查阅入口中。
