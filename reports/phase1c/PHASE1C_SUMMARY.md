# Phase 1C 小结

> 查阅注记（2026-08-29）：历史聚合文件中的 `manual_coordinate_and_channel_review` 是原有字段名；现有记录没有独立审阅者身份，因此不把它解释成独立人工或医学复核。详见[结果解释与勘误](../../docs/RESULT_INTERPRETATION_NOTES.md)。

这轮把导师方案里的两个专业增强模块接到了现有 H2 独立解码器上：PS 分支是真实可变形卷积、modulation mask 与空间注意力，FH 分支是 ASPP-lite 与 SE。结构、梯度、保存加载、四样本学习门禁和正式训练链路均已通过。

四样本跑满 500 步，MRE_ALL 为 4.689 px，AoP MAE 为 1.389°，有效率 4/4。第一次门禁尝试暴露的是梯度审计范围过宽，不是模型学习失败；一次最小修复后重新从未训练初始化运行，模型、损失、步数和阈值都没改。

正式 H3 跑满 16/16 轮，按“惩罚 AoP→MRE_ALL→较早轮次”选到 epoch 14：PS1 12.426、PS2 21.446、FH1 40.831、MRE_ALL 24.901 px，AoP MAE 10.289°。selected best 相对 H2 的三个点与总体指标都更低；但 matched epoch 3 的 PS2 更高，matched epoch 16 的 FH1 与 AoP MAE 更高，所以不能写成“所有关键点稳定改善”。更准确的说法是：当前单 seed、16 轮 validation 结果支持继续研究。

训练固定 seed 42 与数据顺序，并启用了 deterministic algorithms；DeformConv2d CUDA backward 只能 warn-only，因此不声称位级复现。H3 比 H2 增加 40,420 个可训练参数，正式运行约 77.6 分钟，预算内完成。

这一阶段仍是 B2 增强监督工程参照，不是导师原文的纯 MSE，也没有 EMA 教师、伪标签、置信度机制、无标签一致性损失或半监督结论。testing 始终冻结，公开报告只有脱敏配置、聚合 train/validation 指标和曲线。

无标签审计显示当前完整可训练文件数为 0；总量口径、归档完整性、许可和分区重叠都还没闭环。详情见 [UNLABELED_INTAKE.md](UNLABELED_INTAKE.md)。结构说明见 [SPECIALIZED_ARCHITECTURE.md](SPECIALIZED_ARCHITECTURE.md)，逐点对照见 [SPECIALIZED_COMPARISON.md](SPECIALIZED_COMPARISON.md)。
