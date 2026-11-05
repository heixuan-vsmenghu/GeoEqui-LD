# 当前研究进展

结果截至2026年11月5日。最新[阶段报告](../reports/progress_20261105_reviewed/REPORT.md)及[PDF](../reports/progress_20261105_reviewed/REPORT.pdf)、[实验与消融对应表](../reports/progress_20261105_reviewed/EVIDENCE_MAP.md)集中呈现初始化、学习率、辅助损失及FH结构对照；详细入口见[阶段查阅导航](ADVISOR_REVIEW_INDEX.md)。

初始化配对已完成。两份新预热来源的完整模型ALL为16.286/18.414 px，误差均值17.350 px；同来源SGP相对S的五项误差均值均较低。原生FH均值29.644 px，相对固定STLD误差均值25.646 px仍有差距。单来源FULL/NO_FH出现均值与尾部取舍。后续拟进一步分析FH方向与径向误差，并结合结果讨论对应训练及结构验证的范围。

下文保留10月22日整理时的阶段状态，其中“尚未执行”等表述仅描述当时进展。

# 2026年10月阶段记录

2026-10-22整理，已完成结果截至2026-10-20。

本阶段完成Geo结构、训练信号、监督训练时长及FH目标宽度的首轮对照，STLD/PIP和R18初始化配对作为补充。主要未决问题包括训练与验证误差差距、结构表现随训练阶段的变化及单个模块的独立贡献。

见[两页报告](../reports/progress_20261021/REPORT.pdf)、[研究问题落实表](../reports/progress_20261021/REQUEST_ALIGNMENT.md)、[逐seed完整结果](../reports/progress_20261021/RESULTS.md)。Geo纯监督训练中的训练与验证误差差距尚未得到解释。编码器初始化是待检验因素之一；拟在核对同架构预训练权重、输入通道映射及BN适配后，比较随机初始化与预训练初始化。该对照保持结构、监督损失、普通DSNT读出、数据和训练预算一致，不同时调整G/P、σ或分支，目前尚未执行。

源码入口及对应范围见[源码阅读导航](CODE_MAP.md)。

旧[Testing501主表](../reports/model_comparison_20261006/RESULTS.md)不覆盖、不混算；完整生产训练系统未全部公开。
