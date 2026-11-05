# 复核版证据来源与阅读顺序

先读 [阶段汇报](REPORT.md) 或 [PDF](REPORT.pdf)，再按问题查看 [证据与主实验对应表](EVIDENCE_MAP.md) 或 [PDF](EVIDENCE_MAP.pdf)。

1. 性能下降排查和旧训练身份见 [10月阶段报告](../progress_20261021/REPORT.md)。后续状态、计划、Adam、初始化、步幅和固定末轮的对应证据已在本复核版列明，公开材料只保留聚合。
2. 43/44 低步幅四格及同期步幅配对来源于已完成的 20261030/31 固定任务。四格保留全部八条末轮；F1101A/B 是后续不同预热来源。
3. F1101A/B 同源 S/SGP 见 [11月5日既有主线资料](../progress_20261105/REPORT.md)；本复核版补齐来源、匹配范围和全部单 STLD 参照。
4. FH 结构来自 20261103 完整课程单来源 FULL/NO_FH，推理旁路来自 DUAL16。其自己的 FULL 末轮与当前两来源原生主行分别列明。
5. [11月5日早版专项报告](../progress_20261105/REPORT.md)、[固定比较](../progress_20261105/KEY_COMPARISONS.md)、[方法边界](../progress_20261105/METHODS_AND_LIMITS.md) 和 [证据状态](../progress_20261105/EVIDENCE_STATUS.json) 保留半径组合与过夜覆盖。组合依赖 STLD，不替代原生训练优势。

本版表格的 [机器可读聚合与任务来源](EVIDENCE_DATA.json) 保留实际训练实例、匹配范围和来源标识，可与正文逐项核对。

作者模型与外部基线的来源、许可和代码公开范围沿 [来源及发布说明](../../docs/ATTRIBUTION_AND_RELEASE_SCOPE.md) 与 [源码导航](../../docs/CODE_MAP.md)。历史 [Testing501 比较](../model_comparison_20261006/RESULTS.md) 保留原集合与模型身份。
