# 阶段查阅导航

最新结果截至2026年11月5日上午。先读[阶段报告 PDF](../reports/progress_20261105/REPORT.pdf)或[同正文 Markdown](../reports/progress_20261105/REPORT.md)，再按下列入口核对比较对象和证据范围。

| 要核查的问题 | 公开入口 |
|---|---|
| 原生训练恢复、FH结构对照及固定半径组合的结论 | [11月5日阶段报告](../reports/progress_20261105/REPORT.md) |
| 清洁五指标、全部固定STLD对照、主条件与网格的正负结果 | [关键对照表](../reports/progress_20261105/KEY_COMPARISONS.md) |
| 原生Geo与A/B组合的区别、评分单位、p95含义及独立验证限制 | [方法与局限](../reports/progress_20261105/METHODS_AND_LIMITS.md) |
| 主实验完整、网格部分、未启动接续及可公开任务标识 | [证据状态](../reports/progress_20261105/EVIDENCE_STATUS.json) |
| 已公开实现、必要依赖和不能直接复现的范围 | [源码阅读导航](CODE_MAP.md) |

原生Geo与使用STLD输出的半径组合分别报告，组合分数不改写原生模型成绩。固定单模型、单模型误差均值与坐标集成是不同对照；五项均值更低也不等于每图每项不退步。完整的五主条件与部分交付的整数网格分列，不能从已完成范围外推完整±16或临床稳定性。具体数值与反向结果同见关键对照表。

## 10月阶段记录

[10月22日整理的两页报告](../reports/progress_20261021/REPORT.pdf)保留结果截至10月20日的结构、训练信号、监督训练时长及FH目标宽度对照。其[问题与结果对照](../reports/progress_20261021/REQUEST_ALIGNMENT.md)、[完整逐seed及阶段结果](../reports/progress_20261021/RESULTS.md)、[实验协议](../reports/progress_20261021/PROTOCOLS.md)、[聚合指标](../reports/progress_20261021/metrics.json)、[CDF计数](../reports/progress_20261021/cdf_counts.json)与[限制和下一步](../reports/progress_20261021/LIMITATIONS.md)继续独立保存，原数字不重写。

[10月6日Testing501主表](../reports/model_comparison_20261006/RESULTS.md)与[原报告](../reports/model_comparison_20261006/REPORT.pdf)另列。历史Testing、当前反复使用的Validation100开发集和不同训练来源不能直接拼表作独立确认。更早记录见[历史导航](HISTORY.md)。

## 源码与材料范围

公开仓提供聚合报告、原strict核心、历史脚本及已列出的合成检查；完整生产训练、端点评价和封存流程未全部公开，不是全部数字的一键复现包。医学图像、病例标识、逐例标签或坐标、权重、真实采样计划、私人学习材料及通信记录不在公开范围。

研究与实现来源见[公开方法引用](../reports/progress_20261021/REFERENCES.md)和[来源与发布范围](ATTRIBUTION_AND_RELEASE_SCOPE.md)。[仓库LICENSE](../LICENSE)与第三方代码、数据及权重的各自许可边界分别适用。
