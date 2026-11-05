# GeoEqui-LD

产时超声三关键点（PSR、PSL、FH）定位与产程进展角（AoP）研究。

## 本周期研究进展

[本周期报告 PDF](reports/progress_20261105_reviewed/REPORT.pdf) · [报告正文](reports/progress_20261105_reviewed/REPORT.md) · [主实验与消融证据对应表](reports/progress_20261105_reviewed/EVIDENCE_MAP.md) · [阶段查阅导航](docs/ADVISOR_REVIEW_INDEX.md)

本周期报告介绍初始化与训练修正、整体性能、几何一致性与伪监督、新来源复验及FH结构对照。原生Geo两来源的ALL均值恢复至 **17.350 px**，同源SGP与S的组合辅助增量已有配对结果；FH均值 **29.644 px** 相对固定STLD两单模型误差均值 **25.646 px** 仍有差距。一个来源的FULL/NO_FH完成各自完整课程，结果呈指标取舍。原生FH误差及来源间变化仍有待进一步分析。后续拟结合这些结果明确主实验条件，并补充相应的结构和损失对照。

## 2026-11-05 专项结果快照

<details>
<summary>展开 FH 推理组合与过夜覆盖的早版记录</summary>

[一页报告 PDF](reports/progress_20261105/REPORT.pdf) · [报告正文](reports/progress_20261105/REPORT.md) · [比较表与完整口径](reports/progress_20261105/KEY_COMPARISONS.md) · [方法与限制](reports/progress_20261105/METHODS_AND_LIMITS.md) · [证据状态](reports/progress_20261105/EVIDENCE_STATUS.json)

原生 Geo 两训练来源在 Validation100 开发图像上的 ALL 误差均值为 **17.350 px**；匹配的 SGP−S 对照分别降低 **6.918 / 9.795 px**。S 保留无标签前向与教师流程，仅关闭辅助反传。一个来源的 FULL/NO_FH 训练对照出现指标取舍，尚不支持删除 FH 增强模块。

使用两份 Geo 与两份 STLD 输出的 B 对称半径组合达到 **ALL 15.758 px、FH 25.287 px、AoP 4.938°**。这是依赖 STLD 的组合输出，原生 Geo 的结果仍单列。其清洁 FH 相对 STLD 坐标集成高 **0.068 px**，相对两 STLD 单模型误差均值低 **0.359 px**；两种对照不可互换。

A/B 主实验各完成五条件；扩展按预算以 **partial** 关闭。A 完整覆盖离散 ±11 方格，B 完整覆盖离散 ±8 方格并有额外条件，均未完成预定 ±16。B 对 STLD1901 的 341 个实交条件五项均值均较低；对较强坐标集成、STLD1801 以及逐图检验仍有不利项。完整 ±8 的逐图五指标全部不退步条件为 **0/289**。这些是同一批100开发图像上的重复扰动，独立患者数未知，不构成临床或外部验证。

</details>

## 如何阅读与复核

- [阶段查阅导航](docs/ADVISOR_REVIEW_INDEX.md)：按研究问题查看证据与限制。
- [数值表](reports/progress_20261105/KEY_COMPARISONS.md)与[CSV](reports/progress_20261105/KEY_COMPARISONS.csv)：区分模型身份、误差平均、坐标集成、边际p95与配对差值p95。
- [源码导航](docs/CODE_MAP.md)：公开 strict 核心、历史实现与当前可复核部分。
- [B组合函数](src/geoequi_ld/fh_radius_postprocess.py)及[合成测试](tests/test_fh_radius_postprocess.py)：精确提取封存组合逻辑；不包含四端点推理或完整生产训练入口。
- [来源与发布范围](docs/ATTRIBUTION_AND_RELEASE_SCOPE.md)、[方法引用](reports/progress_20261105/REFERENCES.md)与[MIT LICENSE](LICENSE)。

本仓包含聚合结果、可读报告及部分合法源码。完整 V2、当前结构消融与外部适配的生产训练入口未全部公开，**不能一键复现当前全部实验数字**。本次整理没有新增训练、模型推理、Testing 或 bootstrap。

## 历史结果保持原样

[2026-10 阶段报告](reports/progress_20261021/REPORT.pdf)与[完整五指标](reports/progress_20261021/RESULTS.md)保留当时口径。历史 [2026-10-06 Testing501 主表](reports/model_comparison_20261006/RESULTS.md)及[原报告](reports/model_comparison_20261006/REPORT.pdf)保持原样；旧 Testing501 的15.390 px与Validation100的19.978 px及本轮组合开发结果分属不同对象，不直接相减作提升结论。两个训练来源不是两份独立临床测试队列。

医学图像、逐例标签与预测、病例标识、权重、真实采样计划、私有执行记录和通信材料不在本次公开范围。更完整历史见[HISTORY](docs/HISTORY.md)。
