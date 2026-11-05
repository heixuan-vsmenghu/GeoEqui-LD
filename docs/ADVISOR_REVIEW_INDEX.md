# 阶段查阅导航

本周期结果截至2026年11月5日上午。[本周期报告 PDF](../reports/progress_20261105_reviewed/REPORT.pdf)、[报告正文](../reports/progress_20261105_reviewed/REPORT.md)和[主实验与消融证据对应表](../reports/progress_20261105_reviewed/EVIDENCE_MAP.md)按原生性能恢复、对应消融和FH专项诊断组织。评价使用同一固定Validation100开发集，位置误差为原图像素，角度误差为度，数值越低越好。

## 原生模型的性能恢复

两份新训练来源的原生SGP末轮ALL为16.286/18.414 px，均值17.350 px；AoP均值5.054°。同一开发集上的旧完整模型ALL为19.978 px，新结果支持性能量级恢复。两批训练各自保留来源与固定学生终点身份；11月2日S/SGP验证复用新warm9000，没有再次重训预热。

原生Geo与两个固定STLD端点的误差均值具有以下取舍。

| 固定开发集结果 | PSR px | PSL px | FH px | ALL px | AoP ° |
|---|---:|---:|---:|---:|---:|
| 原生Geo两来源，各自评分后等权 | 8.711 | 13.696 | 29.644 | 17.350 | 5.054 |
| STLD两单模型，各自评分后等权 | 11.229 | 16.823 | 25.646 | 17.900 | 5.767 |

原生Geo的PS两点、ALL和AoP均值较低，FH均值高约3.997 px。当前原生FH差距与训练稳定性仍是主问题。逐来源、逐指标和尾部共同界定改善范围。完整清洁数表及各固定STLD单端点见[关键对照表](../reports/progress_20261105/KEY_COMPARISONS.md)。

## 与主实验对应的消融

[证据对应表](../reports/progress_20261105_reviewed/EVIDENCE_MAP.md)以来源、初始化与预热、半监督课程、预测角色、终点和评价集合标明每项控制的关系。

| 研究问题 | 已完成的对应证据 | 当前结论与剩余范围 |
|---|---|---|
| 新来源中的组合辅助增量 | 两来源SGP与同源S共享规定预热，保留相同U前向和教师流程，S关闭G/P辅助反传 | SGP−S的ALL分别为−6.918/−9.795 px，两来源五项均值均较低。G与P各自的作用仍需该主协议下的对应控制 |
| FH整块的训练期贡献 | 单来源FULL与NO_FH分别从对应医学初态完成9000步监督和3000步半监督，使用同批FULL参照 | NO_FH−FULL的FH为−0.126 px、ALL为+0.087 px、AoP为−0.032°，FH p95为+3.082 px。证据覆盖该来源整块对照，内部组件及来源重复仍有缺项 |
| 已训练系统对FH块的依赖 | 同一FULL权重推理时临时旁路FH增强 | FH由31.364升至33.232 px。这项依赖诊断与完整课程结构对照分别记录 |
| 历史结构、损失和训练条件 | 10月阶段的逐seed原表和当时协议继续保存 | 各表解释相应训练家族。当前主实验的结构和损失结论以对应协议的控制为依据 |

当前保留完整模型。主协议确定后，相关结构与损失控制需与该协议对应；主实验重跑若出现明显变化，逐点、角度和尾部差异也进入复核范围。已完成的单来源FH探索与组合辅助增量，尚未覆盖当前模型全部结构和损失因素。

## FH专项诊断与推理组合

FH误差诊断发现了方向和半径的不同作用。随后固定的A方案逐来源借用配对STLD的FH半径，B方案使用两份Geo坐标与两份STLD预测组成一个输出。清洁条件下，A的FH/ALL为25.686/16.031 px，B为25.287/15.758 px。原生Geo训练结果按上一节的自身身份记录。

五个主条件均已完整执行。整数平移扩展中，A完整覆盖±11方格，B完整覆盖±8方格；预定±16以部分交付结束。B对STLD1901的341个实交条件五项均值均较低。在完整±8的289个位置中，B对两STLD单模型误差均值、STLD坐标集成的五项均值全优条件分别为254/289、53/289；逐图五指标全部不退步为0/289。B清洁FH相对STLD坐标集成仍高0.068 px，相对STLD1801的FH均值也有差距。

[专项一页报告](../reports/progress_20261105/REPORT.pdf)、[全部关键比较](../reports/progress_20261105/KEY_COMPARISONS.md)、[方法与局限](../reports/progress_20261105/METHODS_AND_LIMITS.md)及[机器可读证据状态](../reports/progress_20261105/EVIDENCE_STATUS.json)保存该次实交口径。相同100图的重复扰动、有限数学分解和合成核对提供开发证据，独立病例与采集条件的验证仍待完成。

## 历史报告与源码

[10月阶段报告](../reports/progress_20261021/REPORT.pdf)保留结果截至10月20日的结构、训练信号、监督时长与FH目标宽度对照。其[问题与结果对照](../reports/progress_20261021/REQUEST_ALIGNMENT.md)、[逐seed及阶段结果](../reports/progress_20261021/RESULTS.md)、[实验协议](../reports/progress_20261021/PROTOCOLS.md)、[聚合指标](../reports/progress_20261021/metrics.json)、[CDF计数](../reports/progress_20261021/cdf_counts.json)和[限制与下一步](../reports/progress_20261021/LIMITATIONS.md)继续按原口径保存。

[10月6日Testing501主表](../reports/model_comparison_20261006/RESULTS.md)和[原报告](../reports/model_comparison_20261006/REPORT.pdf)属于历史独立记录。旧Testing501、当前Validation100及不同训练来源具有各自的证据范围，更早记录见[历史导航](HISTORY.md)。

[源码导航](CODE_MAP.md)列出公开strict核心、历史脚本与B方案的精确数组函数，并说明端到端复现所需材料。数据及模型访问、第三方组件条款和本仓原创代码许可分别见[来源与发布范围](ATTRIBUTION_AND_RELEASE_SCOPE.md)、[公开引用](../reports/progress_20261105/REFERENCES.md)与[MIT LICENSE](../LICENSE)。
