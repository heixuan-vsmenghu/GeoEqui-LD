# 历史研究索引

2026-09-08 整理。以下链接指向既有公开历史记录的固定版本，保留早期探索、负结果和当时的数据接入限制，不把旧配置或旧状态改写为当前方案。原报告的数字、日期与结论原样保留；最新概况见[当前结果](../CURRENT_PROGRESS.md)。

## 早期监督与结构探索

| 记录 | 历史内容及阅读边界 |
|---|---|
| [Phase 0](https://github.com/heixuan-vsmenghu/GeoEqui-LD/blob/e3af9e2a2c29e5de3beab912fe1ecf7592300f8b/reports/phase0/BASELINE_REPORT.md) | 早期自定义小型 U-Net 与一次冻结 Testing 评价，不是后来复现的官方 T10 |
| [Phase 0.5](https://github.com/heixuan-vsmenghu/GeoEqui-LD/blob/e3af9e2a2c29e5de3beab912fe1ecf7592300f8b/reports/phase05/SUPERVISED_ABLATION.md) | 监督辅助损失消融和多 seed 记录，不是 strict 纯 MSE 当前配置 |
| [Phase 0.6](https://github.com/heixuan-vsmenghu/GeoEqui-LD/blob/e3af9e2a2c29e5de3beab912fe1ecf7592300f8b/reports/phase06/LONG_BUDGET_COMPARISON.md) | 200 轮长预算探索，保留纯 MSE 失败结果；不同于 strict Stage 1 的 200 轮运行 |
| [Phase 1A](https://github.com/heixuan-vsmenghu/GeoEqui-LD/blob/e3af9e2a2c29e5de3beab912fe1ecf7592300f8b/reports/phase1a/PHASE1A_SUMMARY.md) | B0 退化诊断与 HRNet 共享解码 H1 |
| [Phase 1B](https://github.com/heixuan-vsmenghu/GeoEqui-LD/blob/e3af9e2a2c29e5de3beab912fe1ecf7592300f8b/reports/phase1b/PHASE1B_SUMMARY.md) | BN 诊断与独立解码 H2 |
| [Phase 1C](https://github.com/heixuan-vsmenghu/GeoEqui-LD/blob/e3af9e2a2c29e5de3beab912fe1ecf7592300f8b/reports/phase1c/PHASE1C_SUMMARY.md) | PS/FH 专业增强 H3；另见[结构](https://github.com/heixuan-vsmenghu/GeoEqui-LD/blob/e3af9e2a2c29e5de3beab912fe1ecf7592300f8b/reports/phase1c/SPECIALIZED_ARCHITECTURE.md)和[逐点对照](https://github.com/heixuan-vsmenghu/GeoEqui-LD/blob/e3af9e2a2c29e5de3beab912fe1ecf7592300f8b/reports/phase1c/SPECIALIZED_COMPARISON.md) |

旧 README 的 validation 摘录如下。各行使用 `MSE + 10×坐标 SmoothL1 + JS`，H3 的 DSNT temperature=0.05 属于早期工程探索，与当前 strict 的普通 Softmax、纯 MSE 配置分开记录。

| 历史模型 | 实际轮数 | selected epoch | MRE_ALL | AoP MAE |
|---|---:|---:|---:|---:|
| 小型 U-Net | 20 | 15 | 24.779 px | 8.514° |
| H1 | 20 | 3 | 32.391 px | 12.130° |
| H2 | 16 | 3 | 31.185 px | 13.563° |
| H3 | 16 | 14 | 24.901 px | 10.289° |

同一 validation 用于选模和汇报，主要是单 seed 描述；结构、实际轮数和部分归一化设置不同，非严格因果比较。对齐第 16 轮时，H2 为 28.419 px / 13.933°，H3 为 25.936 px / 14.279°，定位与角度指标并非同时改善。解释和勘误见[原说明](https://github.com/heixuan-vsmenghu/GeoEqui-LD/blob/e3af9e2a2c29e5de3beab912fe1ecf7592300f8b/docs/RESULT_INTERPRETATION_NOTES.md)。

## 历史数据与几何接口

- [Phase 2A 小结](https://github.com/heixuan-vsmenghu/GeoEqui-LD/blob/e3af9e2a2c29e5de3beab912fe1ecf7592300f8b/reports/phase2a/PHASE2A_SUMMARY.md)与[几何接口](https://github.com/heixuan-vsmenghu/GeoEqui-LD/blob/e3af9e2a2c29e5de3beab912fe1ecf7592300f8b/reports/phase2a/GEOMETRY_CONTRACT.md)：证明当时计算与梯度路径可运行，不代表有效精度增益或当前 Stage 2 已成功。
- [数据接入记录](https://github.com/heixuan-vsmenghu/GeoEqui-LD/blob/e3af9e2a2c29e5de3beab912fe1ecf7592300f8b/reports/phase2a/DATA_INTAKE.md)、[归档问题](https://github.com/heixuan-vsmenghu/GeoEqui-LD/blob/e3af9e2a2c29e5de3beab912fe1ecf7592300f8b/reports/phase2a_closeout/ARCHIVE_ISSUE_BRIEF.md)、[当时的接收清单](https://github.com/heixuan-vsmenghu/GeoEqui-LD/blob/e3af9e2a2c29e5de3beab912fe1ecf7592300f8b/reports/phase2a_closeout/APPROVED_ARCHIVE_ACCEPTANCE_CHECKLIST.md)：保留当时候选 0 和 `BLOCKED_ACCESS + BLOCKED_INTEGRITY` 的证据，不用它们覆盖后来已有运行事实，也不据后来运行推定公开授权。
- [官方 T10 复现](https://github.com/heixuan-vsmenghu/GeoEqui-LD/blob/e3af9e2a2c29e5de3beab912fe1ecf7592300f8b/reports/baseline_reproduction/BASELINE_REPRODUCTION.md)：与上述自定义基线分开，原始冻结结果不重评。

## 本地 strict 记录

以下记录目前主要保存在本地未提交工作区，因此列出路径，不设置网页链接：

- `reports/strict_model_pilot/`：strict 接入、DSNT activation 与 MSE 连接诊断。
- `reports/strict_model_stage1/`：监督、BN、学习率、batch 对照、200 轮结果与原生 25 次更新参照。
- `reports/strict_model_stage2/`：Stage 2 当前实现的短试跑、退化归因、弧度/射线保护与 A/B/C 短对照。

上述原始记录保留。诊断中采用的参数与当前主线配置分别说明；私人通信和导师原始材料不属于本索引的查阅内容。
