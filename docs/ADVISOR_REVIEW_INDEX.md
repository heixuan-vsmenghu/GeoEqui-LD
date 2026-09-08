# 导师查阅索引

更新于 2026-09-08。本页与当前结果摘要已公开；近期 strict 源码和完整运行报告尚未发布，下表将这些记录标为“本地”。已公开的 baseline 与历史报告使用固定版本链接。

## 建议阅读顺序

1. [当前结果与实现说明](CURRENT_PROGRESS.md)：同一 checkpoint 的完整指标、监督续训与 DSNT 读出问题的边界、数据数量来源说明。
2. [官方 T10 复现报告](https://github.com/heixuan-vsmenghu/GeoEqui-LD/blob/e3af9e2a2c29e5de3beab912fe1ecf7592300f8b/reports/baseline_reproduction/BASELINE_REPRODUCTION.md)：来源、训练设置及与论文结果的差距。
3. [来源与发布范围](ATTRIBUTION_AND_RELEASE_SCOPE.md)：导师方案、AI 辅助实现、第三方来源与查阅范围。
4. [历史研究索引](history/RESEARCH_INDEX.md)：早期探索与失败结果，需要追溯时查阅，不作为当前配置。

另备两份私下查阅附件：`01_产时超声课题_阶段汇报.pdf`（2 页）和 `02_热图监督与DSNT_实现核对.pdf`（1 页），各有同名 DOCX 源稿。附件未公开上传；本页在线内容不依赖取得附件后才能阅读。

## 当前证据入口

| 问题 | 记录位置 | 阅读边界 |
|---|---|---|
| 官方 T10 是否复现 | [官方 baseline 报告](https://github.com/heixuan-vsmenghu/GeoEqui-LD/blob/e3af9e2a2c29e5de3beab912fe1ecf7592300f8b/reports/baseline_reproduction/BASELINE_REPRODUCTION.md) | 训练和冻结评价完成，数值未完全达到论文；不同于早期自定义 U-Net |
| 200 轮监督预热与最佳状态 | 本地 `reports/strict_model_stage1/STRICT_STAGE1_200E.md` | selected epoch 39 与 final epoch 200 分开；31.059 px 是 validation argmax |
| Stage 2 当前实现的短试跑 | 本地 `reports/strict_model_stage2/STRICT_STAGE2_ORIGINAL_5E.md` | 文件名中的 5 轮是计划上限；实际在 epoch 2 停止，尚无有效增益结论 |
| 退化归因与短对照 | 本地 `reports/strict_model_stage2/STAGE2_COLLAPSE_ATTRIBUTION.md`、`reports/strict_model_stage2/STAGE2_25STEP_PAIRED_CONTROLS.md` | 固定起点短程证据，不能把全部退化归因于 DSNT 或 batch size=1 |
| 原生 Stage 1 续训参照 | 本地 `reports/strict_model_stage1/STAGE1_NATIVE_RESUME_25STEP.md` | 物理 batch=2，25 次更新、50 张次；并非与 batch=1 的纯因果比较 |

上表标为“本地”的完整原记录未随本索引提供，主要结果摘录在[当前结果](CURRENT_PROGRESS.md)中。查阅稿不附机器绝对路径或逐样本材料。

## 结论边界

Stage 1 轮数完成不等于稳定收敛；当前没有半监督有效增益或临床可用性结论。原生监督接续也会短期退步，因此该现象不能直接归罪于 DSNT 几何梯度；同一权重的 DSNT 读出失准仍独立存在。目前希望核对 `H_pred` 从热图监督进入普通 Softmax DSNT 时的数值约定；本文记录已有实现，不预设新的损失或训练方案。

查阅与公开范围见[来源说明](ATTRIBUTION_AND_RELEASE_SCOPE.md)。本次仅发布已核读的聚合说明，不含完整 strict 源码、原始运行材料或私人通信。
