# 导师查阅索引

更新于 2026-09-08。当前主线已完成官方 T10 复现运行、指定 strict 模型的 200 轮监督预热，并进行了 Stage 2 短试跑。下面提供对应结果和当前核心源码；半监督短试跑尚未取得有效增益，目前聚焦核对热图监督与 DSNT 的数值约定。

## 建议阅读顺序

1. [监督训练结果摘录](../reports/review_20260908/SUPERVISED_RESULTS.md)：官方 T10、strict epoch 39 与末轮 epoch 200，各自评价身份及限制。
2. [当前模型与源码入口](../src/geoequi_ld/models/README.md)：实际 strict 模型、普通 Softmax DSNT 和 MSE 核心，不是旧 temperature=0.05 配置。
3. [热图监督与 DSNT 核对](../reports/review_20260908/HEATMAP_DSNT_CHECK.md)：同一热图的读出差异、合成高斯例子和需要确认的数值约定。
4. [当前结果与实现说明](CURRENT_PROGRESS.md)：监督续训与 DSNT 读出问题的边界、当前调用关系、数据数量来源说明。
5. [来源与发布范围](ATTRIBUTION_AND_RELEASE_SCOPE.md)：导师方案、AI 辅助实现、第三方来源与查阅范围。

结果摘录可直接在线阅读，模型和损失核心可直接查阅。摘录不替代原始日志，核心源码也不包含完整训练运行系统；已公开的 baseline 与历史报告使用固定版本链接。

另备两份私下查阅附件：`01_产时超声课题_阶段汇报.pdf`（2 页）和 `02_热图监督与DSNT_实现核对.pdf`（1 页），各有同名 DOCX 源稿。附件未公开上传；本页在线内容不依赖取得附件后才能阅读。

## 当前证据入口

| 问题 | 记录位置 | 阅读边界 |
|---|---|---|
| 当前监督结果的聚合证据 | [监督结果摘录](../reports/review_20260908/SUPERVISED_RESULTS.md) | 同轮完整指标，保留 e200 退步，不宣称完整曲线 |
| 当前指定模型的实现 | [strict 模型](../src/geoequi_ld/models/strict.py)、[热图 MSE](../src/geoequi_ld/training/strict_losses.py) | 当前真实核心及依赖；不是完整训练复现包 |
| 当前热图与 DSNT 的连接 | [读出核对摘录](../reports/review_20260908/HEATMAP_DSNT_CHECK.md) | 真实 validation 与人工高斯的像素单位分开 |
| 官方 T10 是否复现 | [官方 baseline 报告](https://github.com/heixuan-vsmenghu/GeoEqui-LD/blob/e3af9e2a2c29e5de3beab912fe1ecf7592300f8b/reports/baseline_reproduction/BASELINE_REPRODUCTION.md) | 训练和冻结评价完成，数值未完全达到论文；不同于早期自定义 U-Net |

Stage 1 的 selected epoch 39 与 final epoch 200 分开报告；31.059 px 是 validation argmax。Stage 2 的 5 轮是计划上限，实际在 epoch 2 停止。完整本地报告路径及短程对照边界集中在[当前结果](CURRENT_PROGRESS.md)，不把内部诊断列为新的主线成果。查阅稿不附机器绝对路径或逐样本材料。

## 结论边界

Stage 1 轮数完成不等于稳定收敛；当前没有半监督有效增益或临床可用性结论。原生监督接续也会短期退步，因此该现象不能直接归罪于 DSNT 几何梯度；同一权重的 DSNT 读出失准仍独立存在。目前希望核对 `H_pred` 从热图监督进入普通 Softmax DSNT 时的数值约定；本文记录已有实现，不预设新的损失或训练方案。

查阅与公开范围见[来源说明](ATTRIBUTION_AND_RELEASE_SCOPE.md)。本次核心源码及证据页不包含完整训练启动器、原始运行材料或私人通信。

## 历史追溯与目录版本

旧实验集中在[历史研究索引](history/RESEARCH_INDEX.md)，不作为当前配置。进入源文件前可先看[配置目录说明](../configs/README.md)及[模型目录说明](../src/geoequi_ld/models/README.md)，避免把旧 temperature=0.05 的实现误认为当前 strict 读出。
