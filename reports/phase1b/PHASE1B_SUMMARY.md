# Phase 1B 小结

这轮做了两件事：先检查 H1 的 BatchNorm 运行统计是否会影响结果，再把共享三通道热图头拆成 PS 与 FH 两个独立头做监督对照。它们都是进入后续方法前的排查，不是完整 GeoEqui-LD。

BN 检查给出的信号是“有影响，但不能一口咬定是唯一原因”。原 BN 下，H1 的 validation−train MRE_ALL 差距从 e3 的 8.377 px 扩到 e20 的 16.018 px；PS2 差距由 9.032 扩到 24.591 px，FH1 由 8.127 扩到 16.680 px。重估 train 图像统计后，H1 的 epoch 20 validation 明显改善；同样操作放到 epoch 3 best 上，整体指标却变差。具体数字见 [BN_DIAGNOSTICS.md](BN_DIAGNOSTICS.md)。

独立解码器增加 13,920 个可训练参数（29,318,355 → 29,332,275），并通过等价初始化、四样本 tiny-overfit 和 checkpoint 复算。正式 H2 原计划 20 轮，实际完成 16/20；formal elapsed 为 6631.8 秒，低于 7200 秒 formal allocation。运行器在这 7200 秒内给训练后复算留了 600 秒，所以训练循环使用 6600 秒 guard；ledger 另留 120 秒 closing reserve。`budget_exhausted` 表示 training guard 在下一轮前触发，不是 7200 秒实际超时，也不是 3 小时总上限超限；这里只按“16 轮部分结果”汇报。

在可比的 selected best（两者都是 epoch 3）上，H2 的 PS2 从 27.854 降到 24.193 px，FH1 却从 46.837 升到 51.797 px；MRE_ALL 小幅下降，AoP MAE 则升高。简单说，拆头后某些点有改善，但没有形成一致优势。更完整的逐点记录和曲线见 [DECODER_COMPARISON.md](DECODER_COMPARISON.md)。

H2 自身的 validation−train gap 在 e3 为 PS2 9.069、FH1 15.448 px；到 e16 是 PS2 17.910、FH1 14.286 px。PS2 gap 继续扩大，不能用 MRE_ALL 的变化代替逐点判断。

严格对齐到 epoch 16 后，H2 相对 H1 的 PS2 是 +3.1291 px，FH1 是 -9.2642 px，MRE_ALL 是 -0.6016 px，AoP MAE 是 +0.0016°。尤其 PS2 并未改善，所以不能说独立头缓解了后期退步。

旧 U-Net B2 的 epoch 15 best 逐点为 PS1 13.4168、PS2 20.3575、FH1 40.5640、MRE_ALL 24.7794 px，AoP MAE 8.51385°。这里只保留为历史量级参照；它与 HRNet 的架构和预算不同，不拿来证明哪个结构更优。

## 当前可以下的结论

1. H1 的 validation 波动对 BN 运行统计敏感，但现有诊断不足以确认单一原因。
2. PS/FH 独立头在工程上可用，初始化和训练链路都通过；本次单 seed、16/20 轮结果没有证明它整体优于共享头。
3. H2 的 PS2 selected-best 结果更好，FH1 与 AoP selected-best 结果更差，后续若继续需要分别看点，而不能只报 MRE_ALL。
4. 这一阶段仍是有标注监督对照，没有 EMA、伪标签、无标签一致性或半监督结论。

所有公开数字都是 train/validation 聚合结果，testing 保持冻结；公开目录不含权重、逐样本预测、真实图像或本机路径。
