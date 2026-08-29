# Phase 1A 小结

这轮主要做了两件事：先把 Phase 0.6 里纯 MSE 的异常拆开看，再把 HRNet-W32 作为共享解码头的监督参考接进来。所有判断都只用 train 和 validation，testing 没有读取或重新评估。

## 已实现测试

- B0/B1/B2 共 6 个保存端点的完整 validation 诊断，包括 raw heatmap、softmax 概率、DSNT/argmax、射线长度和 AoP 无效原因；
- 标准高斯、低振幅高斯、零热图和平坦热图的解码检查；
- B3 的 stage4 四尺度、梯度、参数更新、train/eval 与 checkpoint 往返检查；
- B4 的四样本数值门槛、坐标换算检查和 4 张叠加图人工检查；
- H1 的 20 行 validation 日志、best 选择 tuple、best/last checkpoint 配置和保存指标复算。

最终本地测试：`138 passed`；ruff 全仓检查通过。

GitHub Actions 覆盖 ruff、pytest 与 7 个 Phase 1A 命令行入口，远端结果以当前分支的 Actions 状态为准。

## 真实运行

B0 的 best 在第 120 轮。用 DSNT 解码时 MRE_ALL 为 **148.471 px**、有效样本 AoP MAE 为 **20.551°**；同一 checkpoint 改用 argmax，MRE_ALL 是 **71.720 px**。到第 200 轮，三张输出热图已经变成空间常数，raw heatmap 的空间标准差为 0，DSNT 三点落在同一中心位置，因此有效 AoP 变成 0/100；此时 180° 是无效预测的惩罚选择分数，不是实测 AoP MAE。

用 train 标签均值直接在 validation 上预测，MRE_ALL 为 **53.555 px**、AoP MAE 为 **12.988°**。这不是图像模型，只是说明 B0 best 的 DSNT 坐标甚至没有超过一个不看图的均值参考。现有证据支持“响应过平、DSNT 被背景质量主导”这一描述；由于没有保存崩溃转折区间的 checkpoint，不能再往前写成已经定位到某一个训练机制。

四样本纯 MSE 诊断也完整跑满 1000 步，但结果不是成功拟合：PS1 约 3.036 px，PS2 约 184.038 px，FH1 约 84.041 px，整体 MRE **90.372 px**。因此这里把“程序完整执行”和“三点学会”分开记录。

B3 结构探针通过；B4 固定四样本跑满 500 步，eval MRE_ALL **4.6088 px**，4/4 AoP 有效，叠加图人工检查通过。H1 监督参考完整跑完 20 轮，best 为第 3 轮：MRE_ALL **32.391 px**、AoP MAE **12.130°**；第 20 轮分别为 **39.642 px** 和 **23.109°**，best/last 都是 100/100 有效 AoP。独立只读复算的 best/last 保存指标最大差值为 0。

## 仅合成

标准高斯用 argmax 的误差是 0；温度 0.05 的 DSNT 误差只有 **0.0018 px**，但温度 1 时会被大片背景质量拉向中心，误差变成 **104.044 px**。即使仍是正确位置的高斯，把振幅缩到 0.1 后，当前温度 0.05 的 DSNT 误差也达到 **103.669 px**。零热图和三张平坦热图都会得到无效 AoP。这一组只验证数学与接口，不替代真实模型结果，也不单独证明 B0 崩溃的训练原因。

## 结果未改善

旧 U-Net B2 的 validation best 是 24.779 px / 8.514°。两者架构不同，这里不做因果比较。H1 的 best 很早、后续波动明显；batch size 1 的 BatchNorm 是需要留意的风险，但现有实验没有证明它就是波动原因。

A4 虽然跑满预算，但没有学会 PS2/FH1；H1 第 20 轮也没有超过自身第 3 轮的 validation best。因此本轮证明的是 HRNet 接入与训练闭环成立，不是正式性能已经改善，更不是半监督方法已经有效。

## 预算内未完成

无。B3、A4、B4 和 H1 都在各自上限内结束；实验 ledger 为 **87.87 分钟**，加上独立只读复算后总审计口径约 **88.23 分钟**，低于 180 分钟总预算。

## 外部阻塞

就 Phase 1A 的监督参考范围而言没有外部阻塞。后续半监督阶段仍缺少可核验的无标签池，但这不影响本轮监督诊断与 HRNet 接入验收。

## 结论边界

HRNet-W32 已按 stage4 最终融合的高分辨率分支接入，四样本门槛证明模型、坐标和反向传播链路能共同工作，20 轮完整运行也证明正式训练闭环可复现。它现在仍只是共享解码头的增强监督参考：没有 PS/FH 解耦、EMA 教师、伪标签或半监督损失，也没有 testing 结论。

实现细节见 [HRNET_IMPLEMENTATION.md](HRNET_IMPLEMENTATION.md)，脱敏数字见 [aggregate_results.json](aggregate_results.json)。
