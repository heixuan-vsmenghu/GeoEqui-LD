# Phase 1A：HRNet 监督参考实现

> 查阅注记（2026-08-29）：本页的叠加图查看属于 AI 辅助视觉检查，没有独立审阅者或医学复核记录。详见[结果解释与勘误](../../docs/RESULT_INTERPRETATION_NOTES.md)。

这一步先把老师资料里的 HRNet-W32 接到现有三关键点流程里，范围保持得比较窄：灰度输入、共享三通道热图头、纯监督 B2 损失。这里还没有做 PS/FH 解耦，也没有加入 EMA、伪标签或无标签一致性。

## 接入方式

- `timm==1.0.28`，`hrnet_w32`，`pretrained=False`，单通道输入；
- `feature_info` 核验为 `channels=(32,)`、`reduction=(4,)`，固定 `feature_location=''`、`out_indices=(1,)`；hook 路径为 `backbone.stage4.2`，取其四尺度融合后的高分辨率输出 `[B, 32, 128, 128]`，不是 stem；
- 共享解码头为 `3×3 Conv 32→32 + BN + GELU`、`3×3 Conv 32→16 + BN + GELU`、`1×1 Conv 16→3`，最后双线性插值到 `256×256`；
- 模型共有 **29,318,355** 个可训练参数；
- 优化器仍是 Adam，batch size 为 1，FP32；为控制 4 GB 显存峰值，Adam 使用 `foreach=False`，没有偷偷改输入尺寸或启用 AMP。

## B3 结构探针

B3 通过。512×512、batch 1 的完整 Adam 更新中，stage4 四个尺度都有非零梯度；backbone 与 decoder 均发生参数更新，train/eval 切换和 checkpoint 往返也一致。峰值 allocated / reserved 显存分别为 **1.10 / 1.22 GiB**。第一次 Adam 完整更新约 **1.890 s**，预热后的完整训练步约 **0.406 s**；探针实际用时 **41.50 s**，低于分配的 **900 s**。

## B4 四样本门槛

HRNet 在固定 4 张训练样本上跑满 500 步，eval 模式得到 MRE_ALL **4.6088 px**、AoP MAE **1.7401°**，4/4 AoP 有效，没有非有限值或坐标换算错误。四张叠加图经 AI 辅助逐张查看，关键点通道、目标圆圈和预测叉号没有发现可见错位，因此完成了这项实现检查；它不是独立医学复核。

作为诊断对照，同样跑满预算的轻量 U-Net 纯 MSE 只把 PS1 学到约 3 px；PS2 与 FH1 分别约 184 px 和 84 px，整体 MRE 为 90.372 px。它的执行过程完整、数值也有限，但没有通过三点学习判据。

## 20 轮监督参考

H1 完整跑完 20 轮，checkpoint 只按 validation 的 `(AoP MAE, MRE_ALL, 较早 epoch)` 选择：

| checkpoint | epoch | PS1 MRE | PS2 MRE | FH1 MRE | MRE_ALL | AoP MAE | 有效 AoP |
|---|---:|---:|---:|---:|---:|---:|---:|
| best | 3 | 22.483 | 27.854 | 46.837 | 32.391 | 12.130° | 100/100 |
| last | 20 | 20.148 | 54.646 | 44.131 | 39.642 | 23.109° | 100/100 |

best 出现在第 3 轮，之后 validation 有明显波动；第 20 轮仍保持 100/100 有效 AoP，但两项选择指标都不如 best。波动是实际观察，batch size 1 下的 BatchNorm 统计量只是一个可能风险，目前不能把它写成已证实的原因。这组结果更适合作为“HRNet 已正确接入并能训练”的监督参考，不应包装成稳定性已经解决。

旧的 U-Net B2 在另一轮实验中的 validation best 是 MRE_ALL 24.779 px、AoP MAE 8.514°。由于 backbone、参数量和训练阶段都不同，这个数字只放在旁边帮助定位量级，不用来得出“HRNet 更好/更差”的因果结论。

实验 ledger 记录的 GPU 用时为 **87.87 分钟**；另一次只读 best/last validation 复算约 **0.35 分钟**，所有保存指标复现差值为 0。两部分合计约 **88.23 分钟**，低于本阶段 180 分钟总预算。

曲线见 [validation_metrics.png](curves/validation_metrics.png)，公开汇总见 [aggregate_results.json](aggregate_results.json)。
