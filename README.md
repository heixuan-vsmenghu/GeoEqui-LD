# GeoEqui-LD

GeoEqui-LD 是一个面向产时超声三关键点检测的毕业设计工程。项目最终希望研究几何等变一致性能否帮助模型利用无标签图像；Phase 0 基础闭环、Phase 0.5/0.6 监督损失审计，以及 Phase 1A–1C 的 HRNet 监督架构检查均已完成。

目前没有把完整半监督方案包装成“已经实现”：本地尚未取得完整无标签池，因此 EMA 教师、伪标签筛选和无标签几何一致性训练都还没有开始。

## 当前任务

模型输入一张经会阴超声灰度图，按固定顺序预测三个点：

```text
KEYPOINT_ORDER = (PS1, PS2, FH1)
```

- `PS1`、`PS2`：耻骨联合两个端点；
- `FH1`：胎头切点；
- `AoP`：以 PS1 为顶点，由 `PS1→PS2` 与 `PS1→FH1` 构成的无向夹角。

默认张量约定：

```text
输入图像     [B, 1, 512, 512]
预测热图     [B, 3, 256, 256]
DSNT 坐标    [B, 3, 2]，顺序为 [x, y]，范围为 [-1, 1]
```

完整坐标说明见 [docs/COORDINATE_CONVENTION.md](docs/COORDINATE_CONVENTION.md)。

## 已经完成的部分

- 有标签数据只读审计与脱敏统计；
- 显式的 `[x, y]` 像素坐标和 DSNT 归一化坐标转换；
- 关键点高斯热图生成，`sigma=4` 表示热图像素；
- 相似变换的正变换、逆变换和跨视图点映射；
- 稳定的 AoP 和 MRE 计算；
- 可微 DSNT；
- DSNT 概率图的 Gaussian 分布约束，避免只优化坐标却得到不可解释的响应图；
- 一个输入单通道、输出三通道半分辨率热图的小型 U-Net；
- 坐标、热图、DSNT、AoP、变换、指标、访问策略、模型、监督训练和结果汇总均有自动化测试；
- 4 张真实训练图上的 tiny-overfit 门槛已经通过。
- 300/100 监督 baseline 已完成，并在方案冻结后对 testing 评估一次。
- HRNet-W32 的共享头、PS/FH 独立头和专业特征增强监督对照已完成。

这些内容说明基础工程可以继续做监督阶段验证，不代表半监督方法或最终论文实验已经完成。

## Phase 0.5 监督消融

Phase 0.5 固定 Phase 0 的数据、轻量 U-Net、随机种子规则和 20 轮预算，只比较三种损失：

| 方案 | 损失 |
|---|---|
| B0 | heatmap MSE |
| B1 | heatmap MSE + coordinate SmoothL1 |
| B2 | heatmap MSE + coordinate SmoothL1 + distribution JS |

三组统一使用 validation 上的 DSNT AoP MAE 选择 checkpoint；B0 还会在同一个 checkpoint 上比较 argmax 和 DSNT。首轮 seed=42，按预先固定的 AoP MAE、MRE_ALL、复杂度顺序留下两个方案，再用全新的 seeds 43/44/45 做确认，避免把筛选用的 seed 42 混进稳定性均值。Phase 0.5 的入口没有 testing 参数，并通过本地数据指纹和访问策略拒绝 testing。

公开协议是 [configs/phase05_ablation.yaml](configs/phase05_ablation.yaml)。本地运行时复制 [configs/phase05_local.example.yaml](configs/phase05_local.example.yaml) 为被 Git 忽略的 `configs/phase05_local.yaml`，再运行：

```powershell
python scripts/train_phase05.py --protocol configs/phase05_ablation.yaml `
  --local-config configs/phase05_local.yaml --variant B0 --seed 42 `
  --output-dir runs/phase05/B0/seed_42
```

实验输出保存在被 Git 忽略的 `runs/`；公开仓库只接收脱敏聚合指标和曲线。

三个新种子（43/44/45）的确认结果如下；seed 42 只用于筛选，没有混入均值：

| 方案 | MRE_ALL（均值 ± 样本 SD） | AoP MAE（均值 ± 样本 SD） |
|---|---:|---:|
| B1 | 39.896 ± 1.170 px | 12.975 ± 0.768° |
| B2 | 24.973 ± 2.282 px | 9.251 ± 2.075° |
| B2 − B1（同种子配对） | −14.923 ± 2.953 px | −3.724 ± 2.810° |

逐种子结果、B0 的 DSNT/argmax 检查、运行时间、显存和限制见
[Phase 0.5 小结](reports/phase05/PHASE05_SUMMARY.md)与
[完整监督消融记录](reports/phase05/SUPERVISED_ABLATION.md)。这里只作 3 个确认种子的 validation 描述性复核，不声称统计显著或测试集结论。

Phase 0 原始 20 轮运行对应的脱敏快照保存在 [configs/phase0_frozen_20e.yaml](configs/phase0_frozen_20e.yaml)，不再用早期的 150 轮通用模板代替实际运行配置。

## Phase 0.6 长预算检查

Phase 0.6 没有改模型或调损失权重，只把 seed 42 下的 B0/B1/B2 用完全相同的
初始化、数据顺序和优化设置跑满 200 轮，用来判断 20 轮结果是不是单纯由收敛速度
造成。checkpoint 仍只按 validation AoP MAE、MRE_ALL、较早 epoch 依次选择，
testing 全程冻结。

| 方案 | best epoch | MRE_ALL | AoP MAE |
|---|---:|---:|---:|
| B0：MSE | 120 | 148.471 px | 20.551° |
| B1：MSE + coordinate SmoothL1 | 194 | 27.084 px | 8.767° |
| B2：B1 + JS | 15 | 24.779 px | 8.514° |

B0 在 20 轮后确实还有改善，但没有追上 B1/B2，并在 epoch 187–200 连续出现
0 个有效 AoP 预测。B1 的优势一直保持到 epoch 200；B2 很早取得更好的 selected
best，但 epoch 200 的两项主指标反而略差于 B1，因此 JS 在这次单 seed 检查中主要
体现为加快收敛，不能据此声称改善长期稳定性。

完整数字和边界见 [Phase 0.6 小结](reports/phase06/PHASE06_SUMMARY.md)、
[200 轮逐节点对照](reports/phase06/LONG_BUDGET_COMPARISON.md)和
[validation 曲线](reports/phase06/curves/validation_metrics.png)。这里只是增强监督
基线审计，不包含 HRNet、PS/FH 解耦、EMA、伪标签或半监督损失。

## Phase 1A：B0 诊断与 HRNet 监督参考

Phase 1A 先回看纯 MSE 的异常，再按老师资料里的主干约定接入 HRNet-W32。B0
第 200 轮的三张 raw heatmap 已经变成空间常数，DSNT 因此把三个点都解到中心，
有效 AoP 降到 0/100。合成高斯检查同时确认：当前 DSNT 计算和坐标接口本身能工作，
但热图振幅过低或过平时，大面积背景概率会把期望坐标拉向图像中心。由于没有保存
崩溃转折区间的 checkpoint，这里只描述端点现象，不把原因写死。

HRNet 使用 `timm==1.0.28`、单通道输入和 stage4 最终融合的高分辨率分支，后接
共享三通道热图头。结构探针在 512×512、batch 1、FP32 下通过，峰值 reserved
显存为 1.22 GiB；固定四样本跑满 500 步后，eval MRE_ALL 为 4.609 px，4/4 AoP
有效，叠加图人工检查也没有发现坐标错位。

20 轮监督参考完整跑完，validation checkpoint 仍按 AoP MAE、MRE_ALL、较早 epoch
依次选择：

| checkpoint | epoch | MRE_ALL | AoP MAE | 有效 AoP |
|---|---:|---:|---:|---:|
| best | 3 | 32.391 px | 12.130° | 100/100 |
| last | 20 | 39.642 px | 23.109° | 100/100 |

第 3 轮之后 validation 波动明显。batch size 1 下的 BatchNorm 是一个需要留意的
风险，但现有结果没有证明它就是波动原因。旧 U-Net B2 的 24.779 px / 8.514°只作
量级参考；两者架构不同，不据此作因果比较。详细过程见
[Phase 1A 小结](reports/phase1a/PHASE1A_SUMMARY.md)、
[HRNet 接入记录](reports/phase1a/HRNET_IMPLEMENTATION.md)和
[validation 曲线](reports/phase1a/curves/validation_metrics.png)。Phase 1A 当时尚未实现
PS/FH 解耦、EMA、伪标签或半监督损失。

## Phase 1B：BN 短诊断与解码器对照

Phase 1B 先固定 H1 的 best/last 权重，只用 train 图像重估一次 BatchNorm
运行统计。epoch 20 端点随之改善，epoch 3 best 的整体结果却变差，因此目前只能说
validation 对 BN 统计敏感，不能把波动的原因完全归到 BN。

随后把共享三通道头拆成 PS 两通道头和 FH 一通道头，参数增加 13,920（29,318,355
→ 29,332,275）。拆分初始化与原共享头输出完全一致，四样本 tiny-overfit 也通过。
H2 的 formal allocation 是 7200 秒，其中给训练后复算预留 600 秒，训练循环使用
6600 秒 guard；ledger 另留 120 秒 closing reserve。训练 guard 触发后，H2 停在
16/20 轮，formal elapsed 为 6631.8 秒。`budget_exhausted` 表示主动停在下一轮之前，
并非 7200 秒实际超时或 3 小时总上限超限。两个方案的 selected best 都在 epoch 3：
H2 的 PS2 更低，但 FH1 与 AoP MAE 更高；严格对齐 epoch 16 后，H2 的 PS2 反而高
3.1291 px，因此没有一致优势，也不能说独立头缓解了后期退步。

详细数字见 [Phase 1B 小结](reports/phase1b/PHASE1B_SUMMARY.md)、
[BN 诊断](reports/phase1b/BN_DIAGNOSTICS.md)、
[解码器逐点对照](reports/phase1b/DECODER_COMPARISON.md)和
[validation 曲线](reports/phase1b/curves/validation_metrics.png)。这仍是单 seed 的
有标签监督对照，不是完整 GeoEqui-LD，也没有引入 EMA、伪标签或半监督损失。

## Phase 1C：PS/FH 专业特征增强

Phase 1C 在 H2 的 HRNet-W32 主干和独立解码器之前加入两个小型专属模块：PS
分支使用带显式 offset 与 modulation mask 的真实 `DeformConv2d` 和空间注意力，
FH 分支使用无 BatchNorm 的 ASPP-lite 与 SE 注意力；两路都以残差和通道
LayerNorm 收尾。基础主干与解码器从同一个 seed 42 起点复制且不共享存储，新增
模块使 H3 的完整初始函数不再与 H2 等价。H3 共 29,372,695 个可训练参数，比 H2
增加 40,420。

真实 CUDA 算子前后向和固定四样本门槛均通过。四样本 500 步后的 eval MRE_ALL
为 4.689 px，AoP 4/4 有效；四张本地叠加图没有发现坐标或通道错位。正式 H3
在固定 300/100、B2 工程监督、FP32 和 16 轮预算下完整跑完：

| 方案 | selected epoch | PS1 | PS2 | FH1 | MRE_ALL | AoP MAE |
|---|---:|---:|---:|---:|---:|---:|
| H1 共享头 | 3 | 22.483 | 27.854 | 46.837 | 32.391 px | 12.130° |
| H2 独立头 | 3 | 17.564 | 24.193 | 51.797 | 31.185 px | 13.563° |
| H3 专业增强 | 14 | 12.426 | 21.446 | 40.831 | 24.901 px | 10.289° |

selected best 上 H3 的五项指标都低于 H1/H2，但逐轮结果并不一致：epoch 3 的
PS2 或 FH1 对照仍有混合变化，epoch 16 的 AoP 又比 H1/H2 高约 0.35°，训练曲线
也有明显波动。因此这里只能说当前单 seed、16 轮 validation 结果支持继续研究，
不能声称稳定胜出。DeformConv2d CUDA backward 在锁定环境中没有严格确定性实现；
本轮固定 seed 与数据顺序并启用 deterministic warn-only，不作位级复现承诺。

详细过程见 [Phase 1C 小结](reports/phase1c/PHASE1C_SUMMARY.md)、
[专业模块结构](reports/phase1c/SPECIALIZED_ARCHITECTURE.md)、
[H1/H2/H3 逐点对照](reports/phase1c/SPECIALIZED_COMPARISON.md)、
[无标签接入状态](reports/phase1c/UNLABELED_INTAKE.md)和
[validation 曲线](reports/phase1c/curves/validation_metrics.png)。H3 仍是使用 B2
损失的增强监督工程参照，不是导师原文的纯 MSE，也没有实现 EMA、伪标签或
无标签一致性损失。

## Phase 2A：数据接入与几何一致性接口

Phase 2A 从冻结的 H3 提交 `2be9d5908ce45ed9afc610908ef27620aa958fb4`
另开分支，没有追加网络结构或训练轮次。无标签工作已经实际下载并核验 Zenodo
归档：文件大小和官方 MD5 匹配，但 ZIP 容器不可完整读取；正式 Codabench 流程又
要求导师/负责人签署协议并由组织者批准。两项阻塞分别记为 `BLOCKED_INTEGRITY`
和 `BLOCKED_ACCESS`，当前不把残缺归档强制解压为训练池。

几何侧新增双视图接口：两个预测先逆变换回共同原图坐标，再计算像素空间 AoP
度数差与归一化坐标距离。合成测试覆盖可逆性、单位、裁切、退化、batch、双分支
梯度和 `lambda_geo=0`；冻结 H3 的单张 train 图像反向检查也通过，checkpoint
前后字节一致。这个结果只证明接口接通，没有新增 validation/testing 指标，也不是
半监督效果。

详情见 [Phase 2A 小结](reports/phase2a/PHASE2A_SUMMARY.md)、
[数据接入审计](reports/phase2a/DATA_INTAKE.md)和
[几何契约](reports/phase2a/GEOMETRY_CONTRACT.md)。

Phase 2A 的对外沟通和获准归档验收材料见
[收口状态](reports/phase2a_closeout/CLOSEOUT_STATUS.md)、
[归档问题说明](reports/phase2a_closeout/ARCHIVE_ISSUE_BRIEF.md)、
[导师进展稿](reports/phase2a_closeout/ADVISOR_PROGRESS_DRAFT.md)、
[组织者咨询模板](reports/phase2a_closeout/ORGANIZER_ACCESS_EMAIL_TEMPLATE.md)和
[接收验收清单](reports/phase2a_closeout/APPROVED_ARCHIVE_ACCEPTANCE_CHECKLIST.md)。

## 数据现状

当前可核验的监督部分是官方公开划分：

| 分区 | 图像数 | 状态 |
|---|---:|---|
| Train labeled | 300 | 可用于监督训练 |
| Validation | 100 | 官方公开验证集 |
| Testing | 501 | 仅用于冻结方案后的最终评估 |
| Unlabeled pool | 0（当前可训练） | 已定位正式来源；访问授权与公开包完整性双重阻塞 |

监督数据本身已通过文件、标签、解码、边界和内容哈希检查。更完整的公开安全统计见：

- [reports/phase0/DATA_AUDIT.md](reports/phase0/DATA_AUDIT.md)
- [reports/phase0/dataset_statistics.json](reports/phase0/dataset_statistics.json)
- [reports/phase0/duplicate_report.csv](reports/phase0/duplicate_report.csv)

无标签数据和分组元数据的基础需求见 [reports/phase0/DATA_REQUIRED.md](reports/phase0/DATA_REQUIRED.md)，当前接入核验见 [reports/phase2a/DATA_INTAKE.md](reports/phase2a/DATA_INTAKE.md)。

## 当前监督结果

| Split | MRE_ALL | AoP MAE |
|---|---:|---:|
| Validation best（epoch 15） | 24.779 px | 8.514° |
| Testing frozen once | 19.930 px | 7.553° |

三点和风险说明见 [BASELINE_REPORT.md](reports/phase0/BASELINE_REPORT.md)，完整阶段总结见 [PHASE0_SUMMARY.md](reports/phase0/PHASE0_SUMMARY.md)。testing 结果没有用于调参；其 501 条记录只对应 493 种唯一图像内容，且患者/视频独立性尚未确认。

## 数据不会进入公开仓库

IUGC 数据包内协议与公开页面的许可表述并不完全一致，而且协议对数据复制、衍生内容和签署流程有限制。因此本仓库只保存代码、合成测试和脱敏聚合结果，不包含：

- 原始图像、标签 CSV 或数据压缩包；
- 带真实图像的可视化；
- 逐样本预测、伪标签或可还原个体数据的缓存；
- 任何本机绝对数据路径。

真实数据应存放在仓库之外，并通过运行时参数传入。公开材料前仍需由导师或数据所有者确认许可范围。

## 运行测试

在 Python 3.11 环境中安装 PyTorch、NumPy、Pandas、Pillow、PyYAML 和 pytest 后运行：

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q -p no:cacheprovider
```

测试数量会随阶段增加，以本地 `pytest` 与 GitHub Actions 的当次输出为准。

## 当前边界

- 无标签正式训练包仍未通过访问和完整性审计；
- tiny-overfit、监督 baseline 与几何一致性接口已完成；完整半监督方法尚未开始；
- 尚未实现 EMA 教师、伪标签、置信度筛选或正式无标签训练；
- 标签中没有患者、病例或视频分组字段，不能证明各 split 在患者级独立；
- 当前任务是关键点检测和 AoP 定量，不是直接预测分娩结局。

Phase 0 的工程决策记录在 [reports/phase0/DECISIONS.md](reports/phase0/DECISIONS.md)。

## 参考来源

- IUGC 2025 官方代码：https://github.com/0oTyTo0/IUGC2025
- IUGC 2025 数据页：https://www.kaggle.com/datasets/aspirexxx/iugc-ultrasound-dataset-miccai-2025
- IUGC 2025 Codabench：https://www.codabench.org/competitions/7105/
- Zenodo 数据记录：https://zenodo.org/records/17355570
- 官方监督 baseline 论文：https://openreview.net/attachment?id=hj7hmvKc2r&name=pdf
