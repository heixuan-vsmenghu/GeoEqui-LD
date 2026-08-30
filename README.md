# GeoEqui-LD

这是一个产时超声三关键点检测的毕业设计工程。本项目依据老师提供的研究方案开展，代码实现、接口适配和实验整理使用了 AI 工具辅助，具体方案来源和实现差异见[来源说明](docs/ATTRIBUTION_AND_RELEASE_SCOPE.md)。

当前分支先按导师要求复现 IUGC 2025 官方 `UNet Heatmap / T10` baseline。官方代码来源、固定 commit、Kaggle 数据和实际结果统一记录在[官方 baseline 复现报告](reports/baseline_reproduction/BASELINE_REPRODUCTION.md)。下面的 H1/H2/H3 与损失对照是此前探索记录，不是官方 baseline，本轮已暂停。

项目最终希望研究几何等变一致性能否帮助模型利用无标签图像。目前监督模型和几何一致性计算接口已经实现，但真实无标签数据上的半监督训练还没有开始。

仓库当前为私有协作状态。它此前曾公开过，改为 private 不能收回已经下载或复制的历史内容；后续代码、权重和数据衍生物的公开范围仍需确认。详细说明见[来源与发布范围](docs/ATTRIBUTION_AND_RELEASE_SCOPE.md)。

## 研究任务与方案来源

输入是一张经会阴超声灰度图，模型按固定顺序预测三个点：

```text
KEYPOINT_ORDER = (PS1, PS2, FH1)
```

- `PS1`、`PS2`：耻骨联合两个端点；
- `FH1`：胎头切点；
- `AoP`：以 PS1 为顶点，由 `PS1→PS2` 与 `PS1→FH1` 构成的无向夹角。

默认输入为 `512×512`，预测三张 `256×256` 热图，再通过 DSNT 得到 `[x,y]` 坐标。完整坐标定义见[坐标约定](docs/COORDINATE_CONVENTION.md)。HRNet、PS/FH 专业增强和半监督方向来自导师材料；仓库记录的是具体工程实现与实际实验结果。

## 已有实现

- 有标签数据的文件、标注、坐标范围、重复内容和分区统计检查；
- 小型 U-Net 监督参照；
- HRNet-W32 共享解码器 H1；
- PS/FH 独立解码器 H2；
- H3：PS 分支采用可变形卷积与空间注意力，FH 分支采用多尺度卷积与通道注意力；
- 热图 MSE、坐标 SmoothL1、分布 JS、DSNT、MRE 和 AoP；
- 相似变换的正逆变换、双视图坐标对齐，以及几何一致性损失的梯度接口；
- 配置、训练记录、聚合结果和主要接口测试。

H3 当前使用的监督损失是 `MSE + 10×坐标 SmoothL1 + JS`，DSNT 温度为 `0.05`。这是额外的工程对照，不是导师原文只使用高斯热图 MSE 的监督定义。

## 现有实验结果

下表都是官方 100 张 validation 上按既定规则选出的存档。MRE 和 AoP 越低越好。

| 模型 | 实际训练轮数 | 选中轮次 | 分区 | 监督损失 | MRE_ALL | AoP MAE |
|---|---:|---:|---|---|---:|---:|
| 小型 U-Net | 20 | 15 | validation | MSE + 10×坐标 SmoothL1 + JS | 24.779 px | 8.514° |
| HRNet 共享解码 H1 | 20 | 3 | validation | MSE + 10×坐标 SmoothL1 + JS | 32.391 px | 12.130° |
| HRNet 独立解码 H2 | 16 | 3 | validation | MSE + 10×坐标 SmoothL1 + JS | 31.185 px | 13.563° |
| HRNet 专业增强 H3 | 16 | 14 | validation | MSE + 10×坐标 SmoothL1 + JS | 24.901 px | 10.289° |

这些结果有几项需要一起看：

- 同一 validation 同时用于选模和汇报，当前结果主要是单 seed 描述；
- 模型结构、实际训练轮数和部分归一化设置并不完全相同，不能把表格直接解释为严格因果对照；
- 对齐到第 16 轮时，H2 为 `28.419 px / 13.933°`，H3 为 `25.936 px / 14.279°`。H3 的定位误差更低，但角度误差更高；
- H3 尚未在表中两项汇总指标上超过早期 U-Net，也不是每个点、每轮都改善；
- 早期 U-Net 方案冻结后曾进行一次 testing 评估，后续阶段没有重新用 testing 选参数或汇报新成绩。

完整逐点结果见[H1/H2/H3 对照](reports/phase1c/SPECIALIZED_COMPARISON.md)，解释边界见[结果解释与勘误](docs/RESULT_INTERPRETATION_NOTES.md)。

## 尚未完成的部分

- 真实无标签图像上的半监督训练；
- EMA 教师模型；
- 伪标签和双重置信度筛选；
- 无标签图像裁切后的真实可见性策略；
- 几何项与监督项的正式权重选择；
- HRNet 架构对照及后续半监督方案的多随机种子复核与稳定性评估尚未完成；早期 U-Net 损失对照已用 seed 43、44、45 进行复核，见 [Phase 0.5 记录](reports/phase05/SUPERVISED_ABLATION.md)。当前实验不作临床可用性结论。

几何一致性接口已经完成坐标逆变换、有效性处理、AoP 差和双分支梯度检查。这只说明计算链路能够运行，不表示模型精度已经提高。具体定义见[几何一致性接口](reports/phase2a/GEOMETRY_CONTRACT.md)。

## 当前数据问题

项目不是“没有图像数据”。已有监督实验使用的是 train 300 张、validation 100 张；早期冻结方案还对 testing 501 条记录评估过一次。

当前缺的是可以正式进入训练的无标签池，验收候选数仍为 0，原因有两项：

1. 本课题是否被现有数据授权覆盖还需要书面确认；
2. Zenodo 公开归档虽然大小和 MD5 与发布记录相符，但 ZIP 中央目录不能正常读取。

因此目前状态仍是 `BLOCKED_ACCESS + BLOCKED_INTEGRITY`，没有把流式检查中观察到的成员下界当作可训练数据。简要证据见[数据包核查记录](reports/phase2a_closeout/ARCHIVE_ISSUE_BRIEF.md)，取得获准完整归档后的步骤见[接收检查清单](reports/phase2a_closeout/APPROVED_ARCHIVE_ACCEPTANCE_CHECKLIST.md)。

## 运行方法

Python 版本范围为 3.10–3.12。PyTorch 与 torchvision 的 CPU/CUDA 版本需要按运行机器选择。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
```

本地数据路径和运行产物不会提交 Git。需要运行具体阶段时，先查看对应脱敏配置和命令帮助：

```powershell
python scripts/train_phase1c_specialized.py --help
python scripts/check_phase2a_h3_geometry.py --help
python scripts/audit_phase2a_unlabeled.py --help
```

这些命令入口不代表当前具备启动无标签训练的授权或完整数据。

## 记录索引

给导师查看时建议从[导师查阅索引](docs/ADVISOR_REVIEW_INDEX.md)开始。该索引按当前状态、架构对照、几何接口、数据问题和历史负结果排列，不需要从提交历史逐个寻找。

主要机器可读结果保存在各阶段的 `aggregate_results.json`，原始训练产物、checkpoint、医疗图像、逐样本预测和私人 manifest 均不在 Git 中。

## 许可

仓库保留现有 [MIT LICENSE](LICENSE)，只覆盖仓库中的原创源代码，不自动授予数据、医学图像、标注、预训练权重、导师研究设计或第三方库的权利。详细依赖和使用范围见[来源与发布范围](docs/ATTRIBUTION_AND_RELEASE_SCOPE.md)。
