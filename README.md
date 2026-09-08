# GeoEqui-LD

本项目依据导师提供的方案，研究产时超声三关键点检测与产程进展角（AoP）测量。代码实现、接口适配和实验整理使用了 AI 工具辅助，具体来源见[来源说明](docs/ATTRIBUTION_AND_RELEASE_SCOPE.md)。

## 当前进展

截至 2026-09-08，本地记录支持以下结论：

- 官方 IUGC 2025 `UNet Heatmap / T10` 的训练与冻结评价已完成，但论文数值尚未完全复现。
- strict GeoEqui-LD 已完成总计 200 轮纯监督预热。选中 epoch 39 的 raw-BN、hard-argmax validation MRE_ALL 为 **31.059 px**；完成轮数不等于稳定收敛。
- 同一 checkpoint、同一 validation 的普通 Softmax DSNT MRE_ALL 为 **153.767 px**，与 argmax 的差距仍未解决。已有 Stage 2 短试跑出现退化，尚未证明半监督联合训练有效。
- 本次公开更新为阶段结果摘要和查阅文档；近期 strict 源码、完整运行报告与模型权重仍保留在本地，未随本文发布。`main` 中现有代码属于早期探索版本，不代表当前 strict 完整实现。

建议先看[导师查阅索引](docs/ADVISOR_REVIEW_INDEX.md)，再看[当前结果与实现说明](docs/CURRENT_PROGRESS.md)。

## 任务与模型

关键点固定顺序为 `PS1, PS2, FH1`：PS1、PS2 是耻骨联合两个端点，FH1 是胎头切点。AoP 是以 PS1 为顶点、由两条射线 `PS1→PS2` 和 `PS1→FH1` 构成的无向夹角。

strict 实现采用单通道 `512×512` 输入、HRNet-W32、PS/FH 专业增强与独立解码器，输出三张 `256×256` 热图。Stage 1 使用 sigma=4 的高斯目标和纯 raw heatmap MSE；普通空间 Softmax DSNT 在该阶段仅用于诊断，不参与损失或选模。坐标体系见[坐标约定](https://github.com/heixuan-vsmenghu/GeoEqui-LD/blob/e3af9e2a2c29e5de3beab912fe1ecf7592300f8b/docs/COORDINATE_CONVENTION.md)，导师文字与具体实现选择的区别见[当前说明](docs/CURRENT_PROGRESS.md#实现口径)。

## 结果与评价方式

| 记录 | checkpoint 与口径 | MRE_ALL | AoP MAE |
|---|---|---:|---:|
| 官方 T10 本地复现 | epoch 150 final，validation，argmax | 31.6133 px | 15.7964° |
| strict Stage 1 selected | epoch 39，validation，raw-BN argmax | 31.059 px | 12.312° |
| strict Stage 1 final | epoch 200，validation，raw-BN argmax | 92.855 px | 50.086° |
| 同一 strict selected 的 DSNT 读出 | epoch 39，同一 validation、raw BN | 153.767 px | 37.293° |

T10 的 final 是评价前固定的，strict selected 则由 validation 选出；T10 的 validation AoP 还对 1 个退化预测作 180° 惩罚，strict 报告另记数值可计算数量。这不是同协议的优劣或因果比较。31.059 px 不是 Testing 成绩、DSNT 成绩或最终半监督成绩。完整同轮指标和局限见[当前结果](docs/CURRENT_PROGRESS.md)及[官方 T10 复现报告](https://github.com/heixuan-vsmenghu/GeoEqui-LD/blob/e3af9e2a2c29e5de3beab912fe1ecf7592300f8b/reports/baseline_reproduction/BASELINE_REPRODUCTION.md)。

早期 U-Net、H1/H2/H3、辅助损失、BN 和 DSNT 对照保留为[历史探索记录](docs/history/RESEARCH_INDEX.md)，与当前方法配置分开记录，负结果仍可追溯。

## 本地代码与产物

以下是本地未提交实现的相对路径，不是已发布的网页链接：

- `src/geoequi_ld/models/strict.py`：strict 模型及普通 Softmax DSNT。
- `src/geoequi_ld/training/strict_losses.py`：监督和伪标签热图 MSE。
- `src/geoequi_ld/training/strict_runner.py`：训练与评价流程。
- `scripts/run_strict_stage1_200e.py`、`scripts/run_strict_stage2_original_5e.py`：已有实验入口。

本地运行记录采用的项目 Python 版本范围为 3.10–3.12；已公开基线版本的依赖声明见 [pyproject.toml](https://github.com/heixuan-vsmenghu/GeoEqui-LD/blob/e3af9e2a2c29e5de3beab912fe1ecf7592300f8b/pyproject.toml)。数据位置以 `DATA_ROOT` 表示，运行输出以 `RUN_DIR` 表示；权重、医学图像、逐样本记录和私人材料另行保存，不随本次文档更新提供。

## 查阅与发布范围

2026-09-08，本次查阅文档更新发布于公开仓库的默认分支 `main`。官方 T10 与历史探索记录链接到既有公开版本的固定提交；近期 strict 结果在本文中作聚合摘录，完整实现和原始记录仍在本地。

研究使用、向导师私下提供材料和公开发布的范围分别说明，见[来源与发布范围](docs/ATTRIBUTION_AND_RELEASE_SCOPE.md)。

仓库保留现有 [MIT LICENSE](LICENSE)。它不自动授予医学数据、标注、权重、导师研究设计或第三方库的权利。
