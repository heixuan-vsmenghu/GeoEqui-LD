# GeoEqui-LD

本项目依据导师提供的方案，研究产时超声三关键点检测与产程进展角（AoP）测量。代码实现、接口适配和实验整理使用了 AI 工具辅助，具体来源见[来源说明](docs/ATTRIBUTION_AND_RELEASE_SCOPE.md)。

## 当前进展

截至 2026-09-08，本地记录支持以下结论：

- 官方 IUGC 2025 `UNet Heatmap / T10` 的训练与冻结评价已完成，但论文数值尚未完全复现。
- strict GeoEqui-LD 已完成总计 200 轮纯监督预热。选中 epoch 39 的 raw-BN、hard-argmax validation MRE_ALL 为 **31.059 px**；完成轮数不等于稳定收敛。
- 同一 checkpoint、同一 validation 的普通 Softmax DSNT MRE_ALL 为 **153.767 px**，与 argmax 的差距仍未解决。已有 Stage 2 短试跑出现退化，尚未证明半监督联合训练有效。
- 仓库已接入当前 [strict 模型与普通 Softmax DSNT](src/geoequi_ld/models/strict.py)、[严格热图 MSE](src/geoequi_ld/training/strict_losses.py)及其 HRNet/PS-FH 依赖，配有离线合成测试。长训练启动器、完整运行报告和权重仍未公开；核心源码集成不等于整套实验已经可独立复现。

目前需要核对的是高斯热图 MSE 监督与普通 Softmax DSNT 之间的数值约定。

主要入口：[监督训练结果](reports/review_20260908/SUPERVISED_RESULTS.md)、[当前核心源码](src/geoequi_ld/models/README.md)、[热图与 DSNT 核对](reports/review_20260908/HEATMAP_DSNT_CHECK.md)、[导师查阅索引](docs/ADVISOR_REVIEW_INDEX.md)。结果摘自已有记录，本次源码整理没有新增实验。

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

## 当前源码与产物范围

当前可核读的核心是 [strict.py](src/geoequi_ld/models/strict.py)、[hrnet.py](src/geoequi_ld/models/hrnet.py)、[specialized.py](src/geoequi_ld/models/specialized.py) 和 [strict_losses.py](src/geoequi_ld/training/strict_losses.py)。这些代码取自当前真实工作区，计算保持原样；这次源码集成记录当前版本，不补造历史运行源码版本。实际设置来源见[配置说明](configs/README.md)。

以下仍仅在本地，纯路径不是线上源码链接：`src/geoequi_ld/training/strict_runner.py`、`scripts/run_strict_stage1_200e.py`、`scripts/run_strict_stage2_original_5e.py`。它们是已有训练与评价入口，不包含在本次核心源码集成中。

项目声明的 Python 版本范围为 3.10–3.12，这不是某次运行的实际解释器版本；当前依赖见 [pyproject.toml](pyproject.toml)，[基线版本依赖](https://github.com/heixuan-vsmenghu/GeoEqui-LD/blob/e3af9e2a2c29e5de3beab912fe1ecf7592300f8b/pyproject.toml)另作追溯。数据位置以 `DATA_ROOT` 表示，运行输出以 `RUN_DIR` 表示；权重、医学图像、逐样本记录和私人材料另行保存，不随本次更新提供。

## 历史追溯

早期 U-Net、H1/H2/H3、辅助损失、BN 和 DSNT 对照集中在[历史研究索引](docs/history/RESEARCH_INDEX.md)，与当前方法配置分开，负结果仍可追溯。旧配置和旧 DSNT 参数未改动，不代表当前 strict 运行采用了它们。

## 查阅与发布范围

2026-09-08，在已审阅查阅文档的基础上，仓库补充当前核心源码、必要依赖与测试、目录说明和结果摘录。官方 T10 与历史探索记录链接到既有公开版本的固定提交；完整训练运行系统和原始记录仍在本地。

研究使用、向导师私下提供材料和公开发布的范围分别说明，见[来源与发布范围](docs/ATTRIBUTION_AND_RELEASE_SCOPE.md)。

仓库保留现有 [MIT LICENSE](LICENSE)。它不自动授予医学数据、标注、权重、导师研究设计或第三方库的权利。
