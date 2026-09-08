# 来源与发布范围

更新于 2026-09-08。以下记录研究与实现来源，并区分本次公开的查阅摘要、未公开的研究材料和私下提供的附件。

## 研究与实现来源

- 研究任务、HRNet 主干方向、PS/FH 专业增强和半监督总体思路依据导师提供的材料；本项目的实现与试验在此基础上开展。
- 代码实现、接口适配、自动化检查和实验整理使用了 AI 工具辅助。具体实现选择与实验结果由项目记录说明，AI 辅助不改变导师方案和第三方实现的来源。
- 本次五份文档不附导师原始材料、医学图像、逐样本标签、训练 checkpoint、私人 manifest、通信记录或内部整理指令。

任务定义和数据口径还参考了 [IUGC 2025 Codabench 页面](https://www.codabench.org/competitions/7105/)、[官方 baseline 仓库](https://github.com/0oTyTo0/IUGC2025)和 [Zenodo 数据记录](https://zenodo.org/records/17355570)。已公开的 baseline 复现版本通过 `third_party/IUGC2025` 子模块引用官方 T10，来源和固定版本见[既有复现报告](https://github.com/heixuan-vsmenghu/GeoEqui-LD/blob/e3af9e2a2c29e5de3beab912fe1ecf7592300f8b/reports/baseline_reproduction/BASELINE_REPRODUCTION.md)；它与本仓库自定义 U-Net 和 strict HRNet 实现分开。

目标幅度、MSE 空间归约、角度单位和近零射线处理均在[当前结果与实现说明](CURRENT_PROGRESS.md)中列出，并与材料明确规定的内容区分。该页另记 31,121 / 31,421 两项无标签数量的来源，差异原因尚未解释。

## 直接依赖

HRNet 通过 `timm` 接口使用，变形卷积通过 `torchvision.ops.DeformConv2d` 使用。以下表格对应已核对的基线版本及本地研究环境，不表示当前 `main` 已包含近期 strict 完整代码。直接运行依赖和开发依赖如下。许可标识沿用此前包元数据和上游 LICENSE 核对记录，本轮未安装、升级依赖或重新作全量许可审计；最终条款以各上游随包提供的 LICENSE 为准。

| 包 | 项目版本或约束 | 上游许可标识 |
|---|---:|---|
| matplotlib | 3.10.0 | Matplotlib license（包元数据标为 PSF 类许可） |
| numpy | 1.26.4 | BSD-3-Clause |
| pandas | 2.2.3 | BSD-3-Clause |
| Pillow | 11.1.0 | MIT-CMU |
| PyYAML | 6.0.3 | MIT |
| huggingface-hub | 0.36.2 | Apache-2.0 |
| safetensors | 0.6.2 | Apache-2.0 |
| timm | 1.0.28 | Apache-2.0 |
| torch | 2.5.1 | BSD-3-Clause |
| torchvision | 0.20.1 | BSD-3-Clause |
| tqdm | 4.67.1 | MPL-2.0 AND MIT |
| pytest（开发） | 8.4.2 | MIT |
| mypy（声明的开发依赖，CI 未执行） | >=1.11,<2 | MIT |
| ruff（开发/CI） | >=0.9,<1；当前 CI 固定 0.9.10 | MIT |

依赖版本来自 `pyproject.toml`、`requirements.txt` 和项目虚拟环境的包元数据。仓库的 MIT 许可不会把这些第三方库重新许可为本项目所有。

## 当前许可和访问范围

2026-09-08，本次五份查阅文档发布于 **public** 仓库的默认分支 `main`；旧分支文档中的 private 描述是历史状态。近期 strict 源码、完整运行记录和权重仍在本地，未随本次文档更新发布。仓库可访问性本身不作为导师或数据权利人的公开授权。

| 使用场景 | 当前说明 |
|---|---|
| 本课题研究使用 | 已有监督和短程半监督运行记录；实验已发生不等于可以转授数据或公开衍生物 |
| 向导师私下查阅 | Word/PDF 附件与仓库说明分别保存；当前仅完成材料准备，未发送，附件不在公开提交清单中 |
| 对公众发布 | 本次发布五份查阅文档及聚合结果；既有公开代码和历史报告保持原版本，近期 strict 完整实现未随本次更新发布 |
| 发布医学图像或标签 | 未获明确许可，不发布 |
| 发布 checkpoint、可视化或逐样本预测 | 数据协议和组织者许可尚未明确，不发布 |

当前仓库保留原有 [MIT LICENSE](../LICENSE)。本次未调整仓库可见性或历史；已存在的克隆和缓存不属于本次文档整理的处理范围。MIT 文件不替导师许可研究设计，也不授予数据、医学图像、标注、预训练权重或第三方实现的权利。

查阅文字只摘录聚合结果，路径使用 `DATA_ROOT`、`RUN_DIR` 等占位符。旧通信文件及历史内容的处理另行决定；本次更新不覆盖全部历史提交、外部克隆或缓存。

以上是项目现状记录，不是对相互冲突的数据许可作法律裁定。研究使用、向导师私下展示和向公众发布是三个不同范围，需分别确认。
