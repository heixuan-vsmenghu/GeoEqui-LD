# 来源与发布范围

## 研究与实现来源

- 研究任务、HRNet 主干方向、PS/FH 专业增强和半监督总体思路依据导师提供的材料，不作为学生独立提出的方法申明。
- 仓库中的代码实现、接口适配、自动化检查和实验整理是在 AI 辅助下完成的。提交和报告记录实际执行结果，不声称全部代码由学生独立手写，也不声称导师已经批准公开。
- 当前跟踪内容中没有导师原始文档、医学图像、逐样本标签、训练 checkpoint 或私人 manifest。

任务定义和数据口径还参考了 [IUGC 2025 Codabench 页面](https://www.codabench.org/competitions/7105/)、[官方 baseline 仓库](https://github.com/0oTyTo0/IUGC2025)和 [Zenodo 数据记录](https://zenodo.org/records/17355570)。当前 `src/` 中没有直接复制这些页面或 baseline 仓库的实现文件；它们主要用于核对关键点顺序、数据划分和访问说明。

## 直接依赖

项目没有在 `src/` 中复制一份第三方 HRNet 或 DeformConv 实现；HRNet 通过 `timm` 接口使用，变形卷积通过 `torchvision.ops.DeformConv2d` 使用。项目声明的直接运行依赖和开发依赖如下，表中许可标识结合项目虚拟环境中的包元数据和上游 LICENSE 核对，最终条款仍以各上游项目随包提供的 LICENSE 为准。

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

| 使用场景 | 当前说明 |
|---|---|
| 本课题研究使用 | 监督实验已有记录；无标签数据仍需确认项目授权和完整归档 |
| 向导师私下查阅 | 可使用本地 Word/PDF；查看 private 仓库需要导师自己的 GitHub 账号获得访问权限 |
| 对公众发布源代码 | 仓库现在为 private；导师方案和代码的后续公开范围尚未得到明确确认 |
| 发布医学图像或标签 | 未获明确许可，不发布 |
| 发布 checkpoint、可视化或逐样本预测 | 数据协议和组织者许可尚未明确，不发布 |

当前仓库保留原有 [MIT LICENSE](../LICENSE)。改为 private 不会撤回此前已按该许可证取得的源代码副本，也不会使历史公开事实消失。MIT 文件只覆盖仓库中的原创源代码；它不替导师许可研究设计，也不授予数据、医学图像、标注、预训练权重或第三方实现的权利。

以上是项目现状记录，不是对相互冲突的数据许可作法律裁定。研究使用、向导师私下展示和向公众发布是三个不同范围，需分别确认。
