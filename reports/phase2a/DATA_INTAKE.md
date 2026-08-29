# Phase 2A：无标签数据接入审计

> 查阅注记（2026-08-29）：第 2、7 节记录的是 IUGC 2025 页面当时公布的申请方式。比赛时间表已经结束，不能直接假定旧流程在 2026 年仍有效；当前应先询问课题组是否已有覆盖本项目的授权，没有再向组织者确认现行赛后研究申请方式。数据状态仍为 `BLOCKED_ACCESS + BLOCKED_INTEGRITY`。

审计日期：2026-08-29

## 1. 当前结论

当前无标签数据同时处于两个阻塞状态：

- **`BLOCKED_ACCESS`**：比赛正式发放路径要求先申请、签署数据协议并取得组织者批准；目前没有已批准的正式训练包或签署凭据。
- **`BLOCKED_INTEGRITY`**：公开 Zenodo 归档的实际字节数和 MD5 与页面记录一致，但归档容器未通过完整可读性检查，不能安全解压或作为训练输入。

因此当前最终候选无标签池仍为 **0**。本轮没有启动半监督训练，也不把“归档已下载”解释成“数据已可训练”。

## 2. 来源和版本核实

| 来源 | 实际内容 | 版本与校验 | 本轮判断 |
|---|---|---|---|
| [Zenodo 15172238](https://zenodo.org/records/15172238) | 挑战说明 PDF，不是图像数据包 | 页面标记 v2；唯一文件为约 102.7 kB 的挑战说明 | 只能作为任务和数量说明来源 |
| [Zenodo 17355570](https://zenodo.org/records/17355570) | `Dataset.zip` 图像归档 | v1；1,071,644,672 B；官方 MD5 `8f787f72a839cee4ef5bf017f877609d` | 本地文件大小与 MD5 均匹配，但容器完整性未通过 |
| [Kaggle 数据页](https://www.kaggle.com/datasets/aspirexxx/iugc-ultrasound-dataset-miccai-2025) | IUGC 2025 数据页面及另一份归档记录 | 页面元数据为 CC BY-NC 4.0；与 Zenodo 归档不能默认视为同一字节版本 | 当前环境匿名 API 请求返回 404，且没有本地 Kaggle 凭据；未绕过访问控制 |
| [Codabench IUGC2025](https://www.codabench.org/competitions/7105/) | 比赛正式申请和训练数据发放流程 | 页面要求先发送参赛申请，组织者发送数据协议；签署完成后才发训练数据 | 当前缺少签署和组织者批准，属于正式访问阻塞 |

Zenodo 数据记录的开放下载状态，只说明文件可从该页面取得，不自动替代 Codabench 的申请流程、包内协议或医学数据使用边界。

## 3. 下载与归档完整性

`Dataset.zip` 的下载耗时 **6 分 35 秒**，有限重试次数为 **0**。下载后：

- 实际大小与官方 1,071,644,672 B 一致；
- 实际 MD5 与官方值一致；
- Python `zipfile` 未能将其识别为完整可读 ZIP；
- 只读流式 `tar` 检查观察到至少 **30,407** 个位于 `Unlabeled` 范围的图像成员，但命令退出码为 **1**。

30,407 只是失败前观察到的下界，不能当作归档总数或可用图像数。由于容器检查失败，本轮没有解压、没有图像解码、没有计算候选图像哈希，也没有进行去重。

| 接入阶段 | 当前结果 |
|---|---:|
| 完整字节下载 | 1 个归档，1,071,644,672 B |
| 流式观察到的无标签图像成员 | ≥30,407，非完整计数 |
| 成功安全解压 | 0 |
| 可解码候选图像 | 0 |
| 字节去重后 | 0 |
| 像素去重后 | 0 |
| 排除有标签分区重叠后 | 未执行 |
| 最终候选无标签池 | **0** |

H 盘剩余空间超过 **102 GB**，存储空间不是当前阻塞原因。当前阻塞来自访问授权和归档完整性，不能靠重复下载或强制解压掩盖。

## 4. 数量口径

[Codabench 数据说明](https://www.codabench.org/competitions/7105/)和[挑战 Zenodo 说明](https://zenodo.org/records/15172238)均把 `Unlabeled cases` 写为 **31,421**。Codabench 同时说明 2,045 张 `ExampleX` 是无标签目录中给出的标准切面参考，因此：

- 31,421 是本轮采用的官方无标签口径；
- 2,045 属于其中的 Example 参考子集，不能再加到 31,421 上；
- 31,121 来自另一种口径：[官方 baseline 论文](https://openreview.net/pdf/71922c24febd48892f29252646cb7d8d0c741447.pdf)把训练集写成 31,421 例且“包含 300 个标注样本”，据此才会得到 `31,421 - 300 = 31,121`。它不是本轮采用的官方无标签池计数。

[Zenodo 数据记录](https://zenodo.org/records/17355570)的说明文字还同时出现“28,919 ultrasound images and videos”和“300 labeled + 31,421 unlabeled cases”，两者在同一页内不能直接互相校验。本轮不把 28,919 当作归档成员数，也不通过加减消除这一发布方口径冲突。

归档当前不可完整读取，所以这些数字仍是文档口径，不是本轮重新枚举得到的验收结果。

## 5. 许可与公开边界

目前可见的许可表述并不一致：

- Zenodo 17355570 元数据标记为 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)；
- Kaggle 页面标记为 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)；
- 包内协议文本又同时出现 `CC BY-NC-ND` 与 `CC BY-NC-SA` 表述，并包含申请和签署要求。

本报告不对这些冲突作法律结论，也不选择其中最宽松的一条来替代其余条件。在组织者明确适用版本前，不公开医学图像、逐样本清单、逐样本预测或 checkpoint；“可以下载”不等于“可以训练后公开权重”。

## 6. 重叠审计状态

早期封存的图像指纹清单包含：

| 分区 | 封存指纹行数 |
|---|---:|
| train | 300 |
| validation | 100 |
| testing | 501 |

当前没有成功解压的候选池，因此没有候选指纹可以与这些封存集合求交。与 train、validation、testing 的内容重叠状态均为 **未知**，不是“已通过”。患者、病例和视频层级独立性同样未知。

本轮没有读取真实 testing 图像或标签，没有运行模型，也没有进行 testing 评估。审计过程中曾有一次 `tar -tf` 命令错误地把部分 testing 成员的**文件名元数据**输出到本地终端；本报告不记录这些名称。发现后已停止这种输出方式，后续检查改为只返回聚合计数。该事件没有读取图像内容或标签，也没有计算 testing 哈希或指标。

## 7. 最少人工动作

要解除当前阻塞，只需要用户和导师完成以下动作：

1. 由导师或课题负责人按数据包内协议完成签署；不要由 Codex 代签。
2. 使用学校邮箱向 Codabench 页面列出的官方邮箱 **`fugc.isbi25@gmail.com`** 发送参赛/研究使用申请，提供真实姓名、学校与院系、导师或队长信息，并按组织者回复提交签署件。
3. 在同一邮件中请组织者书面确认：本项目应采用哪一版许可，以及论文公开仓库是否允许发布 checkpoint、真实图像可视化和逐样本结果。
4. 收到组织者批准和正式训练归档后，将原始文件不改名、不手工解压地放入 Git 忽略目录 `data/phase2a/inbox/`，再执行大小、官方校验值、容器完整性、安全解压、全量解码和去重审计。

用户不需要向 Codex 提供邮箱密码、Kaggle 密钥或任何访问令牌。正式批准、正式文件和明确的公开边界到位前，真实半监督实验不启动。

收到完整获批归档后，使用下面的入口执行安全解压、全量解码、字节/像素去重和封存指纹交集；尖括号内容替换为本地 Git 忽略路径，逐文件结果只能写入 `runs/phase2a`：

```powershell
.\.venv\Scripts\python.exe scripts/audit_phase2a_unlabeled.py `
  --candidate-dir data/phase2a/candidates/approved-unlabeled `
  --archive data/phase2a/inbox/<approved-archive.zip> `
  --expected-size <organizer-byte-size> `
  --expected-md5 <organizer-md5> `
  --extract `
  --sealed-fingerprint-csv reports/phase0/generated/file_inventory.csv `
  --private-manifest runs/phase2a/approved_private_manifest.json `
  --public-aggregate reports/phase2a/approved_data_aggregate.json `
  --source-id <approved-source-version>
```

脚本没有 raw testing 路径参数；testing 只接受既有封存指纹 CSV。
