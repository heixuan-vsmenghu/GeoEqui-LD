# Phase 2A 小结

本轮从 Phase 1C 提交 `2be9d5908ce45ed9afc610908ef27620aa958fb4` 建立 `feature/phase2a-data-geometry`，并用 `phase1c-h3-v0.1.0` 标记冻结起点。H3 的结构、配置、best/last 和既有报告没有被改写，也没有追加训练。

数据工作已经从“目录缺失”推进到真实来源和阻塞证据。Zenodo 记录 17355570 的 `Dataset.zip` 已完整下载 1,071,644,672 字节，实际大小和发布页 MD5 均吻合，下载用时约 6 分 35 秒且没有重试。但该发布文件缺少可正常读取的 ZIP 中央目录；Python/Windows ZIP 检查失败，流式读取也以非零状态结束，只观察到至少 30,407 个无标签图像成员，不能据此补成官方 31,421 张完整池。因此没有解压图像、没有把残缺内容送入训练，数据完整性状态为 `BLOCKED_INTEGRITY`。

访问条件也单独核对过。Codabench 正式流程要求用真实身份申请，由团队导师/负责人签数据使用协议，组织者批准后发送训练数据。现有包内协议还同时出现 CC BY-NC-ND 与 CC BY-NC-SA，和 Zenodo/Kaggle 页面元数据的 CC BY、CC BY-NC 也不一致。在组织者澄清和导师签署前，不能只挑最宽松的一条作为依据；这部分状态为 `BLOCKED_ACCESS`。当前最终可训练无标签候选仍是 0，不是“下载了 1 GB 就等于有 1 GB 可训练数据”。

几何工作线已独立完成。新增接口把两个视图预测分别逆变换到共同原图坐标，计算像素空间 AoP 的度数差和归一化坐标距离，并返回有效点数、有效角度数与跳过原因。合成测试覆盖单位、可逆性、非方形图、裁切、退化、batch 隔离、双路径梯度和 `lambda_geo=0`。随后用冻结 H3 和一张 train 图像做了 eval+autograd 的前后向；没有打开标签、没有 optimizer 或 step，有限非零梯度到达主干、PS/FH 增强器和两个独立解码器，checkpoint 前后字节一致。

H3 接线中的几何损失值只用于证明接口和梯度路径可运行。它不使用新增无标签池，不是半监督训练，也没有产生新的 validation/testing 成绩。本轮没有实现 EMA、伪标签、置信度筛选或完整半监督损失。

详细数据证据见 [DATA_INTAKE.md](DATA_INTAKE.md)，数学和测试口径见 [GEOMETRY_CONTRACT.md](GEOMETRY_CONTRACT.md)，脱敏机器结果见 [aggregate_results.json](aggregate_results.json)。通信草稿已在导师查阅分支中移出技术入口并保存在本地 Git 忽略目录；历史提交仍保留原记录，也没有自动发送。
