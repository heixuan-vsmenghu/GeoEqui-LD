# IUGC 组织者研究访问与归档问题咨询（未发送、身份空白模板）

核验说明：IUGC 2025 的[官方 Codabench 页面](https://www.codabench.org/competitions/7105/)仍列出 `fugc.isbi25@gmail.com` 作为联系和历史数据申请邮箱；同一邮箱也出现在组织团队的[相关 2026 超声挑战页面](https://www.codabench.org/competitions/11539/)。原比赛时间表已经结束，因此下面只写成 **2026 年毕业设计学术研究咨询**，不声称报名或参赛。当前研究授权流程是否仍沿用 2025 页面，需由收件人确认。归档技术问题可选择抄送 IUGC 页面列出的技术联系邮箱 `tyt6xx@163.com`，研究访问的主收件人仍使用前者。

身份占位符必须由用户按事实填写；不要从聊天记录推断，也不要在公开仓库提交填好的版本。

## English draft

**To:** `fugc.isbi25@gmail.com`

**Optional CC for the archive issue:** `tyt6xx@163.com`

**Subject:** Research access and archive integrity inquiry for the IUGC 2025 dataset

Dear IUGC Organizing Team,

My name is **[FULL NAME]**, an undergraduate student at **[UNIVERSITY]**, **[DEPARTMENT / PROGRAM]**. I am preparing a 2026 graduation thesis on semi-supervised landmark detection in intrapartum ultrasound under the supervision of **[SUPERVISOR / PRINCIPAL INVESTIGATOR]**. This is an academic research-access inquiry; I am not claiming to register for the concluded 2025 competition.

Before using any IUGC data, we would like to confirm the current authorization route and the applicable terms:

1. Does our research group already have an authorization that may cover this student thesis? If not, what is the current application process, which agreement should be used, and who must sign it?
2. What identity, affiliation, research-plan, supervisor, or ethics materials should we submit? We will provide them through the channel you specify.
3. What permissions apply separately to research use and to public release of images, labels, trained checkpoints, visualizations, or per-sample predictions?
4. What is the official interpretation of 31,421, 31,121, and the 2,045 Example subset?

We also encountered an archive-integrity issue with the public Zenodo record 17355570, version v1, accessed on 29 August 2026:

- file: `Dataset.zip`;
- published and locally measured size: 1,071,644,672 bytes;
- published and locally calculated MD5: `8f787f72a839cee4ef5bf017f877609d`;
- Python and Windows ZIP readers could not find a usable end-of-central-directory record;
- a streaming listing ended with exit code 1 after observing at least 30,407 image members in the unlabeled subtree.

The 30,407 figure is only a lower bound observed before the error. We have not treated it as an extracted, decodable, or trainable pool, and we have not redistributed the archive. Could you please confirm whether this file requires additional volumes or a specific extraction method, or whether a corrected package is available?

If access is approved, could you provide the authoritative archive name/version, exact byte size, checksum, and file manifest together with the current usage agreement?

Thank you for your guidance.

Sincerely,<br>
**[FULL NAME]**<br>
**[UNIVERSITY / DEPARTMENT]**<br>
**[INSTITUTIONAL EMAIL]**<br>
Supervisor/PI: **[NAME AND INSTITUTIONAL EMAIL]**

## 中文对照

**收件人：** `fugc.isbi25@gmail.com`

**归档问题可选抄送：** `tyt6xx@163.com`

**主题：** IUGC 2025 数据集研究访问及归档完整性咨询

IUGC 组织团队您好：

我是 **[姓名]**，来自 **[学校] [院系/专业]** 的本科生，目前在 **[导师/课题负责人]** 指导下开展 2026 年毕业设计，研究方向是产时超声关键点检测的半监督方法。这是一封学术研究访问咨询，不代表申请参加已经结束的 2025 比赛。

在使用任何 IUGC 数据前，我们想确认当前有效的授权流程和适用条款：

1. 课题组已有的数据授权是否可以覆盖本次学生毕业设计？若不能，目前应采用什么申请流程、哪一版协议、由谁签署？
2. 需要提交哪些身份、单位、研究计划、导师或伦理材料？我们会通过您指定的渠道提供。
3. 研究使用，以及公开图像、标签、训练权重、可视化或逐样本预测，各自允许到什么范围？
4. 31,421、31,121 与 2,045 张 Example 子集的正式口径分别是什么？

我们还发现 Zenodo 记录 17355570 的 v1 公开归档存在完整性问题。2026 年 8 月 29 日取得的 `Dataset.zip`，发布页和本地实测大小均为 1,071,644,672 字节，发布页和本地计算 MD5 均为 `8f787f72a839cee4ef5bf017f877609d`；但 Python 与 Windows ZIP 工具无法找到可用的中央目录结束记录，流式列举也以退出码 1 结束，异常前只观察到至少 30,407 个无标签图像成员。

30,407 只是异常前观察下界，我们没有把它当作已解压、可解码或可训练数据，也没有重新发布归档。请问该文件是否需要其他分卷或特定解包方式，或者是否可以提供修正后的完整归档？

如果研究访问获批，也烦请同时提供正式归档的名称/版本、精确字节数、校验值、文件清单和现行使用协议。

感谢您的指导。

此致

敬礼

**[姓名]**<br>
**[学校/院系]**<br>
**[学校邮箱]**<br>
导师/课题负责人：**[姓名及学校邮箱]**

> 发送前：先由导师确认课题组没有现成获准路径；仅填写真实身份信息。不要附坏包、医学图像、标签、签名扫描件或凭据，除非组织者通过明确安全渠道要求并由本人确认发送。
