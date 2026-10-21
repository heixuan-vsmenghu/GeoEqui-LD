# GeoEqui-LD

产时超声三关键点定位与产程进展角（AoP）研究。

## 当前阶段：先查清Geo的模块与训练问题

**结果截至2026-10-20；主线整理于2026-10-21。** 本阶段围绕五种粗结构、同源S/SG/SP/SGP、基础长监督及FH目标宽度形成首轮对照。外部STLD和PIPNet-GSSL各两个seed用于补充参照，R18初始化配对提供下一步线索。

**阅读入口：** [两页主线报告](reports/progress_20261021/REPORT.pdf) · [研究问题与落实情况](reports/progress_20261021/REQUEST_ALIGNMENT.md) · [完整五指标与逐seed结果](reports/progress_20261021/RESULTS.md) · [实验协议](reports/progress_20261021/PROTOCOLS.md)

本轮首先回答：结构优势能否保留到半监督末轮；G/P在什么背景下有用；延长监督是否解决训练/验证分化；FH加宽的收益是否伴随其他点位代价。现有结果支持部分条件性收益，但没有证明所有模块有效，也没有解决基础泛化。

**下一优先建议（尚未执行）：** 在Geo内部核查编码器初始化因素，先验证同架构权重、输入通道及BN适配，再固定其余条件做纯监督配对。不是把R18的结果直接外推到HRNet。

## 本轮与旧主表的关系

为公平比较结构，本轮从共同未训练起点分别训练，并使用Validation100开发集。它不是旧e39状态的同口径复测。原[2026-10-06 Testing501主表](reports/model_comparison_20261006/RESULTS.md)与[原报告](reports/model_comparison_20261006/REPORT.pdf)保持原样，不将旧15.390 px与新开发结果直接相减。

两个seed描述训练随机性，样本SD不是置信区间。位置单位为原图像素，角度为度；外部方法为资源适配，不作严格等资源或临床可用性声明。完整缺项见[局限与后续](reports/progress_20261021/LIMITATIONS.md)。

## 代码与公开范围

当前公开源码主要是原strict核心、历史脚本和合成测试。**完整V2、当前消融与外部适配的生产训练入口未全部公开，本仓不是当前全部数字的一键复现包。** 本轮只更新聚合、协议和审阅入口，不以文档更新冒充代码实现更新。

医学图像、逐例标签/预测、病例标识、权重、真实采样计划、私有执行记录与通信材料不在公开范围。原版与V2修订、第三方来源分列，见[来源与发布范围](docs/ATTRIBUTION_AND_RELEASE_SCOPE.md)。

[导师查阅导航](docs/ADVISOR_REVIEW_INDEX.md) · [历史](docs/HISTORY.md) · [方法引用](reports/progress_20261021/REFERENCES.md) · [LICENSE](LICENSE)
