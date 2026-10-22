# GeoEqui-LD

产时超声三关键点定位与产程进展角（AoP）研究。

## 当前阶段实验结果

**结果截至2026-10-20；报告与源码导航整理于2026-10-22。** 本阶段完成五种结构变体、同源S/SG/SP/SGP、监督训练时长及FH目标宽度的首轮对照。STLD与PIPNet-GSSL各两个seed作为外部参照，另保留短预算R18初始化配对结果。

**阅读入口：** [两页主线报告](reports/progress_20261021/REPORT.pdf) · [研究问题与落实情况](reports/progress_20261021/REQUEST_ALIGNMENT.md) · [完整五指标与逐seed结果](reports/progress_20261021/RESULTS.md) · [实验协议](reports/progress_20261021/PROTOCOLS.md) · [源码阅读导航](docs/CODE_MAP.md)

当前两seed对照中，SGP相对匹配S的ALL均降低。完整结构在监督预热阶段的ALL优势未延续至半监督终点；延长监督训练未消除训练与验证误差差距；FH目标加宽的收益伴随其他点位误差增加。现有结果支持条件性结论，尚不能确定各模块的独立贡献或基础泛化问题的原因。

**后续工作（拟开展，尚未执行）：** 编码器初始化拟作为训练与验证误差差距的待检验因素。在核对同架构预训练权重、输入通道映射及BN适配后，拟开展其他条件一致的纯监督配对。现有R18配对结果不能直接解释HRNet的表现。

## 本轮与旧主表的关系

为公平比较结构，本轮从共同未训练起点分别训练，并使用Validation100开发集。它不是旧e39状态的同口径复测。原[2026-10-06 Testing501主表](reports/model_comparison_20261006/RESULTS.md)与[原报告](reports/model_comparison_20261006/REPORT.pdf)保持原样，不将旧15.390 px与新开发结果直接相减。

两个seed描述训练随机性，样本SD不是置信区间。位置单位为原图像素，角度为度；外部方法为资源适配，不作严格等资源或临床可用性声明。完整缺项见[局限与后续](reports/progress_20261021/LIMITATIONS.md)。

## 代码与公开范围

当前公开源码主要是原strict核心、历史脚本和合成测试。**完整V2、当前消融与外部适配的生产训练入口未全部公开，本仓不是当前全部数字的一键复现包。** 已公开源码的模块与版本对应关系见[源码阅读导航](docs/CODE_MAP.md)；原strict、早期U-Net训练脚本与本轮实验入口分列说明。

医学图像、逐例标签/预测、病例标识、权重、真实采样计划、私有执行记录与通信材料不在公开范围。原版与V2修订、第三方来源分列，见[来源与发布范围](docs/ATTRIBUTION_AND_RELEASE_SCOPE.md)。

[查阅导航](docs/ADVISOR_REVIEW_INDEX.md) · [历史](docs/HISTORY.md) · [方法引用](reports/progress_20261021/REFERENCES.md) · [LICENSE](LICENSE)
