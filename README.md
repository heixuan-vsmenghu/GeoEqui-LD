# GeoEqui-LD

产时超声三关键点定位与产程进展角（AoP）测量。

## 当前阶段：模型实现与基线对比

更新于2026-10-06。已完成模型实现、半监督训练流程和基线对比，下面汇总当前结果与仍需解决的问题。本次整理使用已有结果，没有新增训练或评价。

**阅读入口：**[五指标主表](reports/model_comparison_20261006/RESULTS.md) · [两页PDF](reports/model_comparison_20261006/REPORT.pdf) · [实现版本与方法来源](reports/model_comparison_20261006/METHODS.md)

| 工作内容 | 当前状态 |
|---|---|
| 本模型 | HRNet、PS/FH增强、独立解码器及完整S/G/P、双门与EMA流程已运行；当前为明确标记的V2修订版 |
| 主要对照 | T10、Mean Teacher、FixMatch、MoCo v2 + FT：本机训练或任务适配，范围逐项说明 |
| 补充参考 | Noisy Student作者终态、TransUNet + USFM初始化适配；不混称本机完整复现 |
| 评价 | 同一Testing501，五指标MRE_PSR、MRE_PSL、MRE_FHT、MRE_ALL与ΔAoP |
| 当前问题 | FH困难样本定位和角度稳定性仍有改进空间；部分基线采用本地任务适配 |

## 指标

三个点的平均位置误差为MRE_PSR / MRE_PSL / MRE_FHT，MRE_ALL为三点等权平均，单位原图像素；ΔAoP为角度平均绝对误差，单位度。五项均越低越好，退化角度规则与有效数量在主表旁注明。

当前Geo V2完整流程MRE_ALL为**15.390像素**、ΔAoP为**6.955度**。同桥接、同后续预算的监督参考为17.698像素、7.420度。T1作者双学生参考13.157像素仍低于Geo；不同来源、预算与读出不构成完全等资源排名。

## 实现与公开范围

本地实验采用V2修订流程，公开仓中的[模型核心](src/geoequi_ld/models/README.md)、[strict模型](src/geoequi_ld/models/strict.py)、[原MSE损失](src/geoequi_ld/training/strict_losses.py)仍为原实现。V2的调整在文档中单独说明；完整V2训练入口、权重及病例材料尚未公开，目前公开代码不能直接复现全部V2实验。

## 历史与来源

[初版失败表](reports/model_comparison_20261006/HISTORY_V1.md) · [历史导航](docs/HISTORY.md) · [来源与发布范围](docs/ATTRIBUTION_AND_RELEASE_SCOPE.md) · [LICENSE](LICENSE)

早期探索和后续补证另行记录，旧报告、代码和负结果保留。研究设计、第三方实现、医学数据和权重的权利边界见原来源说明。
