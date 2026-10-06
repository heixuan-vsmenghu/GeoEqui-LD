# GeoEqui-LD

产时超声三关键点定位与产程进展角（AoP）测量。

## 当前阶段：模型实现与基线对比

更新于2026-10-06。本次先交付完整模型与对照方法的性能表，再据结果确定优化重点；没有新增训练或评价。

**阅读入口：**[五指标主表](reports/model_comparison_20261006/RESULTS.md) · [两页PDF](reports/model_comparison_20261006/REPORT.pdf) · [实现版本与方法来源](reports/model_comparison_20261006/METHODS.md)

| 检查项目 | 当前状态 |
|---|---|
| 本模型 | HRNet、PS/FH增强、独立解码器及完整S/G/P、双门与EMA流程已运行；当前为明确标记的V2修订版 |
| 主要对照 | T10、Mean Teacher、FixMatch、MoCo v2 + FT：本机训练或任务适配，范围逐项说明 |
| 补充参考 | Noisy Student作者终态、TransUNet + USFM初始化适配；不混称本机完整复现 |
| 评价 | 同一Testing501，五指标MRE_PSR、MRE_PSL、MRE_FHT、MRE_ALL与ΔAoP |
| 后续 | 先依据当前性能指导优化；本次文档更新不追加模型或训练 |

## 指标

三个点的平均位置误差为MRE_PSR / MRE_PSL / MRE_FHT，MRE_ALL为三点等权平均，单位原图像素；ΔAoP为角度平均绝对误差，单位度。五项均越低越好，退化角度规则与有效数量在主表旁注明。

当前Geo V2完整流程MRE_ALL为**15.390像素**、ΔAoP为**6.955度**。同桥接、同后续预算的监督参考为17.698像素、7.420度。T1作者双学生参考13.157像素仍低于Geo；不同来源、预算与读出不构成完全等资源排名。

## 实现与公开范围

本地已运行的结果与公开代码范围分开。[模型核心](src/geoequi_ld/models/README.md)、[strict模型](src/geoequi_ld/models/strict.py)、[原MSE损失](src/geoequi_ld/training/strict_losses.py)保持原样。文档解释V2，不把原代码改名为V2；完整V2训练入口、权重及病例材料未由本次更新公开，不承诺一键复现全部实验。

## 历史与来源

[初版失败表](reports/model_comparison_20261006/HISTORY_V1.md) · [历史导航](docs/HISTORY.md) · [来源与发布范围](docs/ATTRIBUTION_AND_RELEASE_SCOPE.md) · [LICENSE](LICENSE)

早期探索和后续补证不与当前主表混排，旧报告、代码和负结果保留。导航整理不是删除历史；研究设计、第三方实现、医学数据和权重的权利边界按原来源说明处理。
