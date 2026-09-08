# 配置与运行版本

## 当前研究入口

当前结果见[监督训练结果摘录](../reports/review_20260908/SUPERVISED_RESULTS.md)和[热图与 DSNT 核对](../reports/review_20260908/HEATMAP_DSNT_CHECK.md)，概况见[当前进展](../docs/CURRENT_PROGRESS.md)。

strict Stage 1 已完成 200 轮纯 MSE 监督预热，主结果采用物理 batch=2、lr=0.0001、weight_decay=0.0001、clip=5、sigma=4、无 scheduler。Stage 2 当前实现的短试跑从 epoch 39 开始，实际在第 2 轮停止。[当前模型和损失核心](../src/geoequi_ld/models/README.md)已收录，完整训练运行系统和原报告仍在本地，不能用本目录旧 YAML 代替当前运行设置。

## 设置来自哪里

| 运行 | 实际配置来源与版本 | 查阅位置 |
|---|---|---|
| strict Stage 1 的 200 轮路线 | 本地 `scripts/run_strict_stage1_200e.py` 及其复用的 batch2/reference 工厂、源 checkpoint 状态和已保存运行记录 | [当前结果](../docs/CURRENT_PROGRESS.md)、本地 `reports/strict_model_stage1/STRICT_STAGE1_200E.md` |
| strict Stage 2 当前实现的短试跑 | 本地 `scripts/run_strict_stage2_original_5e.py` 的运行协议与恢复状态；不是 phase0/05/06 YAML | [热图与 DSNT 核对](../reports/review_20260908/HEATMAP_DSNT_CHECK.md)、本地 `reports/strict_model_stage2/STRICT_STAGE2_ORIGINAL_5E.md` |
| 官方 T10 复现 | 官方 `0oTyTo0/IUGC2025` 的 `heatmap_train_only.py`，固定提交 `bc8fce2032c000c2569e916268ab918c0905ab4e` | [已有固定版本复现报告](https://github.com/heixuan-vsmenghu/GeoEqui-LD/blob/e3af9e2a2c29e5de3beab912fe1ecf7592300f8b/reports/baseline_reproduction/BASELINE_REPRODUCTION.md) |

本地另有早期 `configs/strict_model_pilot.yaml`，但它不是 200 轮续训或后续 Stage 2 实际设置的替代品。上述后续入口在脚本及保存的运行记录中明确参数；这里没有新造 YAML 并声称历史运行使用过它。

## 历史配置追溯

本目录现有 `phase0_*.yaml`、`phase05_*.yaml`、`phase06_*.yaml` 属于早期自定义监督基线、辅助损失消融和长预算探索，既不是官方 T10 配置，也不是当前 strict Stage 1/2 配置。其中 `dsnt_temperature=0.05` 等字段保留原样；当前 strict 的普通 Softmax 读出另见[版本对应关系](../docs/CURRENT_PROGRESS.md#读出版本与实际调用)。

截至本次核对，旧 YAML 在远端的最近相关提交为 `df9e6d9f472266aeb1e4a7f27e34fc6a5aee7b4b`（2026-08-28）。旧日期表示内容没有在本次改动，不表示当前研究仍处于旧阶段。本次不移动、不删除或改写旧 YAML；早期结果见[历史索引](../docs/history/RESEARCH_INDEX.md)。
