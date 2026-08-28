# Phase 0 数据需求与阻塞项

状态：**完整无标签数据缺失，半监督阶段阻塞。**

当前监督数据足以继续核心模块测试、tiny-overfit 和最小监督 baseline，但不足以实现或验证 GeoEqui-LD 的无标签几何一致性、EMA 教师和伪标签机制。

## 1. 需要补齐的数据

数据应放在公开 Git 仓库之外，并保持官方目录结构。运行时通过配置传入 `<data_root>`：

```text
<data_root>/
└── Dataset/
    ├── Training/
    │   ├── Labeled cases/
    │   │   ├── label.csv
    │   │   └── 300 images
    │   └── Unlabeled cases/
    │       ├── official nested unlabeled images
    │       └── standard-plane Example images
    ├── Validation/
    │   ├── landmarks_data.csv
    │   ├── aop_results.csv
    │   └── 100 images
    └── Testing/
        ├── landmarks_data.csv
        ├── aop_results.csv
        └── 501 images
```

预期公开数量：

```text
Labeled train: 300
Validation:    100
Testing:       501
Unlabeled:     31,421（当前 Kaggle v1 包结构预期）
Examples:      2,045
```

`Examples` 是否与 31,421 张无标签图存在内容重复，必须在完整下载后通过 SHA-256 确认；不能直接把二者相加当成独立训练样本。

部分方法材料和旧指令使用 31,121，可能来自“training total 31,421 减去 300 labeled”的口径。完整数据到位时应同时核对数据版本、中央目录和数据方说明；在确认前将其记录为版本差异，而不是擅自选择一个数字覆盖另一个。

## 2. 需要数据方提供的分组字段

现有标签没有患者或视频分组信息。为判断 split 泄漏和设计稳健验证，需要至少一种不可逆匿名标识：

```text
patient_id_hash
case_id_hash
video_id_hash
sequence_id_hash
frame_index
source_site_id
```

不需要姓名、病历号、日期等直接身份信息。若数据方不能提供完整映射，至少需要说明文件名两段数字各自代表什么，以及相邻编号是否可能来自同一视频或患者。

## 3. 下载后验收

完整数据到位后，在启动任何半监督训练前应完成：

1. 记录来源、版本、归档文件大小和校验值；
2. 核对目录和文件数量；
3. 全量解码，统计真实编码、尺寸和通道；
4. 检查空文件、损坏图、重复路径和不支持格式；
5. 对所有图像计算文件 SHA-256 和解码像素 SHA-256；
6. 检查 Example 图与主无标签池是否重复；
7. 检查无标签池与 train/validation/testing 是否内容重合；
8. 如有分组字段，检查患者、病例和视频是否跨 split；
9. 只输出脱敏聚合报告，不在公开仓库保存文件清单或绝对路径；
10. 在配置中冻结最终纳入和排除规则。

## 4. 许可前置条件

在下载、公开仓库发布或共享衍生产物前，需要确认：

- 适用的是哪一版数据许可；
- 是否已完成负责人或导师签署；
- 是否允许公开聚合统计、checkpoint 和预测结果；
- 是否允许发布真实图像可视化；
- 数据应该由谁保管、谁可以访问。
- 当前版本的无标签图数量应按 31,421 还是 31,121 解释。

未确认前，仓库只发布代码、合成测试和聚合说明。

## 5. 当前允许和禁止的工作

允许：

- 完成 Phase 0 核心单元测试；
- 使用合成点和合成图验证坐标、热图与变换；
- 使用 300 张有标签训练图进行 tiny-overfit；
- 仅使用官方 validation 选模型；
- 冻结方案后再运行一次 testing 评估。

暂不允许：

- 用两张孤立样例冒充完整无标签池；
- 伪造 31,421 张无标签数据统计；
- 提前宣称完成半监督训练；
- 从 testing 调参或选择 checkpoint；
- 将真实数据、衍生图像或逐样本预测提交到公开 Git。
