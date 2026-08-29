# 获准完整归档接收与验收清单

只有“使用权限明确”和“文件完整”同时满足，才执行既有审计入口；邮件已发出或下载完成都不能代替验收。

## A. 授权与范围

- [ ] 课题组或组织者书面确认本毕业设计属于允许的研究用途。
- [ ] 确认适用协议的版本/日期、正确签署人及授权覆盖的学生和课题。
- [ ] 由真实负责人本人签署；不由工具代签，不提交未经确认的冲突协议。
- [ ] 分别确认能否公开：原图、标签、少量可视化、训练 checkpoint、聚合指标、逐样本预测。
- [ ] 授权证明和填好身份的协议只存受保护位置，不提交 Git。

## B. 文件身份与接收

- [ ] 记录发放方、下载/传输页面、联系人、取得日期和获批渠道。
- [ ] 记录正式文件名、版本、发布日期、精确字节大小和官方 MD5/SHA-256。
- [ ] 记录官方文件清单及 31,421、31,121、2,045 Example 的口径说明。
- [ ] 原始归档不改名放入 `data/phase2a/inbox/`；该目录保持 Git 忽略。
- [ ] 不覆盖当前坏包；新文件和既有文件分别保留身份与校验记录。
- [ ] 下载若中断，只在 URL、版本、字节身份和 Range 语义一致时续传。

## C. 完整性和安全解压

- [ ] 本地实际字节数与官方值一致。
- [ ] 本地实际校验值与官方值一致。
- [ ] ZIP/TAR 中央目录或容器尾部可正常读取，完整测试返回成功。
- [ ] 先生成允许成员计划，拒绝绝对路径、`..`、路径碰撞、符号链接和覆盖现有文件。
- [ ] 只解压获准的训练/无标签子树，不把 validation/testing/Example 的未知集合混入。
- [ ] 逐张完整解码并记录格式、尺寸、通道和异常；损坏图不进入候选池。

## D. 私有清单、去重与重叠

- [ ] 私有 manifest 记录真实路径、字节 SHA-256、规范化 RGB+尺寸像素 SHA-256 和排除原因。
- [ ] 先做精确字节去重，再做规范化像素去重；感知近似只标风险，不自动删除。
- [ ] 明确识别 Example 子集，不擅自与主池相加，也不因名称自动认定独立。
- [ ] 与封存 train 300、validation 100 指纹求交并排除重叠。
- [ ] 与封存 testing 501 指纹只做集合交集；不读取测试图或标签，不做测试推理，不按测试表现选样本。
- [ ] 若缺可靠患者/病例/视频元信息，这三个层级继续写“未知”，不能由文件名或无精确重复推导独立性。

## E. 数量链和最终放行

- [ ] 分别记录：归档发现成员 → 完整下载 → 成功解压 → 可解码 → 字节去重后 → 像素去重后 → 排除跨分区重叠后 → 最终候选。
- [ ] 最终状态必须是有证据的 `READY_FULL` 或明确身份的 `READY_NAMED_SUBSET`；否则保持对应阻塞状态。
- [ ] 公开报告只有许可允许的聚合数字；医疗图像、逐文件清单、路径、指纹和签名仍为私有。
- [ ] 导师确认数据版本与监督口径后，才制定下一轮真实半监督实验，不沿用接口冒烟值作为效果基线。

## 既有审计入口

在填入组织者提供的真实大小、校验值和版本后运行：

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

不要把占位符替换为猜测值；测试分区仍只允许使用既有封存指纹。
