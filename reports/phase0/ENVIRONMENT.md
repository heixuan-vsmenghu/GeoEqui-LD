# Phase 0 环境快照

审计日期：2026-08-28

本文件只记录可复现实验需要的环境信息，不保存本机用户名、绝对路径或远程凭据。

## 仓库状态

| 项目 | 状态 |
|---|---|
| Git 分支 | `main` |
| 训练代码 HEAD | `d421cab667319070a44d5155c6abc0153925d6b3` |
| 远程仓库 | `https://github.com/heixuan-vsmenghu/GeoEqui-LD`（Public） |
| 监督训练产物 | 本地 Git 忽略目录，未公开 checkpoint/图像 |

代码与 Phase 0 脱敏报告已经推送到公开仓库。真实数据、checkpoint、运行日志和预测图未上传。

## 系统与硬件

| 项目 | 值 |
|---|---|
| 操作系统 | Windows 10 x64 |
| GPU | NVIDIA GeForce GTX 1650 |
| 显存 | 4096 MiB |
| NVIDIA 驱动 | 546.33 |

该显卡适合单元测试和小型监督模型验证，不适合在未经资源评估的情况下直接启动大批量 512×512 半监督训练。

## 已验证 Python 环境

| 依赖 | 版本 |
|---|---|
| Python | 3.11.4 |
| PyTorch | 2.5.1+cu121 |
| torchvision | 0.20.1+cu121 |
| CUDA runtime | 12.1 |
| NumPy | 1.26.4 |
| Pandas | 2.2.3 |
| Pillow | 11.1.0 |
| PyYAML | 6.0.3 |
| pytest | 8.4.2 |

PyTorch 检测到 CUDA，GPU 可用。

系统默认的另一个 Python 3.14 环境没有安装 PyTorch，不能作为当前项目运行环境。正式训练前应创建项目专用 Python 3.11 虚拟环境并锁定依赖，避免误用默认解释器。

本机从当前中文项目路径执行 `pip install -e .` 时，wheel 可以成功构建，但 setuptools 生成的 editable `.pth` 含 UTF-8 路径，而该 Python 启动时按 GBK 读取，导致 `UnicodeDecodeError`。本次安装已完整撤回，解释器恢复正常。当前使用 `PYTHONPATH=src` 或脚本自带的 `src` 路径启动；若需要 editable 安装，应将仓库克隆到纯 ASCII 路径后再执行。

非 editable 的 `pip install .` 已在相同中文路径下完成构建、安装、`import geoequi_ld` 和卸载闭环，wheel 包安装不受上述 `.pth` 编码问题影响。

## 测试命令与结果

使用不写 pytest 缓存的方式执行：

```powershell
$env:PYTHONPATH = "src"
$env:PYTHONDONTWRITEBYTECODE = "1"
python -m pytest -q -p no:cacheprovider
```

最新完整测试结果：

```text
33 passed
```

覆盖范围包括坐标转换、高斯热图、DSNT、AoP、相似变换、指标、U-Net 输出形状、监督训练循环、原图像素评估和 checkpoint 往返。

## 依赖与配置状态

- 已创建 `pyproject.toml` 和 `requirements.txt`，声明 Python `>=3.10,<3.13` 及 Phase 0 依赖；
- 已创建公开的 `configs/phase0_baseline.yaml` 和本地配置示例；
- PyTorch 的 CPU/CUDA wheel 仍需在目标机器上显式选择；
- 审计环境与依赖文件均固定使用 PyYAML 6.0.3；
- tiny-overfit、监督训练和冻结 checkpoint 评估脚本均已有真实运行产物；

## 当前缺项

- 尚无由实际运行保存的配置快照和环境快照；
- tiny-overfit 与 20 轮正式 baseline 均已完成，本地有 best/last checkpoint；
- 尚无完整无标签数据；
- 尚无初始 Git commit。

checkpoint 与图像诊断因数据协议限制留在 Git 忽略目录，不能用“公开仓库里没有权重”误解为训练未执行。
