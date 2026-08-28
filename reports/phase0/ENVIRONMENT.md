# Phase 0 环境快照

审计日期：2026-08-28

本文件只记录可复现实验需要的环境信息，不保存本机用户名、绝对路径或远程凭据。

## 仓库状态

| 项目 | 状态 |
|---|---|
| Git 分支 | `feature/phase0-data-baseline` |
| HEAD | 尚无初始 commit |
| 远程仓库 | 尚未绑定 |
| 审计时工作树 | `src/`、`tests/` 为未跟踪内容 |

这意味着当前实现仍是工作副本。公开前应先确认文件范围、补齐忽略规则，再创建有说明的初始提交。

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
- tiny-overfit、监督训练和冻结 checkpoint 评估脚本已经存在；tiny-overfit 已有真实运行产物。

## 当前缺项

- 尚无由实际运行保存的配置快照和环境快照；
- tiny-overfit 已通过；尚无正式 baseline checkpoint；
- 尚无完整无标签数据；
- 尚无初始 Git commit。

这些缺项不能用旧项目中的环境、checkpoint 或结果静默代替。
