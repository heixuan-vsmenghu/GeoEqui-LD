# Phase 2A 几何一致性接口

## 这轮实现的定义

坐标顺序固定为 `[PS1, PS2, FH1]`，每个点为 `[x, y]`。已有变换矩阵 `T` 的方向不变：`T` 把原图归一化坐标映射到视图归一化坐标，图像重采样时在内部使用逆映射。

对同一张图的两个视图，模型分别预测 `p1` 和 `p2`。先还原到共同原图坐标系：

```text
q1 = inverse(T1) · p1
q2 = inverse(T2) · p2
```

只在两个视图都显式标为有效的对应点上计算：

```text
L_coord = mean(||q1 - q2||_2)
```

`L_coord` 的单位是原图 `[-1, 1]` 归一化坐标距离。AoP 不在非方形归一化平面上直接计算；`q1/q2` 先按 `align_corners=True` 转回原图像素，再以 PS1 为顶点、PS1→PS2 和 PS1→FH1 为两条射线计算无符号夹角：

```text
L_angle = mean(|AoP(q1) - AoP(q2)|)  # degree
L_geo   = L_angle + 0.1 * L_coord
```

这里的 `0.1` 只是附件指定的接口验收系数。角度项是度，坐标项是归一化距离，两项数值不能被解释成已确定的正式训练权重，也不能与导师原文中的总损失权重直接画等号。

## 变换与单位

- 允许有限、可逆、保持方向的二维相似变换：均匀缩放、旋转和平移。
- 拒绝非均匀缩放、切变、反射、奇异矩阵和投影变换。
- 平移参数使用 `[-1, 1]` 网格单位。`align_corners=True` 时，`tx=0.1` 对应 `0.1 × (W-1) / 2 = 0.05 × (W-1)` 像素；在宽 512 时为 25.55 像素，不是图宽的 10%。
- 当前 H3 接线检查使用两组显式矩阵：`scale=0.95, t=(0.04,-0.03)` 与 `scale=1.05, t=(-0.03,0.02)`，没有靠随机范围掩盖单位歧义。

## 可见性和退化结构

接口强制调用方传入两个布尔可见性 mask，不从“预测落在图内”推导真实可见性。合成或有标签诊断可以用已知点与变换计算可见性；真实无标签图仍需另定遮挡/裁切策略。

H3 单图检查使用的是 `synthetic_all_visible_diagnostic_mask`，只为验证梯度通路，不声称三个结构在该图中真实可见。三点重合、任一射线长度接近零、非有限预测或缺少完整可见结构时不把占位角度计入损失；没有有效角度会返回与两条计算图相连的零角度项，同时记录 `no_valid_geometry`。

两套预测即使一致但同时定位错误，也可能得到很小的一致性损失。因此监督锚点不能删除，一致性下降本身不是定位准确率提升。

## 已覆盖的测试

`tests/test_geometry_consistency.py` 覆盖：

1. 已知点经两组变换后可逆恢复，角度项和坐标项接近零。
2. 人为移动点后损失按预期变化，两个预测分支都有有限梯度。
3. 均匀相似变换保持角度，非均匀缩放被拒绝。
4. 证明两个视图不能裸比较坐标。
5. 合成亮点与点同步变换、边界和裁切可见性。
6. 零长度射线被标为无效，不把零损失当成成功。
7. “一致但都错误”仍可能低损失。
8. batch 大于 1 时矩阵不串样本。
9. 两个视角均能反向传播。
10. `lambda_geo=0` 时，隔离状态影响后的旧监督目标数值与梯度逐元素保持一致。
11. 非方形图先转像素再算角度，以及 `tx=0.1` 的像素换算。

## H3 最小接线检查

脚本 `scripts/check_phase2a_h3_geometry.py` 从本地 `train.image_dir` 读取一张图像，不打开标签 CSV；加载冻结的 Phase 1C epoch 14 H3 best，模型设为 eval 但保留 autograd。它不创建 optimizer、不调用 step、不写 checkpoint。

锁定环境为 `torch 2.5.1+cpu`，因此实际复核使用 CPU，墙钟 16.79 秒，GPU 预算使用 0 分钟。一次 backward 后，有限非零梯度到达 backbone、PS/FH enhancer 与两个 decoder；best checkpoint 的字节数和 SHA-256 在检查前后相同。该次接口数值为 `L_angle=8.0956°`、`L_coord=0.0241405`、`L_geo=8.0980`，仅证明真实 H3 计算图已接通，不是模型效果指标。

Backbone 的 915 个可训练参数张量中，876 个得到有限非零梯度，39 个为 `grad=None`。复查表明这 39 个全部位于最终 `stage4.2.fuse_layers` 中为低分辨率 branch 1/2/3 专门生成输出的路径（分别 9/12/18 个）；当前锁定 feature contract 只取最终高分辨率 branch 0（32 通道、reduction 4），所以这些未被选中的目标输出不进入 enhancer、decoder 或损失。参与 branch 0 的主干和 `fuse_layers.0` 均有梯度。这是既有 selected-feature 路径的预期未使用参数，不是 PS/FH 断路，本轮没有为消除该计数而改 H3 结构。

逐项 checkpoint 指纹、梯度范数和本地输入证据只保存在被 Git 忽略的 `runs/phase2a/h3_geometry_gradient_check_timed.json`。

## 复现命令

合成/单元测试不需要真实医学图像：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_geometry_consistency.py tests/test_phase2a_h3_geometry_check.py -q
```

本地具备冻结 H3 和 Phase 0.5 train 路径时，可执行一次只读接线检查；输出必须放在被 Git 忽略的 `runs/phase2a`，且脚本拒绝覆盖旧证据：

```powershell
.\.venv\Scripts\python.exe scripts/check_phase2a_h3_geometry.py `
  --sample-count 1 `
  --device cpu `
  --output-json runs/phase2a/h3_geometry_gradient_check_new.json
```

这条命令只做一次前向和反向，不包含训练循环。

## 进入正式训练前仍需确认

- 无标签真值不可用时的可见性/裁切策略。
- `L_geo` 加入 B0/B1/B2 哪个监督口径，以及外层 `lambda_geo`。
- 导师原文的平移百分比究竟指网格单位还是像素/图宽百分比。
- 是否保留角度差的“度”单位，或在总损失中做尺度归一化。

以上问题没有在 Phase 2A 中通过调参或新增损失静默决定。
