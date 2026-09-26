# 低重叠肝脏配准方法对比可视化设计

## 目标

为以下两个低重叠度数据集分别生成一张论文级方法对比大图：

- `/mnt/data3/yangx/P2P/in_silico_visibility_0.2_0.4/`
- `/mnt/data3/yangx/P2P/in_vitro_visibility_0.2_0.4/`

每个数据集以 Ours 的 RMS-TRE 从低到高选取最好的 3 个样本。每张大图包含 3 行样本和 12 列结果，视觉效果复刻
`/home/yangx/code/new_deform/ai_worker/LiverMatch/P2P_demo.png`：白色背景、蓝色完整源点云、红色部分目标点云、球形点和 Courier 风格标题。最终交付两张大图，并保留可复核的预测、指标和选样记录。

## 方法与显示顺序

每行固定采用以下从左到右顺序：

1. Initial Position
2. Ours
3. GeoTransformer
4. CASTv2
5. DFAT
6. Lepard
7. Lepard+P2P
8. PARENet
9. LiverMatch
10. LiverMatch+P2P
11. Go-ICP
12. Ground Truth

模型 checkpoint、项目目录和 Conda 环境以用户给出的路径为准。Lepard+P2P 与 LiverMatch+P2P 分别复用对应基础网络 checkpoint，只更换后端配准求解器。P2P 的参数保持现有评测协议中的 `K=5`，不得为选中样本单独调参。

## 样本选择

样本选择直接读取现有完整评测结果：

`results/visibility_0.2_0.4_benchmark_strict_v3/ours/`

- in-silico 使用无额外合成噪声的 `in_silico_none.json`；
- in-vitro 使用 `in_vitro.json`；
- 只保留状态正常、RMS-TRE 有限且能从原始数据重新加载的样本；
- 分别按 Ours 的 `rms_tre_mm` 升序选前 3 个；
- 若 RMS-TRE 相同，则先按 `original_index`、再按字符串样本标识升序决定顺序。

脚本必须把候选数、最终样本标识、原始索引、可见度和 Ours RMS-TRE 写入清单。选样完成后固定这 6 个样本，所有方法必须处理完全相同的输入，禁止按方法替换样本。

## 推理与预测缓存

现有完整评测 JSON 对多数方法只保存指标，没有保存估计变换。因此，仅针对选中的 6 个样本重新运行十种方法，并为每个“数据集 / 样本 / 方法”保存：

- 4×4 source-to-target 估计变换；
- RMS-TRE（mm）；
- 推理状态和失败原因；
- checkpoint、环境、命令和必要的协议参数；
- 原始样本标识与数据集索引。

各模型保留自己的 Conda 环境与项目代码，通过统一的样本清单和统一的预测文件格式交换数据。预测缓存使用可恢复执行：已存在且元数据匹配的成功结果直接复用；缺失、损坏或元数据不匹配的结果才重新计算。

Go-ICP 使用现有低重叠评测适配器的相同归一化、采样、trim 和超时设置。任何方法失败时不得静默使用单位矩阵冒充成功结果；缓存中明确标记失败，最终图格显示 `FAILED`，并在清单中记录原因。

## 坐标、变换与指标

统一约定如下：

- 完整点云为 source，部分点云为 target；
- 所有显示变换均为 source-to-target；
- 行向量形式应用变换：`points @ R.T + t`；
- Initial Position 使用单位变换；
- Ground Truth 使用当前 benchmark 数据加载器返回的 `transform`；该变换与正式评测中的刚性真值定义保持一致；
- 标题中的 RMS-TRE 采用当前 benchmark 的体积标记点定义，并以 mm 为单位；
- Initial、十种方法和 Ground Truth 均通过同一个指标函数计算，避免展示指标与表格指标不一致。

每个重新计算的 Ours RMS-TRE 必须与用于选样的完整评测结果核对。允许小量浮点或 GPU 非确定性误差，但超出配置容差时停止出图并报告，而不是继续使用不一致结果。

## 图形设计

每个数据集生成一张 3×12 大图：

- 每行一个选中样本，按 Ours RMS-TRE 从低到高排列；
- 每列标题显示方法名和 `RMS-TRE: xx.xx mm`；
- source 使用 RGB `[0, 150, 255]`，target 使用 `[254, 92, 92]`；
- 白色背景，球形点，不显示坐标轴、网格、图例或色条；
- 字体、标题换行、点大小、材质和留白尽量与 `P2P_demo.png` 一致；
- 同一行的 12 个面板共享相机方向、投影和缩放范围；
- 同一数据集三行使用一致的列宽和面板尺寸；
- 每行左侧添加紧凑样本标识，整图顶部只显示数据集名称，不干扰每格的原始标题风格。

渲染优先复用 LiverMatch 的 PyVista/VTK 球形点实现。为保证服务器无显示环境下可批处理，采用 off-screen 截图并由 Matplotlib 或 Pillow 无损拼接。若 PyVista 的无头渲染不可用，则使用当前仓库的软件透视渲染作为明确记录的后备方案，但颜色、背景、相机和标题布局仍保持一致。

## 输出结构

输出固定放在以下独立目录：

```text
output/visualization/visibility_0.2_0.4_livermatch_style/
├── selection.json
├── predictions/
│   ├── in_silico/<sample>/<method>.npz
│   └── in_vitro/<sample>/<method>.npz
├── panels/
│   ├── in_silico/<sample>/<method>.png
│   └── in_vitro/<sample>/<method>.png
├── in_silico_best3_comparison.png
├── in_silico_best3_comparison.pdf
├── in_vitro_best3_comparison.png
├── in_vitro_best3_comparison.pdf
├── metrics.csv
└── manifest.json
```

`manifest.json` 记录生成时间、代码版本、模型次序、数据路径、样本选择、预测文件、最终图路径、渲染后端和失败状态。`metrics.csv` 每行对应一个样本与方法，至少包含数据集、样本标识、方法、RMS-TRE、状态和预测路径。

## 验证与验收

实施时需要完成以下验证：

1. 对选样排序、并列排序和缺失值过滤编写单元测试；
2. 对 12 列固定次序、变换方向和 RMS-TRE 计算编写单元测试；
3. 对缓存元数据匹配与失败状态处理编写单元测试；
4. 对每种模型至少进行一个选中样本的推理烟雾测试；
5. 检查每个数据集恰好 3 个样本、每个样本恰好 10 个方法结果；
6. 核对 Ours 重算 RMS-TRE 与完整评测记录的误差不超过容差；
7. 检查两张 PNG/PDF 均能打开、尺寸一致、标题无裁切、点云未越界；
8. 人工对照 `P2P_demo.png` 检查白底、配色、球形点、相机和标题风格。

验收结果为两张可直接用于论文排版的大图。即使某个方法推理失败，整项任务也要产出可审计结果；失败格必须明显标注，不能伪造或隐藏。
