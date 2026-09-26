# Low-overlap LiverMatch-style Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从 in-silico 与 in-vitro 低重叠数据中各选出 Ours RMS-TRE 最低的 3 个样本，生成两张 3×12 的 LiverMatch 风格方法对比图。

**Architecture:** 先从已完成的 Ours benchmark 中确定固定样本，再构建只含三个样本的兼容数据根目录，使十种模型沿用各自原生测试入口。将异构预测统一为一个可审计的 NPZ/JSON 缓存格式，最后由独立渲染器加载同一份点云、相机和指标，输出 PNG、PDF、CSV 与 manifest。

**Tech Stack:** Python 3.10、NumPy、PyTorch、PyVista/VTK、Matplotlib、pytest、Bash、各模型既有 Conda 环境。

**Spec:** `docs/superpowers/specs/2026-09-26-low-overlap-livermatch-style-visualization-design.md`

## Global Constraints

- 两个数据集各按 Ours `rms_tre_mm` 升序选 3 个样本；并列时依次按 `original_index` 和字符串样本标识升序。
- in-silico 固定使用无额外噪声的 `in_silico_none.json`，in-vitro 固定使用 `in_vitro.json`。
- P2P 变体固定 `K=5`，不得针对样本调参；Go-ICP 沿用现有 benchmark 参数。
- 12 列固定为 Initial、Ours、GeoTransformer、CASTv2、DFAT、Lepard、Lepard+P2P、PARENet、LiverMatch、LiverMatch+P2P、Go-ICP、Ground Truth。
- source 为蓝色 `[0, 150, 255]`，target 为红色 `[254, 92, 92]`；白底、球形点、Courier 标题、无坐标轴和色条。
- 估计变换一律为 source-to-target，并按 `points @ R.T + t` 应用；标题指标一律为体积标记点 RMS-TRE（mm）。
- 失败不得伪装为成功或静默使用单位矩阵；失败格显示 `FAILED`，manifest 保留原因。
- 不覆盖现有完整 benchmark 结果，也不改动工作区中与本任务无关的未提交文件。

## Review Focus

- 完整结果中出现 NaN、失败记录或重复样本时，筛选器必须过滤无效项并稳定选出恰好 3 项；Task 1 覆盖。
- 子集索引与 `original_indices.npy` 容易错位，构建后必须逐项核对样本名、统计值和原始索引；Task 1 覆盖。
- 不同模型导出 3×4、4×4、批量 N×4×4 等变换格式时，统一缓存必须验证形状、有限值和末行；Task 2 覆盖。
- GPU 重跑导致 Ours RMS-TRE 漂移时，超过 `0.05 mm` 必须中止，而不能生成误导图；Task 3 覆盖。
- 无头 PyVista/VTK 不可用或某方法失败时，仍须使用明确记录的软件后备渲染并生成完整大图；Task 4 覆盖。

---

### Task 1: 稳定选样与三样本兼容数据子集

**Files:**
- Create: `tools/prepare_visibility_best3.py`
- Create: `tests/test_prepare_visibility_best3.py`

**Interfaces:**
- Consumes: Ours 完整评测 JSON、两个原始低重叠数据根目录。
- Produces: `select_best_cases(summary: Mapping, top_k: int = 3) -> list[dict]`、`build_subset(dataset: str, source_root: Path, cases: Sequence[Mapping], output_root: Path) -> dict`，以及 `selection.json` 和两个三样本数据根目录。

- [ ] **Step 1: 编写筛选失败测试**

在 `tests/test_prepare_visibility_best3.py` 中增加 `test_select_best_cases_filters_invalid_and_breaks_ties_stably`，构造含失败、NaN、同分和重复样本的记录，断言只返回 3 个有效且顺序为 `rms_tre_mm/original_index/sample`。

- [ ] **Step 2: 运行筛选测试并确认失败**

Run: `/home/yangx/miniconda3/envs/geo_v2/bin/python -m pytest tests/test_prepare_visibility_best3.py::test_select_best_cases_filters_invalid_and_breaks_ties_stably -v`

Expected: FAIL，原因是 `tools.prepare_visibility_best3` 尚不存在。

- [ ] **Step 3: 实现 `select_best_cases` 与 CLI 选样输出**

实现严格有限值、成功状态、样本唯一性和 `top_k` 数量校验；CLI 固定读取 `ours/in_silico_none.json` 与 `ours/in_vitro.json`，写出包含数据集、样本名、子集索引、原始索引、visibility 和 Ours RMS-TRE 的 `selection.json`。

- [ ] **Step 4: 编写子集映射失败测试**

增加 `test_build_subset_preserves_name_statistics_and_original_index`，用临时 list/stat/original-index 文件断言输出顺序与选样一致，数据目录采用链接且不复制大体积 NPZ；增加缺失样本名时报错的断言。

- [ ] **Step 5: 实现 `build_subset`**

in-silico 写入 `Deform_mesh_npz_test/list.npz`、筛选后的 `stat_svd.npz`、`original_indices.npy` 并链接 `Test`；in-vitro 写入 `rigid_list.npy`、筛选后的 `stat.npz`、`original_indices.npy` 并链接 `Rigid_test_data`。

- [ ] **Step 6: 运行 Task 1 测试**

Run: `/home/yangx/miniconda3/envs/geo_v2/bin/python -m pytest tests/test_prepare_visibility_best3.py -v`

Expected: PASS。

- [ ] **Step 7: 提交 Task 1**

```bash
git add tools/prepare_visibility_best3.py tests/test_prepare_visibility_best3.py
git commit -m "feat: prepare best-three visibility subsets"
```

### Task 2: 统一预测缓存与模型导出适配

**Files:**
- Create: `tools/collect_visibility_predictions.py`
- Modify: `tools/evaluate_livermatch_visibility.py`
- Modify: `tools/evaluate_goicp_visibility.py`
- Create: `tests/test_collect_visibility_predictions.py`
- Modify: `tests/test_goicp_p2p_adapter.py`

**Interfaces:**
- Consumes: 各原生测试入口生成的 JSON、逐样本 NPZ 或批量 NPY 变换，以及 Task 1 的 `selection.json`。
- Produces: `normalize_transform(value: ArrayLike) -> np.ndarray`、`collect_method_predictions(method: str, dataset: str, selection: Sequence[Mapping], artifacts: Mapping, output_root: Path) -> list[dict]`；统一文件为 `predictions/<dataset>/<sample>/<method>.npz`，至少包含 `estimated_transform`、`rms_tre_mm`、`status` 和元数据 JSON。

- [ ] **Step 1: 编写变换归一化与缓存校验失败测试**

增加测试覆盖 3×4、4×4、N×4×4 输入，拒绝 NaN、错误末行、样本数不一致和 checkpoint 元数据不匹配；断言失败记录不产生单位矩阵预测。

- [ ] **Step 2: 运行测试并确认失败**

Run: `/home/yangx/miniconda3/envs/geo_v2/bin/python -m pytest tests/test_collect_visibility_predictions.py -v`

Expected: FAIL，原因是收集模块尚不存在。

- [ ] **Step 3: 实现统一收集器**

按 selection 的样本顺序解析：GeoTransformer/RTOR/DFAT/PARENet 的逐样本 NPZ、CASTv2 的批量 transforms NPY、Lepard 的 JSON 变换，以及 LiverMatch/Go-ICP 的 JSON 变换。统一验证 source-to-target 齐次矩阵后写入每样本缓存和方法汇总。

- [ ] **Step 4: 为 LiverMatch 输出估计变换**

在 `tools/evaluate_livermatch_visibility.py` 的每条记录中加入 `estimated_transform=details["estimated_transform"].tolist()`；失败时保留 `None` 和状态字段，CSV 序列化为 JSON 字符串。

- [ ] **Step 5: 为 Go-ICP 本地入口保存成功变换并保持失败语义**

将 `tools/evaluate_goicp_visibility.py` 从纯转发入口改为薄适配入口：调用现有 Go-ICP 实现时，把成功结果的 4×4 变换写入记录；timeout/error 写 `estimated_transform=None`、保留 solver 状态，不把 benchmark 的 identity 罚值当成可视化预测。

- [ ] **Step 6: 运行 Task 2 测试**

Run: `/home/yangx/miniconda3/envs/geo_v2/bin/python -m pytest tests/test_collect_visibility_predictions.py tests/test_goicp_p2p_adapter.py -v`

Expected: PASS。

- [ ] **Step 7: 提交 Task 2**

```bash
git add tools/collect_visibility_predictions.py tools/evaluate_livermatch_visibility.py tools/evaluate_goicp_visibility.py tests/test_collect_visibility_predictions.py tests/test_goicp_p2p_adapter.py
git commit -m "feat: normalize selected visibility predictions"
```

### Task 3: 六样本十方法推理编排与一致性门禁

**Files:**
- Create: `scripts/run_visibility_best3_predictions.sh`
- Create: `tests/test_run_visibility_best3_predictions.py`
- Create: `docs/VISIBILITY_BEST3_VISUALIZATION.md`

**Interfaces:**
- Consumes: Task 1 的子集数据根、Task 2 的收集器、设计文档中指定的 checkpoint/Conda 环境。
- Produces: `output/visualization/visibility_0.2_0.4_livermatch_style/predictions/` 下 60 个方法预测缓存与运行日志；支持 `P2P_GPU`、`P2P_DRY_RUN`、`P2P_FORCE`。

- [ ] **Step 1: 编写命令展开失败测试**

增加 `test_dry_run_contains_all_models_environments_checkpoints_and_two_datasets`，断言 dry-run 恰好覆盖 10 种方法×2 个子集，环境、checkpoint、`K=5`、save-predictions 和输出路径正确。

- [ ] **Step 2: 运行 dry-run 测试并确认失败**

Run: `/home/yangx/miniconda3/envs/geo_v2/bin/python -m pytest tests/test_run_visibility_best3_predictions.py -v`

Expected: FAIL，原因是编排脚本尚不存在。

- [ ] **Step 3: 实现可恢复编排脚本**

先运行 Task 1 生成固定子集；随后依次调用 Ours、GeoTransformer、CASTv2、DFAT、Lepard 两种 solver、PARENet、LiverMatch 两种 solver 和 Go-ICP 的既有测试入口；每个命令完成后立即用 Task 2 收集器写统一缓存。已有且 checkpoint、样本和协议元数据匹配时跳过，`P2P_FORCE=1` 才覆盖。

- [ ] **Step 4: 增加 Ours 指标一致性门禁**

在脚本收集阶段比较六个 Ours 重算值与 selection 中的完整评测值；绝对差大于 `0.05 mm` 时返回非零并打印数据集、样本和两个数值。

- [ ] **Step 5: 编写运行说明**

在 `docs/VISIBILITY_BEST3_VISUALIZATION.md` 记录一键命令、输出目录、恢复执行、GPU 选择、失败重跑和结果审计方法。

- [ ] **Step 6: 运行 Task 3 测试与 dry-run**

Run: `/home/yangx/miniconda3/envs/geo_v2/bin/python -m pytest tests/test_run_visibility_best3_predictions.py -v`

Run: `P2P_DRY_RUN=1 scripts/run_visibility_best3_predictions.sh`

Expected: 测试 PASS；dry-run 输出 20 个模型/数据集命令且不创建预测。

- [ ] **Step 7: 提交 Task 3**

```bash
git add scripts/run_visibility_best3_predictions.sh tests/test_run_visibility_best3_predictions.py docs/VISIBILITY_BEST3_VISUALIZATION.md
git commit -m "feat: orchestrate best-three visibility inference"
```

### Task 4: LiverMatch 风格 3×12 对比图渲染

**Files:**
- Create: `visualization/livermatch_style_comparison.py`
- Create: `tests/test_livermatch_style_comparison.py`
- Modify: `visualization/__init__.py`

**Interfaces:**
- Consumes: Task 1 的 selection/子集、Task 2 的统一预测缓存。
- Produces: `render_dataset_comparison(dataset: str, cases: Sequence[Mapping], predictions_root: Path, output_root: Path, backend: str = "auto") -> dict`；两套 PNG/PDF、72 个可选面板缓存、`metrics.csv` 和 `manifest.json`。

- [ ] **Step 1: 编写布局、颜色和指标失败测试**

增加测试断言固定 12 列顺序、3 行、蓝色/红色常量、Courier 标题模板、`points @ R.T + t` 和体积标记 RMS-TRE；验证同行共享相机参数。

- [ ] **Step 2: 编写失败与后备渲染测试**

模拟一个 `status=failed` 方法和 PyVista 初始化异常，断言输出仍有 36 个格、失败格标题包含 `FAILED`、manifest 记录原因且 backend 为 software。

- [ ] **Step 3: 运行渲染测试并确认失败**

Run: `/home/yangx/miniconda3/envs/geo_v2/bin/python -m pytest tests/test_livermatch_style_comparison.py -v`

Expected: FAIL，原因是渲染模块尚不存在。

- [ ] **Step 4: 实现数据加载、相机和 PyVista 面板渲染**

从子集 NPZ 按当前 benchmark 预处理约定加载 source、target、标记和真值；每行以 Initial、十方法、Ground Truth 计算注册点云和 RMS-TRE。PyVista off-screen 使用白底、球形点、指定 RGB、无坐标轴，并对同行复用同一 camera position/focal point/view-up。

- [ ] **Step 5: 实现软件后备与大图拼接**

复用 `visualization.render.render_point_clouds` 作为 software 后备；用 Matplotlib 拼成 3×12，列标题为方法名与 `RMS-TRE: xx.xx mm`，左侧为样本标识，顶部为数据集名，输出 500-DPI PNG 与矢量容器 PDF。

- [ ] **Step 6: 实现审计输出**

写 `metrics.csv` 与 `manifest.json`，包含样本、方法、RMS-TRE、状态、预测路径、checkpoint、渲染后端和最终图片路径；导出 API 到 `visualization/__init__.py`。

- [ ] **Step 7: 运行 Task 4 测试**

Run: `/home/yangx/miniconda3/envs/geo_v2/bin/python -m pytest tests/test_livermatch_style_comparison.py tests/test_error_heatmap.py -v`

Expected: PASS，且不破坏现有可视化测试。

- [ ] **Step 8: 提交 Task 4**

```bash
git add visualization/livermatch_style_comparison.py visualization/__init__.py tests/test_livermatch_style_comparison.py
git commit -m "feat: render LiverMatch-style method comparisons"
```

### Task 5: 执行六样本推理、生成两张大图并验收

**Files:**
- Generated: `output/visualization/visibility_0.2_0.4_livermatch_style/**`
- Modify: `docs/VISIBILITY_BEST3_VISUALIZATION.md`（写入实际选样和验收结果）

**Interfaces:**
- Consumes: Tasks 1–4 的完整流水线。
- Produces: 两张 3×12 PNG/PDF、60 个真实预测缓存、selection、metrics、manifest、日志和 QA 记录。

- [ ] **Step 1: 运行六样本十方法预测**

Run: `P2P_GPU=1 scripts/run_visibility_best3_predictions.sh`

Expected: 两个数据集各 3 个固定样本；每个样本 10 条明确成功或失败状态；Ours 一致性门禁通过。

- [ ] **Step 2: 运行两数据集渲染**

Run: `/home/yangx/miniconda3/envs/geo_v2/bin/python -m visualization.livermatch_style_comparison --root output/visualization/visibility_0.2_0.4_livermatch_style --backend auto --dpi 500`

Expected: 生成 `in_silico_best3_comparison.{png,pdf}` 与 `in_vitro_best3_comparison.{png,pdf}`。

- [ ] **Step 3: 运行完整自动验证**

Run: `/home/yangx/miniconda3/envs/geo_v2/bin/python -m pytest tests/test_prepare_visibility_best3.py tests/test_collect_visibility_predictions.py tests/test_run_visibility_best3_predictions.py tests/test_livermatch_style_comparison.py tests/test_goicp_p2p_adapter.py tests/test_error_heatmap.py -v`

Expected: 全部 PASS。

- [ ] **Step 4: 执行产物结构检查**

检查 selection 每组恰好 3 项、manifest 为 3×12、预测缓存为 2×3×10、PNG/PDF 非空且尺寸一致、所有标题有方法名和 RMS-TRE 或 FAILED。

- [ ] **Step 5: 人工视觉验收**

逐张打开两张 PNG，与 `/home/yangx/code/new_deform/ai_worker/LiverMatch/P2P_demo.png` 对照白底、蓝红球形点、Courier 标题、同行视角、裁切和留白；若 PyVista 后备发生，记录原因并确认软件渲染仍符合布局。

- [ ] **Step 6: 更新 QA 记录并提交**

```bash
git add docs/VISIBILITY_BEST3_VISUALIZATION.md
git commit -m "docs: record best-three visualization QA"
```
