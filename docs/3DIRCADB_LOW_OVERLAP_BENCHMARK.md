# 3D-IRCADb 低重叠刚性配准基准运行手册

## 已验证协议

- 数据源：`/mnt/data3/yangx/3Dircadb/3Dircadb1`
- 正式数据：`/mnt/data3/yangx/3Dircadb/low_overlap_rigid_40_seed20260822`
- 20 个病例，每个病例生成 1 个嵌套 pair，并分别保存 20% 和 30% 两档，共 40 个样本
- 固定随机种子 `20260822`，target 高斯噪声 `σ=2 mm`
- 主指标：`SR@5 mm`；严格召回：`RRE < 5° 且 RTE < 5 mm`
- 所有输出变换均为原始毫米坐标中的 source→target 4×4 刚性变换
- 不使用 3D-IRCADb 训练、微调、阈值搜索或 checkpoint 修改

## 代码入口

| 方法 | 原生入口 |
|---|---|
| Ours | `RTORv2/experiments/geotransformer.p2p_liver/test_3dircadb.py` |
| GeoTransformer | `GeoTransformer_base/experiments/geotransformer.p2p_liver/test_3dircadb.py` |
| CASTv2 | `CASTv2/test_3dircadb.py` |
| DFAT | `DFAT-main/experiments/geotransformer.p2p_liver/test_3dircadb.py` |
| Lepard / Lepard+P2P | `Lepard/evaluate_3dircadb.py` |
| PARENet | `PARENet/experiments/P2P/test_3dircadb.py` |
| LiverMatch / LiverMatch+P2P | `RTORv2/tools/evaluate_livermatch_3dircadb.py` |
| Go-ICP | `go-icp_cython/evaluate_3dircadb.py` |

统一入口为：

```bash
cd /home/yangx/code/new_deform/ai_worker/RTORv2
./scripts/run_3dircadb_benchmark.sh METHOD VISIBILITY
```

`METHOD` 支持：`ours geotransformer castv2 dfat lepard lepard_p2p parenet livermatch livermatch_p2p goicp all`。`VISIBILITY` 支持：`020 030 all`。

## 数据生成与校验

```bash
cd /home/yangx/code/new_deform/ai_worker/RTORv2
./scripts/build_3dircadb_low_overlap.sh

/home/yangx/miniconda3/envs/geo_v2/bin/python \
  tools/build_3dircadb_low_overlap.py --validate-only \
  --output-root /mnt/data3/yangx/3Dircadb/low_overlap_rigid_40_seed20260822
```

成功输出应为：

```json
{"cases": 20, "pairs": 20, "samples": 40, "vis020": 20, "vis030": 20}
```

目标目录存在且 manifest 参数一致时，构建器只校验并复用；参数不一致时拒绝覆盖，需要指定新的 `--output-root`。

## 每个模型的正式运行命令

```bash
cd /home/yangx/code/new_deform/ai_worker/RTORv2

./scripts/run_3dircadb_benchmark.sh ours all
./scripts/run_3dircadb_benchmark.sh geotransformer all
./scripts/run_3dircadb_benchmark.sh castv2 all
./scripts/run_3dircadb_benchmark.sh dfat all
./scripts/run_3dircadb_benchmark.sh lepard all
./scripts/run_3dircadb_benchmark.sh lepard_p2p all
./scripts/run_3dircadb_benchmark.sh parenet all
./scripts/run_3dircadb_benchmark.sh livermatch all
./scripts/run_3dircadb_benchmark.sh livermatch_p2p all
./scripts/run_3dircadb_benchmark.sh goicp all
```

一次顺序运行全部方法并在完整运行结束后自动汇总：

```bash
./scripts/run_3dircadb_benchmark.sh all all
```

默认结果目录：

```text
/home/yangx/code/new_deform/ai_worker/RTORv2/results/3dircadb_low_overlap_40_seed20260822/
```

## 可选环境变量

```bash
export IRCADB_GPU=1
export IRCADB_LIMIT=0
export IRCADB_NUM_WORKERS=0
export IRCADB_GOICP_TIMEOUT=120
export IRCADB_DATA_ROOT=/mnt/data3/yangx/3Dircadb/low_overlap_rigid_40_seed20260822
export IRCADB_RESULT_ROOT=/home/yangx/code/new_deform/ai_worker/RTORv2/results/3dircadb_low_overlap_40_seed20260822
```

- `IRCADB_LIMIT=1` 用于单样本 smoke；正式实验必须设为 `0`。
- `IRCADB_DRY_RUN=1` 只打印十种方法的完整原生命令，不执行。
- 单独跑 `020` 或 `030` 时输出到各方法的 `visibility_020/` 或 `visibility_030/` 子目录，避免互相覆盖。
- 当前实现按方法完整重跑，不做样本级断点跳过。已有合法结果不会被汇总器误当成完整新结果；中断后应重跑对应方法。
- Go-ICP 每个样本最多 120 秒，40 个样本理论超时上界约 1.3 小时；超时与异常保留为失败样本，不写单位阵预测。

## 手工汇总与完整性校验

```bash
/home/yangx/miniconda3/envs/geo_v2/bin/python \
  tools/summarize_3dircadb_benchmark.py \
  --data-root /mnt/data3/yangx/3Dircadb/low_overlap_rigid_40_seed20260822 \
  --result-root /home/yangx/code/new_deform/ai_worker/RTORv2/results/3dircadb_low_overlap_40_seed20260822
```

该命令严格检查十种方法的 manifest 身份、样本 ID 唯一性和 40 个样本完整性，生成 `summary.csv`、`summary_by_case.csv` 与 `summary.json`。仅调试不完整结果时才使用 `--allow-incomplete`。

## 可视化

全量渲染（每张图 2×12，`viridis`，PNG/PDF/JSON）：

```bash
./scripts/render_3dircadb_comparison.sh
```

渲染指定样本：

```bash
./scripts/render_3dircadb_comparison.sh \
  --sample-ids case01_pair00_vis020 \
  --dpi 400 \
  --colormap viridis
```

Go-ICP 含显式失败/超时记录时，允许将该列标成 `FAILED`：

```bash
./scripts/render_3dircadb_comparison.sh \
  --sample-ids case01_pair00_vis020 \
  --colormap viridis \
  --allow-failed
```

还可使用 `--case-ids 1,2`、`--pair-ids 0,1`、`--visibility 0.20|0.30|all`。默认图像目录为 `$IRCADB_RESULT_ROOT/figures`。

## 已执行验证

- RTORv2 聚焦测试：37 passed
- CASTv2：1 passed；Lepard：2 passed；PARENet：1 passed；Go-ICP：4 passed
- 正式数据清单：40 样本，20%/30% 各 20，20 病例、20 个嵌套 pair
- 十种方法均完成 `case01_pair00_vis020` 原生单样本链路；九种学习方法成功写出变换，Go-ICP 在 5 秒 smoke 限制下写出显式 timeout
- smoke 可视化为 2400×580 PNG，包含 12 列、2 行、`viridis` 统一色标和毫米误差范围
