# RTOR/A3 因果实验矩阵（Stage 2）

本矩阵以当前源码审计为准。完整指标定义、输出schema和fail-closed规则见`docs/superpowers/specs/2026-09-17-rtor-a3-evidence-instrumentation-design.md`。

## 1. 研究问题与主假设

| ID | 研究问题 | 干预 | 首要中间终点 | 下游终点 | 支持条件 |
|---|---|---|---|---|---|
| H1a | RTOR descriptor是否更可辨别 | pre→post RTOR descriptor | R@1/3/5/10、MRR、margin、AUC | PIR | 多数方向/病例按预期改善 |
| H1b | overlap是否保留共同可见候选 | uniform→predicted soft weights，budget固定256 | GT query/pair recall、candidate survival、top-K precision | PIR、RMS-TRE | soft calibration提高survival且非共同scale效应 |
| H1c | hard focus是否误删候选 | soft全support→legacy source focus | GT source-node/pair survival | PIR、failure count | 若下降则该branch不支持claim |
| H1d | overlap head本身是否正确 | prediction vs GT node label | ROC/PR-AUC、F1、IoU、ECE、Brier | 不直接以TRE替代 | 分类与校准证据均成立 |
| H2a | fine refiner是否改善局部matching | 完全相同coarse tuple，OFF→ON | fine precision/recall/IR、MRR、entropy、local error | LGR pose、RMS-TRE | proposal identity通过且fine指标改善 |
| H2b | 是否主要帮助ambiguous patch | ambiguity quantile strata | Hard−Easy增益差 | 分层pose | Hard收益更大且CI支持 |
| H3 | 两模块是否协同 | 独立训练2×2 factorial | coarse/fine stage特异指标 | RMS-TRE、SR@20 | interaction CI决定synergy/redundancy/不确定 |
| H4 | 捕获多少oracle gap | predicted→module→oracle | Oracle Gain Capture | RMS-TRE | oracle确实优于baseline时才定义 |

## 2. Stage 1–4执行矩阵

| Stage | 工作 | 输入 | 输出 | Gate |
|---|---|---|---|---|
| 1 | 当前源码事实审计 | source/config/checkpoint | `research/module_audit.md` | 作用路径与风险有源码定位 |
| 2 | 指标与干预预注册 | Stage 1事实 | 本文件与design spec | 每个claim有直接中间终点 |
| 3 | 旁路instrumentation | 当前模型API | `research/scripts/evidence/*`及tests | 不改核心模型；TDD；预测等价 |
| 4 | 20例smoke | fixed IDs + epoch-150 checkpoints | `research/results/<run_id>` | proposal identity、finite、schema、plots |

## 3. RTOR阶段消融

| ID | Features | overlap weights | source focus | 是否需重训 | 可支持的claim |
|---|---|---|---|---|---|
| R0 | pre-RTOR | uniform | OFF | 否 | baseline retrieval |
| R1 | post-RTOR | uniform | OFF | 否 | descriptor refinement即时作用 |
| R2 | post-RTOR | predicted soft | OFF | 否 | continuous calibration即时作用 |
| R3 | post-RTOR | predicted soft | legacy | 否 | hard focus增益或损害 |
| RO | post-RTOR | oracle soft | OFF | 否 | overlap oracle上界 |
| T-only | topology graph only | OFF | OFF | 是 | 后续训练消融；第一轮不实现 |
| G-attr | +distance/curvature/linearity/planarity | OFF | OFF | 是 | 后续训练消融；第一轮不实现 |
| Poincaré | +Poincaré edge distance | OFF | OFF | 是 | 后续训练消融；第一轮不实现 |

## 4. Fine固定候选矩阵

| ID | checkpoint | coarse pairs/points/masks/node scores | fine refiner | Sinkhorn/LGR | 用途 |
|---|---|---|---|---|---|
| F0 | full | 保存tuple T | OFF | frozen | baseline fine |
| F1 | same full | exact tuple T | ON | frozen | refiner纯即时作用 |
| FO | same full | exact tuple T | GT assignment | same LGR | oracle fine gap |

F0/F1若任一proposal hash不同，整例无效。

## 5. 端到端factorial

| Method | architecture | RTOR | fine refiner | profile | candidate budget | Sinkhorn | LGR |
|---|---|---:|---:|---|---:|---:|---|
| B | geotransformer | OFF | OFF | legacy | 256 | 100 | identical |
| R | rtor_only | ON | OFF | legacy | 256 | 100 | identical |
| F | a3_only | OFF | ON | legacy | 256 | 100 | identical |
| RF | rtor_a3 | ON | ON | legacy | 256 | 100 | identical |

当前历史checkpoint均为单seed 7351且metadata为空；A3-only manifest另有路径/resume异常。它们在Stage 4只用于smoke，正式factorial结论状态为`NOT YET VALIDATED`。

## 6. Smoke样本与预期产物

```text
dataset: in_silico
noise: none
indices: 0:20
seed: 7351
neighbor_limits: [7,22,32,39]
status: SMOKE_ONLY_NOT_FOR_CLAIMS
```

产物：manifest、逐例JSONL、coarse/overlap/fine/pose CSV、summary，以及overlap histogram、candidate rank和fixed-coarse fine paired figures。

## 7. Full run前必须人工确认

- GT node/fine label与论文Methods一致；
- fixed-coarse hash gate在20/20样本通过；
- instrumentation capture前后pose/coarse/fine输出一致；
- 四格checkpoint provenance可接受或完成重训；
- primary endpoint和bootstrap clustering unit已确定；
- 不把smoke数值写成论文结论。
