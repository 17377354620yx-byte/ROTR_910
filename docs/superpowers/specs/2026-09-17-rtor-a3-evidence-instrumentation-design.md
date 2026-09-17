# RTOR/A3 Evidence Instrumentation Design

**日期：** 2026-09-17  
**状态：** 已批准的旁路观测方案  
**事实基础：** `research/module_audit.md`  
**适用实验：** `experiments/geotransformer.p2p_liver`  

## 1. 目标与边界

本设计新增一个只读科研观测层，用于建立以下因果链：

```text
baseline failure
  → RTOR descriptor / overlap calibration / focus 对 coarse ranking 的影响
  → 固定预算下 GT candidate survival
  → 固定 coarse candidates 下 fine refiner 对 point matching 的影响
  → Sinkhorn / LGR 输入质量
  → pose 与 RMS-TRE
```

第一轮仅完成源码审计、实验矩阵、evaluation/visualization instrumentation 和20–50例smoke test。不得修改：

- `geotransformer/modules/liver/registration_model.py`及任何核心模型实现；
- dataset split、loader、preprocessing、voxel size；
- GT correspondence定义、matching radius；
- coarse budget、Sinkhorn、LGR；
- checkpoint参数。

允许新增`research/`下脚本和结果，并新增对应tests。观测器只读取forward中间张量或用现有模块纯函数重算指标；正常预测输出保持逐元素不变。

## 2. 选择的架构

采用“外部旁路观测器”，而不是修改模型暴露debug outputs或复制整条forward。

### 2.1 组件

```text
research/scripts/evidence/
  metrics.py       纯NumPy/Torch指标，无模型状态
  capture.py       临时forward hooks；生命周期显式管理
  interventions.py 固定coarse proposal和RTOR阶段旁路重算
  provenance.py    git/config/checkpoint/data hash与manifest校验
  runner.py        CLI、数据循环、JSONL/CSV输出
  visualize.py     从结果文件生成PNG/PDF/SVG

research/run_smoke_evidence.py
  用户入口；默认20例，不改变实验包
```

测试放在：

```text
tests/test_research_evidence_metrics.py
tests/test_research_evidence_capture.py
tests/test_research_evidence_provenance.py
```

### 2.2 数据流

一次样本分析分成三类运行：

1. **正常端到端运行**：保存模型原始pose、coarse pairs、fine correspondences及模块输入输出。
2. **RTOR旁路阶段重算**：用同一次forward捕获的pre/post descriptor、predicted probability和mask计算四个score阶段，不再次修改模型。
3. **fixed-coarse fine intervention**：保存正常运行的coarse pair indices和node scores；临时以forward hook令`SuperPointMatching`返回同一tuple，分别运行fine refiner OFF/ON。两次运行的coarse indices、points、masks和node scores必须hash一致，否则样本fail-closed。

hook必须在`try/finally`或context manager中移除。每个干预结束后恢复`a3_enabled`、`focus_enabled`和所有临时状态。

## 3. 精确指标定义

### 3.1 GT coarse label

使用当前模型输出的：

```text
gt_node_corr_overlaps > cfg.coarse_matching.overlap_threshold (=0.1)
```

形成exact GT pair集合。有效query定义为至少有一个GT partner的superpoint。不得以“每例至少命中一个pair”的旧字段代替完整recall。

### 3.2 Descriptor retrieval

对ref→src和src→ref分别计算，再报告方向值与macro mean：

- Recall@1/3/5/10：有效query的任一GT partner是否进入per-query top-k；
- MRR：最佳GT partner rank的倒数，对有效query取均值；
- positive score：GT pair score；
- negative score：非GT有效pair score；
- margin：mean positive − mean negative，同时补充mean positive − mean hardest-negative；
- ROC-AUC：GT pair与非GT pair的rank statistic。

baseline descriptor阶段使用RTOR输入features；refined descriptor阶段使用RTOR输出features。score公式与`SuperPointMatching`完全一致：`exp(-pairwise_distance(..., normalized=True))`及同样dual normalization。

### 3.3 Candidate survival

固定global budget `N=cfg.coarse_matching.num_correspondences`，默认256。预选择候选定义为每个有效query的per-query top-10 exact pairs；这是可复现的“descriptor已可检索”集合。

分别计算：

```text
GT query recall before
  = per-query top-10中命中任一GT pair的GT-bearing query / GT-bearing query

GT query recall after
  = global top-N中命中任一GT pair的GT-bearing query / GT-bearing query

GT pair recall after
  = global top-N命中的exact GT pairs / all exact GT pairs

Candidate Survival Rate (query)
  = before可检索且after仍有GT命中的query / before可检索query

Candidate Survival Rate (pair)
  = preselection top-10 GT pairs与global top-N的交集 / preselection top-10 GT pairs

Top-K precision
  = global top-N exact GT pairs / N
```

每个阶段都计算：baseline descriptor、RTOR descriptor、soft overlap calibrated、legacy hard focus。hard focus必须额外报告GT source-node survival和被删除GT pair数量。

### 3.4 Overlap prediction

node target与训练loss一致：把GT pair overlap按ref/src index做`amax`，再以`>0.1`二值化。分别报告ref、src和micro pooled：

- ROC-AUC、PR-AUC；
- threshold=0.5的precision、recall、F1、IoU；
- Brier score；
- 10-bin ECE与reliability rows；
- positive/negative probability distribution summary及原始histogram bins。

如果单例只有一个class，单例AUC写`null`并在aggregate时用pooled predictions重算，禁止伪造0.5或丢弃而不记录。

### 3.5 Fine matching

fixed-coarse OFF/ON使用相同GT transform只做事后标签：

```text
distance = ||T_oracle(p_src) - p_ref||
correct(t) = distance < t
```

同时报告normalized threshold `cfg.fine_loss.positive_radius`，以及按每例`physical_scale`换算的1/2/5 mm阈值。无`physical_scale`的训练样本只报告normalized阈值。

指标包括：

- selected fine precision与IR；
- GT valid pair recall；
- correct inlier count；
- local correspondence error mean/median/P90；
- correct/incorrect selected confidence；
- row-normalized assignment entropy（排除dustbin但记录dustbin mass）；
- reciprocal top-1 mutual match ratio；
- pre-Sinkhorn fine Recall@1/5、MRR和top1-top2 gap；
- post-Sinkhorn同类ranking指标。

fine recall分母是patch内所有valid、阈值内point pairs，因此必须与precision分开解释；它不是一对一GT cardinality。

### 3.6 Ambiguity strata

patch ambiguity primary score预注册为baseline pre-Sinkhorn的top1-top2 gap取负（gap越小越难）。secondary descriptors为patch GT overlap ratio、baseline entropy和局部几何签名最近邻重复度。

full run用全测试集的33.3%/66.7% quantiles划分Easy/Medium/Hard；smoke只验证分层代码，输出明确标记`SMOKE_ONLY_NOT_FOR_CLAIMS`，不得用20例quantile写论文结论。

### 3.7 Pose与case-level指标

复用现有Evaluator与marker RMS-TRE定义，记录PIR、IR、RR、RRE、RTE、RMSE、RMS-TRE、SR@20。fixed-coarse OFF/ON还记录pose变化，但只有fine中间指标按预期变化时才允许归因。

## 4. 实验矩阵

### 4.1 端到端2×2 factorial

| ID | checkpoint architecture | RTOR | fine refiner | 解释 |
|---|---|---:|---:|---|
| B | geotransformer | OFF | OFF | baseline |
| R | rtor_only | ON | OFF | independently trained RTOR endpoint |
| F | a3_only | OFF | ON | independently trained fine-refiner endpoint |
| RF | rtor_a3 | ON | ON | independently trained full endpoint |

仅当四个manifest的训练/数据/solver协议通过validator后，才计算论文级interaction：

```text
interaction = (RF − R) − (F − B)
```

lower-is-better指标在解释时统一转成improvement方向，不能只根据interaction符号声称synergy。

历史四个checkpoint缺metadata，A3-only manifest另有路径异常；smoke可用并标记provenance warning，正式结论前必须补齐可审计证据或重训。

### 4.2 RTOR mechanism matrix

| 阶段 | descriptor | soft overlap | hard focus | 改变变量 |
|---|---|---|---|---|
| R0 | pre-RTOR | OFF | OFF | baseline ranking |
| R1 | post-RTOR | OFF | OFF | descriptor refinement |
| R2 | post-RTOR | ON | OFF | continuous calibration |
| R3 | post-RTOR | ON | ON | legacy discrete focus |
| RO | post-RTOR | oracle | OFF | oracle overlap上界 |

topology-only、几何属性、Poincaré内部拆分需要新增训练开关和重训，不属于第一轮旁路instrumentation；不得用inference-time置零冒充正式训练消融。

### 4.3 Fixed-coarse fine matrix

| 阶段 | coarse tuple | fine refiner | 目的 |
|---|---|---|---|
| F0 | 同一保存tuple | OFF | baseline fine |
| F1 | 与F0逐元素/hash一致 | ON | refiner即时作用 |
| FO | 与F0一致 | oracle assignment | oracle gap上界（后续full stage） |

### 4.4 Oracle interventions

- Oracle coarse：GT coarse + baseline fine、GT coarse + refined fine，与predicted coarse对比。
- Oracle fine：固定coarse，用GT distance构造理想assignment/LGR输入。
- Oracle Gain Capture对lower-is-better统一定义为：

```text
(baseline_error − method_error) / (baseline_error − oracle_error)
```

分母≤0或oracle不优时输出`null`和原因，不生成误导性比例。

## 5. 输出与provenance

每次run写入独立目录，不覆盖已有证据：

```text
research/results/<run_id>/
  manifest.json
  samples.jsonl
  coarse_metrics.csv
  overlap_metrics.csv
  fine_metrics.csv
  pose_metrics.csv
  summary.json
  figures/
```

manifest必须包含：

- git commit及dirty diff摘要；
- checkpoint绝对路径和SHA256；
-完整resolved config；
- seed、dataset、dataset list SHA256、样本IDs和数量；
- coarse budget、neighbor limits、Sinkhorn iterations；
- RTOR/fine-refiner状态、interaction/registration profile；
- Python/PyTorch/CUDA/GPU；
- timestamp和命令；
- provenance validator warnings/failures；
- schema version和metric definitions版本。

JSON禁止NaN/Infinity；不可定义值写`null`并带reason字段。

## 6. 可视化

smoke阶段生成最小验证图：

- overlap positive/negative histogram；
- GT candidate rank stage plot；
- fixed-coarse OFF/ON fine metric paired plot；
- 选定单例的baseline/refined fine correspondence对照。

所有图使用白底、固定颜色语义、固定相机/点大小/线数，并同时保存PNG、PDF、SVG。case selection规则先按metric排序再选，记录在JSON，禁止手选后隐藏退化案例。

## 7. Fail-closed检查

以下任一条件使该run不能标为`VALID_FOR_CLAIMS`：

- checkpoint strict load失败；
-配置与checkpoint metadata冲突；
- fixed-coarse两次run的indices、points、masks、node scores hash不同；
- instrumentation ON/OFF时正常预测输出不一致；
- 样本集合不配对；
- GT threshold、budget、Sinkhorn或LGR发生变化；
- 输出存在NaN/Infinity或未解释null；
- provenance必需字段缺失。

历史metadata缺失允许smoke运行，但状态只能是`SMOKE_ONLY_NOT_FOR_CLAIMS`。

## 8. 测试策略

严格TDD，每个生产函数先有失败测试：

1. synthetic score matrix验证Recall/MRR/margin/AUC与candidate survival手算值；
2. single-class overlap验证AUC=`null`，pooled aggregate可计算；
3. fine distance和physical-scale阈值验证；
4. fixed proposal context验证hook必定恢复；
5. 同一模型正常forward在capture前后输出逐元素一致；
6. fixed-coarse OFF/ON验证proposal hashes完全相同；
7. manifest缺字段或协议冲突时fail-closed；
8. JSON strict serialization拒绝NaN。

环境没有pytest，测试使用仓库现有`unittest discover`约定，不新增测试框架依赖。

## 9. Smoke test协议

- 数据：in-silico、noise none、发布test list前20例；另在manifest记录准确IDs。
- checkpoint：历史epoch-150四格用于端点plumbing；full RTOR+A3用于RTOR阶段和fixed-coarse fine机制。
- neighbor limits：`[7,22,32,39]`，因历史checkpoint无metadata，明确记录为外部已知值而非checkpoint证明事实。
- seed：7351；eval mode；不训练、不更新参数。
- 成功标准：全部20例完成、无非有限值、proposal identity gate通过、capture不改变原始预测、所有文件可重读、图可生成。
- 科研解释状态：`SMOKE_ONLY_NOT_FOR_CLAIMS`。任何数值只用于验证定义和管线，不写入贡献结论。

## 10. 后续full evaluation门

只有smoke通过并人工确认GT定义与fixed-coarse identity后，才运行6315例full test、noise strata、in-vitro、paired bootstrap、case analysis和效率测试。Stage 4结束不自动启动full dataset。
