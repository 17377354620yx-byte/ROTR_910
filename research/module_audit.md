# RTOR 与局部几何细化模块源码事实审计

审计日期：2026-09-17  
审计代码：`37b45ec`（分支 `optimize/p2i-selective-bf16`）  
审计范围：`experiments/geotransformer.p2p_liver` 及其直接调用的 `geotransformer/modules` 代码  
证据等级：本文件仅陈述当前源码事实和由计算图直接推出的结论；所有性能、有效性和因果结论均标为“待实验验证”。

## 1. 结论摘要

两个新增模块都真实参与最终预测，不是只计算辅助 loss 的旁路模块：

- Module 1 的源码名称是 `TopologyOverlapRefiner`。它在 GeoTransformer coarse descriptor 之后、`SuperPointMatching` 之前更新双侧 coarse descriptor，同时预测双侧 overlap logits。推理时 logits 经 sigmoid 转成连续权重，直接重排 coarse similarity；`legacy` profile 还会据此硬筛选完整 source 的 superpoints。
- Module 2 的源码名称是 `GeometryAwareFineRefiner`。它在 coarse patch pair 已选定之后、点积相似度和 Sinkhorn 之前，用刚体不变的局部几何签名作为双向 cross-attention bias，直接修改 fine patch features。

没有发现 marker、GT correspondence 或 GT transform 被预测分支显式读取。测试数据中的 marker 拟合 transform 会在 eval forward 中生成 GT node correspondence，但这些 GT correspondence 只进入输出诊断；只有 `self.training=True` 时才替换/混合 fine-stage coarse proposals。因此当前 eval 预测路径没有直接 GT 泄漏。

存在三个必须控制的因果混杂：

1. `legacy` 训练的 fine stage 默认使用 100% GT coarse proposals，但推理使用 100% predicted proposals；这是明确的 train/test exposure gap。
2. `legacy` 训练保留所有 node，推理才按预测 overlap 硬筛完整 source；这是第二个 train/test behavior gap。
3. `test.py` 对权重 `strict=True`，但元数据只主动核对 `interaction_profile`。`registration_profile`、dual encoder、数据协议等无参数或部分无参数设置可能在 strict load 成功时仍漂移，必须由新增 manifest validator 单独拦截。

## 2. 实际运行链

### 2.1 推理链

```text
Input: ref/target partial surface, src complete liver, constant point features
  ↓
KPConvFPN (shared backbone; optional dual fine backbone)
  ├─ fine features
  └─ coarse features
  ↓
GeometricTransformer
  ↓
L2-normalized baseline coarse descriptors
  ↓
TopologyOverlapRefiner [RTOR, if enabled]
  ├─ topology/cross-support refined coarse descriptors
  └─ bilateral overlap logits → sigmoid probabilities → soft weights
  ↓
optional source-only overlap-region hard focus [legacy inference only]
  ↓
SuperPointMatching
  ├─ dual-normalized descriptor similarity
  ├─ bilateral soft overlap calibration
  └─ global top-256 coarse correspondences
  ↓
point-to-node patch gathering (64 fine points per selected superpoint)
  ↓
GeometryAwareFineRefiner [fine refiner, if enabled]
  ↓
fine feature dot-product similarity
  ↓
LearnableLogOptimalTransport (100 Sinkhorn iterations)
  ↓
LocalGlobalRegistration
  ├─ mutual top-3 fine matches, confidence > 0.05
  ├─ coarse node score multiplies fine score
  ├─ local hypotheses
  └─ 5-step global weighted Procrustes refinement
  ↓
single rigid transformation
```

主要入口为 `geotransformer/modules/liver/registration_model.py::GeoTransformer.forward`。实验入口 `experiments/geotransformer.p2p_liver/model.py` 只转发该类。

### 2.2 训练链与推理链的差异

```text
训练：predicted coarse 仍被计算并写入 output_dict
                  ↓
       legacy/soft_overlap: predicted_ratio = 0
                  ↓
       fine stage 改用最多 128 个随机 GT coarse targets

       cooperative: epoch 5→20 将 predicted ratio 线性升至 25%
                  ↓
       GT 与 predicted proposals 混合，预算仍以 GT target 数为准

推理：始终使用 predicted global top-256 coarse correspondences
```

训练时不会执行 source hard focus；`legacy` 推理会执行，`cooperative` 与 `soft_overlap` 配置会关闭它。

## 3. Module 1：`TopologyOverlapRefiner`

### 3.1 定义、路径与开关

- class：`TopologyOverlapRefiner`
- 文件：`geotransformer/modules/liver/topology_overlap.py`
- 构建位置：`geotransformer/modules/liver/registration_model.py::GeoTransformer.__init__`
- 总开关：`cfg.ablation.rtor_enabled`
- architecture 映射：
  - `geotransformer`：OFF
  - `a3_only`：OFF
  - `rtor_only`：ON
  - `rtor_a3`：ON
- 默认实验 architecture：训练和测试 CLI 都是 `rtor_a3`，故默认启用。

### 3.2 forward 输入输出

输入：

```python
forward(
    ref_points:    Tensor[N_ref, 3],
    src_points:    Tensor[N_src, 3],
    ref_features:  Tensor[N_ref, 256],
    src_features:  Tensor[N_src, 256],
)
```

输出：

```python
(
    refined_ref_features: Tensor[N_ref, 256],
    refined_src_features: Tensor[N_src, 256],
    {
        "ref_overlap_logits": Tensor[N_ref],
        "src_overlap_logits": Tensor[N_src],
    },
)
```

输入 features 是 KPConvFPN coarse features 经 `GeometricTransformer` 和 L2 normalization 后的描述子；coordinates 是 coarse superpoint coordinates。模块不接收 mask、GT correspondence、transform、marker 或标签。

### 3.3 实际计算

每侧点云独立执行：

1. 在 coarse coordinates 上构造 kNN 图，默认 `k=8`。
2. 计算局部刚体不变几何：median-NN 归一化距离、曲率、线性度、平面度。
3. 将 hidden feature 映射到 Poincaré ball，计算中心—邻居双曲距离。
4. 用 `[normalized distance, curvature, linearity, planarity, Poincaré distance]` 生成 edge gate，聚合邻居特征。
5. 在 ref/src graph features 间执行双向全连接 cross-support attention。
6. 由本侧 hidden、跨侧 aggregate、三个局部几何量、cross-attention maximum 和 normalized entropy 预测 overlap logit。
7. 以 residual 形式将 hidden 投影回 256 维并加到原 descriptor。

`_local_geometry` 使用 `@torch.no_grad()`，只阻断 coordinate/eigendecomposition 路径；descriptor graph、cross attention、overlap head 与 output projection 仍可训练。

### 3.4 对 coarse matching 的真实影响

RTOR 有三个彼此可分离的干预：

1. **descriptor refinement**：refined descriptors 替换 baseline descriptors，改变所有 descriptor distances。
2. **soft overlap calibration**：概率 `p` 映射为 `0.05 + 0.95 p`，ref/src 权重乘到 dual-normalized score matrix；随后只做全局 mass rescale。mass rescale 是全矩阵共同常数，不改变排序，因此排序改变来自双侧逐节点权重，而不是总 score scale。
3. **source hard focus（仅 legacy eval）**：当概率 spread ≥ 0.05 时，按阈值 0.5 选 source；至少保留 16 个、最多保留有效 source nodes 的 50%。被删除 node 在 top-K 前已不可参与候选。

ref/partial 一侧从不被 hard focus，始终使用原 `ref_node_masks`。

### 3.5 训练信号与梯度

- coarse circle loss 直接读取 refined `ref_feats_c/src_feats_c`，因此训练 descriptor refinement。
- overlap focal loss 用 GT node correspondence 的最大 overlap 构造双侧 binary labels，阈值为 0.1，权重为总 loss 的 0.5。
- GT indices/overlaps 在 overlap loss 中显式 `detach`；这是正确的标签处理，不会阻断 logits 的梯度。
- predicted coarse selection 整段位于 `torch.no_grad()`。因此 fine loss 不能穿过 top-K selector、overlap calibration 或 hard focus 回传给 RTOR。
- fine loss 仍会更新共享 KPConv fine features与 fine refiner，但不会通过离散 coarse indices 更新 RTOR/GeoTransformer coarse 分支。

初始化时 `output_proj` 和 overlap head 的最后一层为零。第一个优化步中，descriptor residual 分支的更早层不会从 coarse loss 得到梯度，overlap head 的更早层也不会从 overlap loss 得到梯度；最后一层先学习后，上游梯度才打开。这是短暂的零初始化梯度门，不是永久 stop-gradient。

### 3.6 是否进入最终预测

是。

- refined descriptors 进入 `SuperPointMatching`；
- overlap soft weights 进入同一个 top-K score matrix；
- legacy 的 source hard mask在相似度矩阵构造前删除候选；
- coarse pair indices 决定 A3/Sinkhorn 处理哪些 patch；
- coarse node scores 在 `use_global_score=True` 时还会乘到 LGR 的 fine correspondence scores。

因此 RTOR 可通过“候选集合”和“LGR 全局权重”两条路径影响最终 rigid transformation。

### 3.7 Module 1 实际解决什么问题

源码实现针对的是：complete source 中包含大量 partial target 不可见区域时，仅依赖全局 descriptor score 的固定 top-256 预算会把候选分配给不可见或不可靠节点。

它不是单纯的 topology feature enhancer，而是“descriptor refinement + bilateral visibility probability + coarse ranking calibration”的组合。`legacy` 还加入更强的 source hard pruning，但该 pruning 可能意外删除 GT-relevant candidates，必须单独消融，不能与 soft calibration 混为同一贡献。

### 3.8 理论上应优先改善的中间指标

按干预顺序应验证：

- descriptor-only：Recall@1/3/5/10、MRR、positive-negative margin；
- soft calibration：GT pair rank percentile、top-K precision/recall、GT query recall、GT pair recall、Candidate Survival Rate；
- overlap head：ROC-AUC、PR-AUC、F1、precision、recall、IoU、ECE/reliability、正负概率分布；
- hard focus：focus 前后 GT node/pair survival，特别是被误删的 GT source nodes；
- downstream：PIR、进入 fine stage 的有效 patch 比例、LGR 使用的正确 coarse scores、最终 RRE/RTE/RMS-TRE。

## 4. Module 2：`GeometryAwareFineRefiner`

### 4.1 定义、路径与开关

- class：`GeometryAwareFineRefiner`
- 内部 class：`LocalInteractionBlock`、`MaskedGeometryAttention`
- 文件：`geotransformer/modules/liver/fine_local_refiner.py`
- 构建位置：`geotransformer/modules/liver/registration_model.py::GeoTransformer.__init__`
- 总开关：`cfg.ablation.a3_enabled`
- architecture 映射：`a3_only` 与 `rtor_a3` 启用，另外两种关闭。
- 默认实验 architecture 为 `rtor_a3`，故默认启用。

报告和论文中建议使用源码事实名称“geometry-aware bidirectional fine patch refiner”，不要只写内部简称 A3。

### 4.2 forward 输入输出

输入：

```python
forward(
    ref_features: Tensor[P, K, 256],
    src_features: Tensor[P, K, 256],
    ref_points:   Tensor[P, K, 3],
    src_points:   Tensor[P, K, 3],
    ref_masks:    BoolTensor[P, K],
    src_masks:    BoolTensor[P, K],
)
```

其中 `P` 是当前 coarse proposal 数，推理通常为至多 256，训练 legacy 为至多 128 个 GT targets；`K=64`。

输出两个与输入同形状的 refined fine-feature tensors。模块不输出 correspondence、score 或 pose，也不读取 GT transform。

### 4.3 实际计算

1. 各 patch 独立中心化。
2. 用有效点两两距离均值归一化尺度。
3. 每个点的几何签名为 `[归一化径向距离, 归一化邻点距离均值, 归一化邻点距离标准差]`。
4. ref/src 签名的平均绝对差形成 additive attention bias：差越大，bias 越负。
5. 使用同一 `LocalInteractionBlock` 权重进行 ref→src 与 src→ref 双向 cross-attention。
6. 两个方向都读取更新前的 `old_ref/old_src`，无更新顺序依赖。
7. 输出为输入 feature 加带可学习 scalar gate 的 attention/FFN residual。

几何签名只使用 patch 内部中心化、尺度归一化距离，因此不需要已知全局相对 pose，且在理想刚体变换下不变。它并未执行显式图匹配、ICP 或 pose refinement。

### 4.4 对 Sinkhorn、LGR 与 pose 的真实影响

- refined features 直接替换 baseline fine features参与点积相似度。
- 点积除以 `sqrt(256)` 后进入同一 `LearnableLogOptimalTransport`，默认 100 次迭代。
- Sinkhorn 输出的 log assignment 进入同一 LGR；A3 不改变 LGR 参数、coarse node score 或 patch masks。
- 因此固定完全相同的 coarse indices、points、masks 和 node scores 后，baseline fine 与 A3 fine 的差异可归因于 refined fine features及其 Sinkhorn assignment。

### 4.5 训练信号与梯度

- 没有独立名为 A3 的 loss。
- A3 仅由 `FineMatchingLoss` 训练：GT transform 将 source patch points 对齐到 ref，以 0.04 normalized radius 构造 point-pair label和 dustbin label，对标注位置的 Sinkhorn log probability取负均值。
- LGR 位于 `torch.no_grad()`，最终 pose/RMS-TRE 不反向传播。
- `residual_scale` 初始化为 0。第一个优化步 attention/FFN 参数因乘零而收不到有效梯度，但 scalar gate 本身可收到梯度；gate 离开零后内部参数开始训练。这是短暂梯度门，不等于模块训练不到。
- 无显式 `.detach()` 作用于 fine patch features；它们可向 fine refiner和 KPConv backbone 反传。

### 4.6 是否进入最终预测

是。它改变 Sinkhorn 输入 logits，进而改变 mutual top-k fine correspondence、LGR 局部 hypothesis、全局 weighted Procrustes 和最终 rigid transformation。

### 4.7 Module 2 实际解决什么问题

源码实现针对的是：coarse patch pair 已基本正确时，仅依赖逐点 descriptor 点积会因相似局部形状、噪声、低 overlap 或非刚性形变而产生多义 assignment。模块用两侧 patch 内部的刚体不变几何签名约束双向 feature interaction，目标是提高正确 point-to-point match 相对于几何不相容 match 的置信度。

### 4.8 理论上应优先改善的中间指标

- pre/post refiner descriptor ranking：fine Recall@1/3/5、MRR、top1-top2 gap；
- pre/post Sinkhorn：correct/incorrect confidence、assignment entropy、mutual match ratio；
- LGR 输入：fine precision、fine recall、IR、inlier count、local correspondence error；
- ambiguity strata：hard patches 上的增益应大于 easy patches；
- downstream：在固定 coarse candidates 条件下的 LGR pose error 与 RMS-TRE。

## 5. GT、marker 与信息泄漏审计

### 5.1 预测分支不读取 GT 的证据

`GeoTransformer.forward` 在 eval 中虽然接收 `transform` 并计算 `gt_node_corr_indices/overlaps`，但：

- `_apply_rtor` 只接收 coarse coordinates 和 features；
- `select_overlap_region` 只接收 predicted probability 与 valid mask；
- `SuperPointMatching` 只接收 features、masks 和 predicted overlap weights；
- fine refiner只接收 selected patch features、points 和 masks；
- Sinkhorn/LGR 只接收模型 score、points、masks 和 coarse node scores。

GT coarse targets 只在 `if self.training:` 分支进入 fine proposal selection。因此 eval pose 不使用 GT pair。

### 5.2 需要准确表述的标签定义

in-silico target 是非刚性变形表面，并不存在精确唯一 SE(3) ground truth。数据适配器用 paired volumetric markers 做 least-squares rigid fit，得到 `oracle_transform`；它用于：

- 构造 GT node correspondence和 fine loss标签；
- 计算 RRE/RTE 诊断；
- 不用于 eval prediction。

最终临床相关主终点应是 marker RMS-TRE。RRE/RTE 相对的是“最佳拟合刚体近似”，不能写成相对真实物理刚体的绝对误差。

overlap head 的 GT label 也不是 dataset 中单个 `visibility` scalar，而是由 oracle transform、patch points、matching radius 计算出的 node-pair overlap再投影成每个 node 的最大 overlap。

### 5.3 instrumentation 的防泄漏要求

新增评估代码必须保证：

- GT transform只在模型 forward 完成后计算标签和指标；
- oracle intervention 必须单独命名，不能混入正常预测结果；
- fixed-coarse A3 对比只复用离散 indices/node scores，不把 GT fine labels注入 logits；
- 可增加一次“移除 eval transform 后预测输出不变”的 contract test，以证明预测路径不依赖 GT。

## 6. hard selection、detach 与未生效分支审计

| 检查项 | 源码事实 | 判定 |
|---|---|---|
| 两模块是否进入最终预测 | RTOR 改 coarse descriptors/weights/mask；fine refiner 改 Sinkhorn logits | 是 |
| loss 计算但输出未用于配准 | overlap logits既有 loss 又用于 coarse 权重/legacy focus；fine loss训练实际 Sinkhorn scores | 未发现 |
| 默认关闭 branch | 默认 `rtor_a3 + legacy`：两模块 ON，hard focus ON，predicted training proposals OFF | 必须记录 |
| permanent stop-gradient | coarse selection/LGR 被 no-grad；模块内部无永久 feature detach | 模块可训练，但无 pose loss/selector gradient |
| hard selection误删候选 | legacy eval source focus可在 top-K 前最多删至 50% | 高风险，必须单独测 survival |
| GT/marker 推理泄漏 | GT只生成标签；training-only target branch不在 eval执行 | 未发现直接泄漏 |
| train/test behavior一致 | GT proposals vs predicted proposals；train no focus vs legacy eval focus | 不一致 |
| RTOR 初始化训练门 | output projection、overlap final layer零初始化 | 短暂梯度门 |
| A3 初始化训练门 | residual scalar=0 | 短暂梯度门 |

## 7. checkpoint 与协议一致性

### 7.1 当前行为

- `test.py` 使用 `load_state_dict(..., strict=True)`。
- 正常 resume 先 strict load，然后进入基类恢复 epoch/optimizer/scheduler。
- `--warm_start` 使用 `strict=False`，但只允许 RTOR/fine-refiner keys缺失，并拒绝其他 missing/unexpected keys。
- checkpoint metadata保存 architecture、dual encoder、registration profile、interaction profile、neighbor limits和 seed。

### 7.2 缺口

`test.py` 构建模型前仅核对 checkpoint 的 `interaction_profile`。以下比较条件没有被完整 fail-closed 验证：

- architecture（多数情况下会因参数结构不同而 strict load失败，但不应依赖副作用）；
- `registration_profile`（主要改无参数 LGR阈值，strict load无法发现）；
- dual encoder与其他 config字段；
- dataset root/list及其 hash；
- candidate budget、Sinkhorn iterations、voxel/matching radius、seed；
- checkpoint SHA256、git commit、timestamp。

当前 test JSON 记录 checkpoint path、architecture、fine matching等部分字段，但不满足目标文件要求的完整 manifest。

### 7.3 baseline 公平性判定

源码提供四个结构开关，且共享相同 dataset factory、KPConv/GeoTransformer、candidate budget、Sinkhorn和LGR默认值。因此**配置层面可以**构造公平 2×2 factorial。

但仅凭当前 test output 不能证明任意现有四个结果已经公平，必须逐 checkpoint验证：

- 同一代码 commit或明确兼容 hash；
- seed、训练 epoch/checkpoint selection rule一致；
- dataset list/hash、noise、sample IDs一致；
- voxel size、neighbor limits、GT radius一致；
- coarse budget=256、training GT target budget=128一致；
- Sinkhorn=100、LGR完整配置一致；
- interaction profile、hard focus和predicted proposal schedule一致。

尤其不能把 `legacy` 单模块 checkpoint与 `cooperative` full checkpoint直接当成只改变 RTOR/A3开关的 2×2 factorial。

## 8. 数据划分与复现风险

1. `cfg.seed=7351`，engine 初始化 Python/NumPy/Torch 并启用 deterministic cuDNN。
2. 测试 list固定；可用 `P2P_TEST_LIMIT/INDICES`做子集，但 manifest必须写出确切 IDs。
3. 训练是动态 pair、动态 visibility/noise/rotation；公平多 seed训练必须保持相同 seed协议。
4. `DeterministicValidationDataset` 实际包装训练 dataset的前 `validation_size` 个 group，并非源码日志所写的“disjoint deformation groups”。训练 manifest中的该描述与实现不一致，不能据此声称验证集独立。
5. test loader若 checkpoint有 `neighbor_limits`则复用；否则会从随机训练样本重新校准，可能造成协议漂移。评估应要求 metadata中存在 limits并记录。

## 9. 面向因果验证的模块拆分

原目标中的 A0–A6 不能仅靠现有总开关完整实现，因为 `TopologyOverlapRefiner` 内部没有 topology attributes、Poincaré、cross-support 的配置开关。第一轮不得修改核心模型，因此建议采用只读 forward-hook/旁路重算干预：

| ID | 单一变化 | 实现边界 |
|---|---|---|
| R0 | baseline descriptor + 无 overlap权重 + 无 focus | 正常 baseline checkpoint/结构 |
| R1 | refined descriptor，uniform weights，无 focus | 捕获 RTOR输入/输出，旁路重算 coarse ranking |
| R2 | R1 + predicted soft overlap calibration | 同一 RTOR forward输出 |
| R3 | R2 + legacy source hard focus | 同一 checkpoint，仅应用现有 selector |
| R4 | oracle overlap soft calibration | 仅作为显式 oracle intervention |
| R5 | oracle coarse pairs | 固定 fine/Sinkhorn/LGR，显式 oracle |

topology graph only、geometry attributes、Poincaré edge distance等内部结构消融需要重新训练且当前无开关。若后续允许改模型，应另开第二阶段，以 checkpoint-incompatible 的受控 architecture实现；不能在第一轮偷偷 monkey-patch训练图并称为正式消融。

fine 模块首轮可做严格同 checkpoint、同 coarse candidates干预：

| ID | coarse inputs | fine refiner |
|---|---|---|
| F0 | 固定 predicted pairs/points/masks/node scores | OFF |
| F1 | 与 F0逐元素一致 | ON |
| F2 | 与 F0一致 | oracle fine assignment（显式上界） |

F0/F1 必须在同一 checkpoint 内切换，避免 independently trained backbone成为混杂；独立训练的 baseline/A3 checkpoint仅用于端到端 factorial，不用于声称 fine refiner的纯即时作用。

## 10. 两个科学假设

### H1：RTOR coarse-candidate preservation

在固定 coarse budget和相同 baseline descriptors候选全集下，RTOR 的局部拓扑 descriptor refinement与连续 overlap calibration会提高 GT-relevant、共同可见 superpoint pairs的排序，使更多“selection前可检索到”的 GT candidates在top-K后存活；其效果应首先出现在 MRR、GT query/pair recall、Candidate Survival和PIR，然后才可能传递到 LGR与RMS-TRE。

反证条件：descriptor/soft calibration不提高这些中间量，或增益只来自 score共同缩放；hard focus若降低 survival，则该 branch不支持 H1，即使个别最终 TRE变好也不能归因为 candidate preservation。

### H2：geometry-aware fine ambiguity reduction

在 coarse correspondence、patch points/masks、node scores完全固定时，geometry-aware bidirectional fine patch refiner会提高几何正确 point matches的相对置信度，降低错误 assignment或有效 assignment熵，并提高 fine precision/recall/IR、inlier count和降低 local correspondence error；若机制正确，收益应在低 top1-top2 gap、高重复几何或低 patch overlap的 hard patches更大。

反证条件：固定 coarse条件下仅最终 pose改变而 fine correspondence指标不按预期变化，或收益主要集中于 easy patches，则“减少局部几何歧义”的 claim不受支持。

## 11. Stage 1 判定

> Module 1 实际解决什么问题：在 complete-to-partial 的固定 coarse top-K预算下，通过拓扑/跨侧上下文更新描述子，并以预测双侧可见概率校准候选排序；legacy还尝试直接裁剪完整 source支持集。

> Module 2 实际解决什么问题：在已选 coarse patch pair内部，用刚体不变局部几何偏置引导双向 feature interaction，减少 descriptor-only point matching的局部歧义。

> 它们分别改变 pipeline的哪个环节：Module 1改变 GeoTransformer→SuperPointMatching及其候选输入；Module 2改变 coarse patch gathering→Sinkhorn之间的 fine feature/logit。

> 理论预期改善哪些中间指标：Module 1首先应改善 overlap分类、coarse descriptor/ranking和candidate survival；Module 2首先应改善固定候选下的fine ranking、assignment confidence/entropy、precision/recall/IR与local error。只有这些中间变量按预期改变后，才有资格把最终TRE变化归因于对应模块。

当前 Stage 1 的最终事实判断：

- 两模块均真实参与最终 prediction：**YES**。
- 存在只算 loss但不进入 registration的新增模块：**NO**。
- 存在默认关闭的关键训练分支：**YES，legacy predicted coarse exposure=0**。
- 存在训练/推理不一致：**YES，GT-vs-predicted proposals与hard-focus两项**。
- 存在直接 GT/marker推理泄漏：**未发现**。
- baseline与改进模型现有结果已被源码自动保证完全公平：**NO，必须用manifest逐项验证**。
- 当前证据足以宣称模块有效：**NO，源码只能证明作用路径，不能证明效果**。

## 12. 运行时核验记录

核验日期：2026-09-17；环境：`geo_v2`，Python 路径 `/home/yangx/miniconda3/envs/geo_v2/bin/python`，PyTorch `2.9.1+cu130`，CUDA 可用。

### 12.1 现有模块/协议测试

环境没有安装 `pytest`，因此使用测试文件已提供的 `unittest` discovery 入口运行：

```text
python -m unittest discover -s tests -p 'test_topology_overlap.py' -v
python -m unittest discover -s tests -p 'test_rtor_a3.py' -v
python -m unittest discover -s tests -p 'test_p2p_protocol.py' -v
python -m unittest discover -s tests -p 'test_cooperative_matching.py' -v
```

结果为 `5 + 3 + 3 + 6 = 17` 项全部通过。覆盖范围包括：

- RTOR 初始化中性、刚体不变性、uniform-weight等价性和soft calibration可改变ranking；
- overlap loss对logits可反传；
- fine refiner刚体不变性和invalid-point masking；
- flat overlap probability时hard selector回退到全support；
- 四种architecture确实移除相应模块参数；
- cooperative proposal schedule、预算、去重和soft-overlap profile配置。

这些测试证明局部实现契约，不证明数据集上的有效性、candidate survival或最终TRE改善。

### 12.2 历史四格 checkpoint兼容性

对外部历史目录 `/home/yangx/code/new_deform/RTORv6/output` 中四个 epoch-150 checkpoint，用当前 `make_cfg(architecture)` 构建模型并执行 `strict=True`：

| Architecture | epoch | iteration | 当前源码 strict load | checkpoint SHA256 |
|---|---:|---:|---|---|
| geotransformer | 150 | 196350 | OK | `833540a23837158a69e15696ee4f608f4cf29ee3973689c69ecc95019fe5a98c` |
| rtor_only | 150 | 196350 | OK | `52e23ff52417b8d5eaa5e93e36d4413f41edcc17b53c58d33e5d12087110e802` |
| a3_only | 150 | 196350 | OK | `d9cc0c0a9ca5bc70b4b539056db4712b9a092131e97ddcdf27a9e1e858f0c167` |
| rtor_a3 | 150 | 196350 | OK | `5f1854c1bb082205d95f4738203dbc94498201e8024a86ce8b7d5a8c3f1e9562` |

但四个 checkpoint 都只有 `epoch/iteration/model` 顶层字段，`metadata` 为空，因而不能从 checkpoint 自身证明训练协议。外部 run manifests均声明 seed 7351、scratch、150 epochs；其中 A3-only manifest存在额外 provenance异常：目录是 `scratch_v2`，其 `output_dir`和命令却写为`scratch_v22`，并记录两次resume history。该异常不证明结果错误，但在论文级四格比较前必须解释或重建可验证manifest。

### 12.3 当前数据协议指纹与分布

| 文件 | SHA256 |
|---|---|
| train `dict.json` | `4f598c2479fd5cee5253521797315b21de688cc008b29b978159ddb31c8d18cd` |
| in-silico test `list.npz` | `b457199148dfca5391c08602881b603bca644373b91afee7565f7ad0e53f2bc5` |
| in-silico `stat_svd.npz` | `07c2d8c5a0731bcd4f212c057f98fa230abef0d7e11fe9bf6fa453e6b45682bd` |
| in-vitro `rigid_list.npy` | `888a099e8433b13ba5d8f4e8e96c74e5b640a32ee8c3ec15c1554e1156d194e2` |
| in-vitro `stat.npz` | `5ddee334b6d30daab5b3842acba2ab94b2f9e7aa43e02685dda8891d73863200` |

当前 train list有1309个group；in-silico test有6315个唯一sample；in-vitro有800个唯一sample。in-silico visibility的 `[min,Q1,median,Q3,max]` 为 `[0.19969, 0.40644, 0.60231, 0.80207, 0.99978]`；deformation mm为 `[0.24180, 1.81927, 3.42013, 5.53949, 21.76987]`。因此后续困难度分层应优先使用这些真实分布的预注册quantiles，而不是沿用未经说明的任意阈值。
