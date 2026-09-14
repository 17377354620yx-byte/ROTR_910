# RTOR+A3 负交互诊断与待验证修改

## Material Passport

- 状态：训练侧机制诊断已执行；候选模型已完成训练/反向传播冒烟验证，完整训练后的精度尚未测定。
- 项目：`/home/yangx/code/new_deform/ai_worker/RTORv2`；环境：`geo_v2`。
- 证据：`output/interaction_diagnosis/`；本阶段新增诊断只读取训练数据和已有检查点/训练日志，没有使用最终测试 TRE 选择修改。
- 采样：预先固定 16 个训练组及增强随机种子；全模型 epoch 1/5/10、50/75/100、130/140/150，共 144 次结构/梯度回放。三个单模块对照各在 1/75/150 回放相同 16 例，共另 144 次结构回放。
- 限制：这是固定训练输入上的 checkpoint replay，不是原始每个训练 step 的梯度，也不是独立患者泛化试验。仅有 seed7351，不能证明任何方法全局最优。

## 结论

**归因更正：训练用 GT patch、测试用预测 patch 是 GeoTransformer/DFAT 共有设计，不能据此解释 RTOR+A3 特有的负交互。此前将其列为主因、据此推荐 cooperative 的证据不足，现撤回该结论。** 已观察到 RTOR 硬筛选减少 GT 支持，但尚未证明这解释了组合相对单模块的最终精度差异。没有证据支持“中晚期持续梯度对抗”或“RTOR overlap loss 过强”是主因。

原四组结果说明组合模型没有稳定超越单模块，但不能仅由单个 seed 的均值就宣布存在稳定、普遍的 negative interaction。本报告定位可重复的结构瓶颈，不把局部机制解释成所有 TRE 差异的唯一原因。

| confirmed problem / 结论 | supporting metric / figure | suspected mechanism | minimal experiment to verify | recommended fix |
|---|---|---|---|---|
| 硬筛选移除有效 GT 支持 | 晚期固定候选空间 MRR：RTOR 前 0.9540，描述子后 0.9581，软 overlap 加权后 0.9602，硬筛选后 0.8492；GT pair survival 约 81.95% | 描述子及软权重有效，但硬筛选丢失的候选，后续 A3 无法恢复 | 同权重、同训练输入，只关闭硬筛选：GT survival 81.92%→100%，PIR 82.10%→83.20% | 保留 RTOR 描述子和软权重，取消硬筛选 |
| 共同设计下的候选组成差异（不是已确认问题） | 16 例：GT 训练 patch 完全不匹配比例为 0；原预测 patch 为 15.01%；fine loss 1.1246→1.2258 | GeoTransformer/DFAT 同样采用此设计；这组比较没有隔离 RTOR 的额外影响 | 相同权重、输入、dropout 关闭，切换 GT / predicted proposals，记录标签组成与 fine loss | 混合预测训练仅保留为实验假设，不作为已证实修复 |
| 单独取消硬筛选不足以证明整体提升 | 同例 soft-only 的 fine MRR 0.6151，原为 0.6178；GT coverage/PIR 增加，但新候选并未自动变得更易匹配 | 候选分布改变；是否需要训练适应尚无因果证据 | soft-only 与 soft+mixed-exposure 的训练消融 | 单独验证 hard-focus 开关，不能由此推荐组合训练策略 |
| 没有独立 A3 auxiliary loss | `OverallLoss` 实际为 `c_loss + f_loss + 0.5*o_loss` | A3 与 fine matching 共用目标；把全部 fine loss 叫 A3 loss 会错误归因 | 原目标梯度与 `grad(f_on - f_off)` 同时记录 | 保留真实 loss 定义，不增加虚构的 A3 loss |
| fine loss 不直接训练 RTOR/coarse selection | 参数级 autograd 路径记录：fine→RTOR/Transformer 为 NO_GRADIENT_PATH | 离散选择与 GT target 替换阻断直接协同，仅有 shared backbone 的间接联系 | 分模块梯度路径检查；无路径记 NA，不能记 cosine=0 | 该路径也属于常见离散 coarse-to-fine 设计；先隔离特有增量影响 |
| 未发现持续系统性梯度冲突 | 中/晚期 overlap-fine 平均 cosine 0.1395/0.1386；负值比例 14.58%/12.50% | 少量局部冲突存在，但不足以解释成长期对抗 | 9 checkpoints × 16 固定输入；额外 A3 增量梯度 | 不默认使用 PCGrad、detach 或双骨干 |
| 未发现重复软权重是主要瓶颈 | 晚期软权重使粗 MRR 0.9581→0.9602；A3 使 fine MRR 0.6041→0.6180 | 粗、细模块作用尺度不同；相似“geometry”命名不等于信息冗余 | 分阶段 GT rank、正负直方图、对齐 patch 输出相关性 | 保留软 overlap 与 A3，不直接删除一个模块 |
| 无充分证据认为 overlap loss 监督过强 | 晚期 overlap 梯度范数约 0.2446，乘 0.5 后远小于 fine 2.6331；局部 loss-weight 实验差异很小 | 原始 loss 数值不可直接当成有效梯度强度 | 6 组权重、3 对训练样本的一步 shared-backbone 干预 | 保留原权重 1:1:0.5 |

## DFAT 纠正后的增量对照

执行 `conda run --no-capture-output -n geo_v2 python -m diagnostics.interaction.factorial_candidates`，使用同一 full epoch-150 检查点、相同 16 个固定训练输入，在 eval 模式切换 RTOR × A3；无新训练或最终测试集评估。

| 推理开关 | Sinkhorn 后 fine MRR | 完全无 GT 匹配 patch 比例 |
|---|---:|---:|
| RTOR off / A3 off | 0.54405 | 14.038% |
| RTOR off / A3 on | 0.61465 | 14.038% |
| RTOR on / A3 off | 0.53853 | 15.015% |
| RTOR on / A3 on | 0.62025 | 15.015% |

A3 增益在 RTOR off 时为 +0.07059，on 时为 +0.08171；差分交互项均值为 **+0.01112**，16 例中 4 例为负。该观察不支持当前检查点上 RTOR 普遍削弱 A3 的候选条件增益。不同 RTOR 条件的候选集合不同，因此这不是统一点集上的配对 MRR，也不能替代四组独立训练或最终刚体变换精度。

进一步强制使用 full 模型的相同 patch 对应，关闭 RTOR 后，16 例 `matching_scores` 最大绝对差均为 **0**。这确认当前推理图中 RTOR 不直接改变固定 patch 的 A3/fine 输出；共享 backbone 的训练间接影响仍未由此排除。固定候选实验没有用于比较最终位姿（coarse score 被固定为 1）。

原始记录见 `output/interaction_diagnosis/factorial_candidates/records.jsonl` 和 `summary.json`。目前不能确认组合负协同的主因；应继续区分独立训练后的 fine 特征变化、对应空间分布与位姿估计，以及单 seed 波动。混合预测训练不再作为已验证的修复建议。

## 梯度与 loss 的具体定义

`c_loss` 为 coarse matching；`f_loss` 为经过 A3 的 fine matching 路径，并非 A3 独立辅助项；`o_loss` 为 RTOR overlap。
`a3_increment` 是同一 GT patch、同一 backbone feature 上 `∇f_on - ∇f_off`，只用于归因，不是新训练目标。
计算 cosine 使用共同的完整参数坐标系；不连通参数补零，整个模块没有梯度路径时标记 `NO_GRADIENT_PATH`。

| shared backbone 梯度对 | early 平均 cosine | middle | late | late 负值比例 |
|---|---:|---:|---:|---:|
| overlap / fine-through-A3 | 0.0027 | 0.1395 | 0.1386 | 12.50% |
| overlap / coarse matching | 0.2223 | 0.4232 | 0.4431 | 0% |
| fine-through-A3 / coarse matching | 0.1789 | 0.2885 | 0.2859 | 0% |
| overlap / A3 increment | 0.0106 | 0.1018 | 0.0993 | 12.50% |

局部 loss-weight 试验使用相同范数的 shared-backbone SGD 更新，权重为 `(1,1,0/.1/.5/1)`、`(1,.5,.5)`、`(1,2,.5)`，实际精确值见 `loss_weights/results.json`。仅为一步局部敏感性，不是完整 Adam 重训结果。原权重与减小 overlap 权重的 held training-pair 平均 loss 约为 1.99516 和 1.99518，没有支持激进减权的证据。

历史四组日志共解析 1200 条 epoch train/val 汇总。原日志没有训练期体积标志点 TRE，因此没有伪造 TRE 学习曲线；历史 RMSE 实际为 mean normalized source displacement。曲线使用原日志记录的 PIR、IR、RRE 和该位移指标。

## 指标与图

全部原始数值在 `output/interaction_diagnosis/full_phases/records.jsonl`、`single_module_controls/records.jsonl`，带数据索引、增强种子、检查点哈希和执行代码哈希。

- [loss 与匹配指标曲线](../../output/interaction_diagnosis/report/loss_metrics.png)
- [三阶段梯度 cosine 与负值比例](../../output/interaction_diagnosis/report/gradient_conflicts.png)
- [RTOR / A3 前后 GT rank](../../output/interaction_diagnosis/report/gt_ranks.png)
- [positive / negative score histogram](../../output/interaction_diagnosis/report/score_histograms.png)
- [feature norm 与 residual norm](../../output/interaction_diagnosis/report/feature_residuals.png)
- [overlap / descriptor / A3 对齐相关性](../../output/interaction_diagnosis/report/output_correlations.png)
- score entropy、GT-query 数目和 mask-aware rank 同时记录在原始 JSONL 与 `report/summary.json`。

粗、细输出不在同一网格：RTOR/A3 相关性按同一个预测 patch pair 对齐标量置信度和 A3 logit 增量计算，不能把不同形状的 tensor 直接展开计算相关。coarse-after-focus rank 是固定候选空间的 survival-aware 指标，被删除的 GT 对应按 miss 计；不同 candidate set 上的 fine MRR 不是完全配对的点级比较。

## 保留的实验候选（尚非推荐最优版本）

`--interaction_profile cooperative` 为显式启用的实验候选；默认 `legacy` 保持不变。其实现为：

1. 一个原有 KPConvFPN backbone；保留 GeoTransformer、RTOR、A3、Sinkhorn 和原 LGR。
2. RTOR overlap 以原软权重进入 coarse matching；保留所有有效 source superpoints。
3. epoch ≤5 保持 GT fine-training proposals；epoch 5→20 将预测候选比例线性提高到 25%，之后保持。fine patch 总预算保持原 GT budget，并去重。
4. loss 仍为 `c_loss + f_loss + 0.5*o_loss`；LGR acceptance radius 仍为 0.1，没有引入新几何阈值调参。
5. 按训练侧 validation mean source displacement 保存 `best.pth.tar`；测试必须显式使用同一 interaction profile。

25% 是保守的预设暴露强度，不是依据完整测试 TRE 搜索的最优数值。它保留 75% 的 GT 监督，给 A3 提供真实 predicted-patch 训练输入。必须经过用户接下来的完整训练确认最终精度，不能把当前冒烟结果称为性能提升。

本版本没有采用此前探索的双编码器、ICP、较紧 LGR 阈值或新辅助 loss。`output/diagnosis_v1` 为方向修正之前的探索记录，不作为本次模型选择的独立验证证据。

## 近年方法的适用性

- [CAST，NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/821a6e5681b072351fd3c21fac44739a-Abstract-Conference.html)：强调 coarse correspondence 的一致性以及避免不相关区域干扰 fine matching。对本项目的启发是先修复 coarse-to-fine 接口，不只是增加 fine attention。本实现没有复制 CAST 或声称是它的复现。
- [Fully-Geometric Cross-Attention，3DV 2025](https://arxiv.org/abs/2502.08285)：用 Gromov–Wasserstein 结构处理跨点云几何，在 fine 层聚合局部结构。它支持区分 coarse 与 fine 的几何任务；本项目尚无证据需要再增加一套几何度量。
- [FUSER，CVPR 2026，作者仓库](https://github.com/Jiang-HB/FUSER)：面向多视角联合位姿与 SE(3)^N refinement；P2P 的输入为一对 complete/partial liver clouds，直接替换为多视角大模型缺乏任务匹配依据。
- [DFAT 官方实现](https://github.com/fukexue/DFAT/blob/main/experiments/3DMatch/model.py#L177-L214)：先生成预测 coarse 对应，在 `self.training` 分支用 `coarse_target` 的 GT 对应替换，然后提取 patch 并进行局部 Transformer 和 Sinkhorn。与本项目共享的 GT/predicted 设计不能作为特有负协同的因果解释。

这是针对当前任务的代表性方法筛选，不宣称穷尽所有优秀模型；目前没有任何文献能替代本项目的受控诊断来证明“最优”。

## 统计解释边界：11/11 检查

Simpson：按阶段及模块分解；ecological：不推断独立患者；Berkson：训练组预先固定而非按失败挑选；collider：不按最终 TRE 筛样本；base rate：报告 GT-query/positive 比例；regression-to-mean：同权重同输入；survivorship：完成全部预设回放；look-elsewhere：保留全部干预；forking paths：保留历史探索及 manifest；correlation/causation：cosine 和相关性仅作诊断，因果判断限于受控开关；reverse causality：采用训练图和干预确定直接路径。没有多 seed/患者级显著性结论。

## 验证

`tests.log`：22 项测试通过。`cooperative_smoke.log`：实际完成 1 epoch、2 个训练样本和 2 个 validation 样本，保存 best/snapshot；这仅验证训练流程。`cooperative_contract.json`：25% mixed branch 的 128 个 patch 完成反向传播，RTOR/A3 梯度均非零且有限；移除 GT 后推理变换最大差为 0。

最终训练/测试命令见项目根目录 `TRAIN_TEST.md`。
