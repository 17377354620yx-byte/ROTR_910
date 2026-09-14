# RTORv2 相对 GeoTransformer 的论文级模型审计

> 审计对象：`experiments/geotransformer.p2p_liver`  
> 任务：腹腔镜肝脏 complete-to-partial 刚性点云配准  
> 审计日期：2026-09-12  
> 原则：只使用现有代码、检查点、逐病例结果和诊断记录；未补训、未新增测试病例。

## 1. 一句话结论

RTORv2 不是对 GeoTransformer 主干的替换，而是在原有 **KPConvFPN → Geometric Transformer → coarse matching → patch matching/Sinkhorn → LGR** 流程上加入两个轻量残差模块：RTOR 在粗层显式估计双侧可见重叠并重标定候选，A3 在细层用刚体不变局部几何引导双向 patch cross-attention；cooperative 版本进一步取消有害的源点云硬裁剪，并在训练期最多混入 25% 预测 coarse proposals。

审计证据支持下列收敛结论：

1. **问题建模是合理的**：complete-to-partial 的关键困难确实包括非重叠区干扰、低可见率下的重复局部几何，以及 coarse 错误传递到 fine/LGR。
2. **A3 的机制证据最强**：固定候选时，A3 将 fine MRR 从 0.54405 提至 0.61465（RTOR 关闭）或从 0.53853 提至 0.62025（RTOR 开启）；独立训练的 A3-only 在无噪声 in-silico 上也将平均 RMS-TRE 降低 0.0325 mm。
3. **RTOR 应拆成“软重加权”和“硬筛选”讨论**：描述子细化与软 overlap 权重使粗匹配 MRR 从 0.9540 依次升至 0.9581、0.9602；但旧版硬筛选使其跌至 0.8492，并仅保留约 81.95% 的 GT pair。不能把整个 RTOR 简化为“有效”或“无效”。
4. **cooperative 修改修复了结构性支持集损失，但没有形成稳定的大幅终点增益**：相对旧 RTOR+A3，它在无噪声和 2 mm 条件分别改善 0.0300 和 0.0401 mm，在 4 mm 下基本不变；相对原始 GeoTransformer，四个条件的平均差仅为 −0.0117、−0.0232、−0.0066 和 −0.2804 mm，前三个置信区间均跨 0。
5. **不能声称 RTOR+A3 存在已证实协同增益**：RTOR×A3 的逐病例差分中的差分在四个条件下 95% CI 均跨 0，点估计反而为次加性方向。
6. **当前最可信的论文定位不是 SOTA 宣称，而是可解释的任务特异性改造与边界分析**：现有证据能够证明“改了什么、针对什么瓶颈、模块在哪一层起作用、旧设计为何失效”；不能证明“普遍优于 GeoTransformer”或“具有临床泛化能力”。

## 2. 原始 GeoTransformer 与 RTORv2 的结构差异

原始 GeoTransformer 通过旋转/平移不变的点对距离与三元组角度编码构建 superpoint 几何上下文，再经 coarse-to-fine 匹配和 Local-to-Global Registration 估计刚体变换。方法来源为 [GeoTransformer, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Qin_GeoTransformer_Fast_and_Robust_Point_Cloud_Registration_With_Geometric_Transformer_CVPR_2022_paper.html)。本项目保留了这一主体链路及单刚体输出，核心入口见 `geotransformer/modules/liver/registration_model.py:42-87`。

| 层级 | GeoTransformer 基线 | RTORv2 改动 | 直接作用对象 | 设计动机 |
|---|---|---|---|---|
| backbone | 单 KPConvFPN | 保持单骨干 | 多尺度点特征 | 避免把增益混入额外 encoder 容量 |
| coarse context | Geometric Transformer | 保留 | superpoint descriptor | 延续全局刚体不变几何上下文 |
| coarse refinement | 无独立肝脏 overlap 模块 | 加入 RTOR | 双侧粗描述子与 overlap probability | 抑制完整肝脏中不可见区域，强调可被腹腔镜观察到的支持集 |
| coarse selection | 描述子相似度 | RTOR overlap soft weight；legacy 另有 source hard focus | 候选排序与候选支持集 | 将“可能可见”作为配准先验 |
| fine matching | patch 特征点积 → Sinkhorn | Sinkhorn 前加入 A3 | 已选 patch 内逐点特征 | 消解低重叠、相似肝表面局部结构造成的细匹配歧义 |
| training interface | GT coarse proposals 驱动 fine supervision | cooperative 在 epoch 5–20 线性混入预测 proposals，最高 25% | fine 训练输入分布 | 让 A3 接触部分真实预测候选，同时保留 75% GT 监督 |
| estimator/loss | Sinkhorn + LGR；coarse/fine loss | 保持原 LGR；总损失为 `c_loss + f_loss + 0.5 o_loss` | 最终位姿与训练目标 | 隔离模块贡献，不靠后处理阈值或新增优化器制造增益 |

### 2.1 RTOR：粗层拓扑—重叠细化

RTOR 对每侧 superpoint 建立 kNN 图，提取归一化邻域距离、曲率、线性度、平面度和 Poincaré 距离作为边几何，随后执行同侧图消息更新与双侧 cross-support attention，再同时输出残差描述子与双侧 overlap logits（`topology_overlap.py:42-76, 80-169, 186-234`）。

其关键工程约束是：输出投影和 overlap head 末层均零初始化，因此训练开始时复制原描述子并给出均匀 overlap prior，而不是立即破坏预训练 GeoTransformer 表示。概率被映射成带 0.05 floor 的连续权重进入 coarse matching（`registration_model.py:145-172, 312-321`）。

legacy profile 在推理时还根据 overlap 概率硬裁剪完整 source，最多保留 50% superpoints（`registration_model.py:282-294`）。cooperative profile 明确关闭该硬筛选（`config.py:81-87`）。

### 2.2 A3：细层几何感知局部交互

A3 位于候选 patch 提取之后、相似度矩阵与 Sinkhorn 之前（`registration_model.py:342-390`）。每个 patch 通过中心化和尺度归一化计算三维几何签名：径向距离、平均邻点距离及其标准差；由跨 patch 签名差形成 attention bias，再以同一组预更新特征完成双向 cross-attention（`fine_local_refiner.py:110-116, 171-215, 234-250`）。残差尺度初始化为 0，保证初始行为兼容基线。

### 2.3 Cooperative：修复 coarse-to-fine 接口，而非新增第三个网络

cooperative 保持 RTOR、A3、单骨干、Sinkhorn、原 LGR 和原损失，只做两项接口调整：

- 关闭 source hard focus，保留全部有效 coarse support；
- epoch 5 到 20 将预测 proposals 比例线性升至 25%，且维持原 GT patch 总预算并去重（`config.py:76-87`; `registration_model.py:332-340`）。

因此 cooperative 与 legacy RTOR+A3 参数量完全相同，性能差只能归因于交互策略与独立训练所得权重，而非模型容量增加。

## 3. 参数代价

| 架构 | 参数量 | 相对 GeoTransformer 增量 |
|---|---:|---:|
| GeoTransformer | 9,829,377 | — |
| RTOR-only | 10,094,595 | +265,218（+2.70%） |
| A3-only | 10,355,970 | +526,593（+5.36%） |
| RTOR+A3 / cooperative | 10,621,188 | +791,811（+8.06%） |

参数量由当前配置直接实例化计算。项目没有保存统一硬件上的完整 latency/显存对照，因此不应补写运行时间优势。

## 4. 改动是否对应 P2P 肝脏配准的真实瓶颈

本任务不是普通 partial-to-partial 刚体配准：source 是完整肝脏，target 是变形后、受视野限制的表面片段；训练裁剪可见率为 0.18–1.0，并向 target 添加 0–5 mm 噪声（`dataset.py:75-138`）。P2P 原论文指出腹腔镜可见表面通常很少，并以 patches-to-partial 候选和变换筛选处理 complete-to-partial 歧义：[Resolving the Ambiguity of Complete-to-Partial Point Cloud Registration..., IEEE JBHI, DOI 10.1109/JBHI.2025.3583875](https://doi.org/10.1109/JBHI.2025.3583875)。

| 真实瓶颈 | RTORv2 假设 | 审计判定 |
|---|---|---|
| 完整 source 中大量区域在 target 不可见 | RTOR 学习双侧可见概率，降低非重叠 superpoint 权重 | **部分成立**：软权重改善 coarse MRR；硬裁剪会误删真匹配 |
| 肝表面平滑、局部形态重复，低可见时 patch 易歧义 | A3 用局部刚体不变几何引导细粒度双向注意力 | **成立且证据最强**：固定候选 fine MRR 稳定提升约 0.071–0.082 |
| coarse 错误一旦被离散选择，fine 阶段无法恢复 | cooperative 保留完整支持并让 fine 训练接触少量预测候选 | **结构上成立，终点效果有限**：修复 support survival；总体 RMS-TRE 仅小幅变化 |
| target 含非刚性肝变形，但模型输出单刚体 | marker TRE 衡量最终导航误差；RRE/RTE 仅作诊断 | **必须明确边界**：不存在唯一物理“真刚体位姿”，不可用 RRE/RTE 替代临床误差 |

从计算图看，fine proposal 选择位于 `torch.no_grad()` 内；fine loss 不会直接穿过离散 selector 训练 RTOR/coarse Transformer。二者只能通过共享 backbone 产生间接耦合。这解释了为何“两个模块各自合理”并不自动得到端到端协同。

## 5. 数据、终点与统计口径

### 5.1 现有比较

- 五个模型均使用 seed 7351、训练 150 epochs；比较 GeoTransformer、RTOR-only、A3-only、legacy RTOR+A3 和 cooperative RTOR+A3。
- in-silico：发布列表中的 6,315 个相同病例，分别评估 0、2、4 mm 附加噪声。
- in-vitro：800 个相同病例，无附加合成噪声。
- 逐病例 `sample_name` 在五种方法间完全一致，因此采用配对差值。
- 主终点为体积标志点 RMS-TRE（mm），定义见 `test.py:147-155`；SR@20 mm 仅作为成功率补充。
- 配对效应定义为 `candidate − GeoTransformer`，负值表示改善；报告 10,000 次 bootstrap 的均值差 95% CI、Wilcoxon signed-rank、Holm 校正和逐病例 win rate。

注意：代码字段 `coarse_candidate_recall` 实际是“每例是否至少有一个预测 coarse pair 命中任一 GT pair”的二元比例（`test.py:159-171, 255-257`），不是完整 recall，接近 1 不能证明 coarse correspondence 已解决。

### 5.2 主要终点结果

| 条件 | GeoTransformer | RTOR | A3 | RTOR+A3 | RTOR+A3 cooperative |
|---|---:|---:|---:|---:|---:|
| in-silico, 0 mm（n=6315） | 4.440±3.012 | 4.472±2.911 | **4.407±2.971** | 4.458±3.154 | 4.428±2.907 |
| in-silico, 2 mm（n=6315） | 4.490±3.563 | 4.502±3.175 | **4.451±3.070** | 4.507±3.357 | 4.467±2.927 |
| in-silico, 4 mm（n=6315） | 4.738±4.114 | 4.727±3.683 | **4.698±3.730** | 4.732±3.730 | 4.732±4.158 |
| in-vitro（n=800） | 5.253±4.714 | **4.939±2.597** | 5.253±4.553 | 5.034±3.536 | 4.972±2.635 |

数值为 mean±SD RMS-TRE（mm）。粗体只表示该行五种模型中的最低均值，不等于跨 seed 或临床显著性。

### 5.3 相对 GeoTransformer 的配对效应

| 候选 | 0 mm Δ [95% CI] | 2 mm Δ [95% CI] | 4 mm Δ [95% CI] | in-vitro Δ [95% CI] |
|---|---:|---:|---:|---:|
| RTOR | +0.0321 [0.0127, 0.0494] | +0.0119 [−0.0502, 0.0656] | −0.0117 [−0.0906, 0.0621] | **−0.3135 [−0.5915, −0.1213]** |
| A3 | **−0.0325 [−0.0471, −0.0197]** | −0.0390 [−0.0945, 0.0037] | −0.0403 [−0.1054, 0.0186] | +0.0002 [−0.2830, 0.2735] |
| RTOR+A3 | +0.0183 [−0.0089, 0.0575] | +0.0169 [−0.0483, 0.0807] | −0.0066 [−0.0797, 0.0638] | **−0.2183 [−0.4333, −0.1001]** |
| cooperative | −0.0117 [−0.0289, 0.0038] | −0.0232 [−0.0798, 0.0193] | −0.0066 [−0.0871, 0.0776] | **−0.2804 [−0.5694, −0.0787]** |

这里最重要的 Reviewer 解释是：

- in-silico 的绝对效应多小于 0.05 mm，远小于病例间 SD；除 A3 的无噪声结果外，均值差 CI 多跨 0。
- in-vitro 的 RTOR/cooperative 均值改善主要来自降低少数大误差尾部：cooperative 的平均差为 −0.280 mm，但中位差为 +0.006 mm、win rate 仅 49.1%。因此应写成“减少平均/尾部误差”，不能写成“多数病例均改善”。
- 大样本下 Wilcoxon 可对极小的秩分布差异给出很小 p 值，并可能与均值差 CI 结论不同。论文应优先报告效应量、CI 和分布图，而不是只报 p 值。
- SR@20 mm 在所有模型均约 99.5%–100%，存在明显天花板效应，不适合作为主要优越性证据。

### 5.4 低可见率结果

在无噪声 in-silico 的 visibility `[0.2,0.3)` 子组（n=730），GeoTransformer、RTOR、A3、legacy full、cooperative 的 RMS-TRE 分别为 5.492、5.276、5.396、5.403 和 **5.251 mm**。cooperative 相对基线低 0.241 mm，方向上符合“低重叠支持集建模”的设计目标，但这是预定义可见率分层的描述性结果，当前表未给患者/变形组层级 CI，不应单独宣称显著。

in-vitro 同一可见率分层（n=105）中，基线为 8.650±10.787 mm，RTOR 和 cooperative 分别为 6.866±3.402 与 6.858±3.308 mm；均值和 SD 同时下降，支持其可能主要抑制 catastrophic failures，而非均匀改善所有病例。

### 5.5 Cooperative 相对 legacy full

| 条件 | cooperative − legacy RTOR+A3，mm [95% CI] | win rate |
|---|---:|---:|
| in-silico, 0 mm | −0.0300 [−0.0659, −0.0085] | 53.8% |
| in-silico, 2 mm | −0.0401 [−0.0880, −0.0064] | 53.6% |
| in-silico, 4 mm | +0.00003 [−0.0531, 0.0639] | 53.6% |
| in-vitro | −0.0621 [−0.2523, 0.0611] | 48.1% |

这说明 cooperative 对旧 full 的修正并非完全无效，但收益仍小、对高噪声和 in-vitro 不稳定。由于 cooperative 同时改变了 hard-focus 和 proposal exposure，现有完整训练结果不能把终点差异进一步拆成两者各自的因果贡献。

## 6. 模块机制证据与交互审计

### 6.1 RTOR 的真正瓶颈是 hard focus

晚期固定输入诊断得到：

| 阶段 | GT coarse MRR |
|---|---:|
| RTOR 前 | 0.9540 |
| 描述子 refinement 后 | 0.9581 |
| soft overlap weighting 后 | 0.9602 |
| source hard focus 后 | 0.8492 |

hard focus 后 GT pair survival 约为 81.95%。在同病例关闭硬筛选，survival 从 81.92% 回到 100%，PIR 从 82.10% 升至 83.20%；但 fine MRR 从 0.6178 轻微降至 0.6151。这证明硬筛选会损害候选覆盖，却也说明“恢复更多 GT 候选”不必然立即转化为更优 fine match 或最终位姿。

因此论文中应把 RTOR 的贡献表述为 **continuous overlap-aware calibration**，把 hard pruning 作为经审计后删除的失败设计，而不是将两者混称为 overlap selection。

### 6.2 A3 的局部作用得到直接支持

在同一 full epoch-150 检查点、固定 16 个训练输入上做 RTOR×A3 推理开关：

| RTOR | A3 | Sinkhorn 后 fine MRR | 无 GT 匹配 patch 比例 |
|---|---|---:|---:|
| off | off | 0.54405 | 14.038% |
| off | on | 0.61465 | 14.038% |
| on | off | 0.53853 | 15.015% |
| on | on | 0.62025 | 15.015% |

A3 增益分别为 +0.07059 和 +0.08171。强制使用相同 patch 对应后，关闭 RTOR 时 matching scores 最大绝对差为 0，说明当前推理图中 RTOR 不会直接改变固定 patch 内的 A3 输出；二者的联系主要通过候选构成和共享 backbone 的训练历史产生。

### 6.3 没有证据支持“稳定负交互”或“梯度冲突”

基于五个独立训练模型的 RTOR×A3 终点 factorial interaction，定义为 `(full−RTOR)−(A3−baseline)`；正值为次加性/拮抗方向：

| 条件 | interaction，mm [95% CI] |
|---|---:|
| in-silico, 0 mm | +0.0187 [−0.0092, 0.0579] |
| in-silico, 2 mm | +0.0440 [−0.0165, 0.1143] |
| in-silico, 4 mm | +0.0454 [−0.0494, 0.1429] |
| in-vitro | +0.0949 [−0.1975, 0.4127] |

所有 CI 均跨 0，故既不能声称 synergy，也不能声称已证实 antagonism。训练回放同样未发现持续梯度对抗：晚期 overlap/fine、overlap/coarse、fine/coarse 平均 cosine 分别为 0.1386、0.4431、0.2859，负值比例为 12.5%、0%、0%；overlap 梯度范数约 0.2446，乘 0.5 后远小于 fine 的 2.6331，不支持“overlap loss 压制其他任务”的解释。

## 7. 外部有效性：DePoLL 是负面证据

在 13 例 DePoLL 外部数据上，当前 cooperative 检查点的 intraoperative clips/balls 平均 TRE 为 76.66±54.27 / 87.68±65.90 mm，preoperative 为 101.86±36.33 / 101.45±35.41 mm；完全重复评估得到相同 summary，最大 transform 差为 0。对应论文报告的本方法量级为 11.72/13.10 mm 和 39.47/44.38 mm（本地论文：[Liver point cloud registration via multi-feature fusion and hyperbolic..., DOI 10.1016/j.cmpb.2026.109451](https://doi.org/10.1016/j.cmpb.2026.109451)）。

这不是可用来证明优越性的结果，而是明确的 domain-shift failure：当前训练分布、尺度/采样特征或 complete-to-partial 生成机制没有迁移到该外部协议。论文如包含此结果，必须作为外部泛化限制或失败分析呈现。

## 8. Reviewer 级风险清单

1. **单随机种子**：全部完整比较仅 seed 7351。6,315 个 test pairs 提供的是病例配对不确定性，不等于训练随机性不确定性。
2. **验证集不独立**：`DeterministicValidationDataset` 直接包装训练 dataset 的前 100 个索引（`dataset.py:379-388`）。它固定增强随机数，但不是独立 deformation groups。若用其选 best checkpoint，会有模型选择偏倚风险。
3. **测试单元可能非独立患者**：同一形变组/肝脏可产生多个 pair。普通逐病例 bootstrap 可能低估 cluster-level 不确定性；现有 CI 只能按 pair 解释。
4. **发布样本数差异**：本地固定列表为 6,315，而 P2P 论文描述为 6,050。必须在 Methods 中明确“按发布列表评估”，并报告列表哈希，不能直接复述论文样本数。
5. **指标命名风险**：训练日志中的 `RMSE` 实际是归一化 source mean displacement，不是 marker RMS-TRE，不能将历史曲线标成 TRE learning curve。
6. **oracle transform 的角色**：marker-paired least-squares transform 只用于 RRE/RTE 等诊断和 GT correspondence 生成；推理估计不读取 marker。应明确区分输入与评估信息。
7. **非刚性目标与刚体输出**：target 已发生形变，least-squares rigid transform 只是 oracle approximation，不是唯一真实手术位姿；最终导航意义应由 marker TRE 承担。
8. **成功率天花板**：SR@20 mm 几乎饱和，无法支撑临床优越性。
9. **多重比较**：已有表提供 Holm 校正，但论文主结果仍应预先限定一个 primary endpoint 和少量关键 contrasts。
10. **外部泛化失败**：DePoLL 结果阻止任何“跨中心/跨设备稳健”的表述。

## 9. 可以写与不能写的 SCI 结论

### 可以写

- RTORv2 以 8.06% 参数增量，在 coarse 和 fine 两个不同尺度加入任务特异性建模，同时保留 GeoTransformer 与 LGR 主体。
- RTOR 的软 overlap calibration 改善 coarse ranking；硬 source pruning 会显著损害 GT support survival，cooperative 设计据此将其移除。
- A3 在固定 coarse candidates 上显著提高 fine-level GT ranking，是当前最直接、最可重复的模块机制证据。
- 在现有单 seed 评估中，收益主要表现为低可见率与 in-vitro 大误差尾部的下降，而非所有条件下的平均一致提升。
- 完整 RTOR+A3 未显示统计可确认的 factorial synergy。

### 不能写

- “RTORv2 在所有数据集和噪声水平显著优于 GeoTransformer”。
- “RTOR 与 A3 具有显著正协同”或“负交互已被彻底解决”。
- “coarse candidate recall=1 证明所有真对应均被召回”。
- “SR@20 mm 接近 100% 证明临床可用”。
- “DePoLL 验证了跨域泛化”。
- “6,315 个病例等价于 6,315 个独立患者”或“单 seed CI 覆盖训练随机性”。

## 10. 推荐论文叙事

建议题目：

> **Overlap-Calibrated Coarse-to-Fine Registration for Complete-to-Partial Laparoscopic Liver Point Clouds: A Mechanism-Aware Audit of GeoTransformer**

建议贡献写成三点：

1. 提出 coarse-level RTOR，以局部拓扑、双侧 cross-support 和连续 overlap calibration 建模完整肝脏与局部腹腔镜表面的非对称可见支持集。
2. 提出 fine-level A3，以中心化、尺度归一化的刚体不变局部几何约束 patch cross-attention，针对低可见率下的局部形态歧义。
3. 给出机制导向的 coarse-to-fine 审计，证明 hard pruning 会破坏真实支持集，区分模块内 ranking 增益与最终 marker TRE，并公开失败的外部泛化结果。

### 可直接用于英文 Results 的谨慎版本

> Across 6,315 paired in-silico registrations, the cooperative RTOR+A3 model achieved mean RMS-TREs of 4.428, 4.467, and 4.732 mm under 0, 2, and 4 mm added noise, respectively, compared with 4.440, 4.490, and 4.738 mm for GeoTransformer. The paired mean differences were small and their 95% bootstrap confidence intervals included zero in all three conditions. On 800 in-vitro registrations, cooperative RTOR+A3 reduced the mean RMS-TRE by 0.280 mm (95% CI, 0.079–0.569 mm), although its median case-wise difference was +0.006 mm and its win rate was 49.1%, indicating that the mean improvement was driven primarily by fewer large-error cases rather than uniform per-case gains.

> Mechanistic analysis localized the clearest benefit to fine matching. With fixed coarse candidates, enabling A3 increased the post-Sinkhorn fine-level mean reciprocal rank from 0.544 to 0.615 without RTOR and from 0.539 to 0.620 with RTOR. In contrast, the RTOR descriptor update and soft overlap weighting modestly improved coarse-level ranking, whereas hard source filtering reduced the coarse MRR from 0.960 to 0.849 and retained only approximately 82% of ground-truth pairs. These findings motivated removal of hard filtering in the cooperative profile.

> We found no statistically supported factorial synergy between RTOR and A3 at the final registration endpoint. The interaction confidence intervals crossed zero under all in-silico noise levels and in-vitro evaluation. Therefore, the two modules should be interpreted as addressing complementary stages of the pipeline rather than as demonstrating a proven super-additive effect.

## 11. 建议图表及其论证功能

| 编号 | 图/表 | 论文中回答的问题 |
|---|---|---|
| Fig. 1 | RTORv2 pipeline schematic（需按本报告结构绘制） | 相对 GeoTransformer 到底加在什么位置 |
| Fig. 2 | `figures/performance_overview.pdf` | 五模型四条件总体性能与离散程度 |
| Fig. 3 | `figures/paired_effects.pdf` | 相对 GeoTransformer 的配对均值效应与 95% CI |
| Fig. 4 | `figures/visibility_curves.pdf` | 改动是否真正针对低可见率 |
| Fig. 5 | `figures/case_level_changes.pdf` | 平均改善是否来自少数大失败病例 |
| Fig. 6 | `figures/factorial_interactions.pdf` | RTOR×A3 是否存在可确认协同 |
| Fig. 7 | `../../output/interaction_diagnosis/report/gt_ranks.png` | soft RTOR 与 hard pruning 的阶段性影响 |
| Fig. 8 | `../../output/interaction_diagnosis/report/gradient_conflicts.png` | 排除持续梯度冲突解释 |
| Table 1 | 架构改动与参数量 | 改了什么、计算代价多大 |
| Table 2 | 四条件 RMS-TRE | 最终效果 |
| Table 3 | 配对效应、CI、win rate | 统计与实际意义 |
| Table 4 | 机制与 factorial ablation | 为什么有效或无效 |

病例级可视化必须同时展示 improvement 和 deterioration，而不能只挑成功案例。推荐固定展示：低可见率最大改善、低可见率最大恶化、中位差病例、in-vitro 大误差被抑制病例；图注写明选择规则，避免 cherry-picking。

## 12. 证据文件与复现

- 汇总统计：`tables/summary.csv`
- 配对比较：`tables/paired_comparisons.csv`
- cooperative 对 legacy：`tables/cooperative_profile_comparisons.csv`
- factorial interaction：`tables/factorial_interactions.csv`
- visibility 分层：`tables/visibility_strata.csv`
- 图件：`figures/*.pdf` 与 `figures/*.png`
- 文件哈希与生成参数：`manifest.json`
- 训练机制诊断：`../../diagnostics/interaction/REPORT.md`
- DePoLL 协议审计：`../../experiments/geotransformer.p2p_liver/DEPOLL_EVALUATION.md`

复现现有统计和图件：

```bash
/home/yangx/miniconda3/envs/geo_v2/bin/python scripts/build_sci_evidence.py \
  --output artifacts/rtorv2_model_audit --bootstrap 10000 --seed 7351
```

## 13. 最终审稿意见

若稿件把 RTORv2 描述为一个在所有设置下稳定超越 GeoTransformer 的新 SOTA，我会建议拒稿，因为效应量过小、单 seed、验证划分不独立、交互无显著证据且外部数据失败。若稿件改为 **任务特异的机制研究**，清楚区分 RTOR soft calibration 与失败的 hard pruning，突出 A3 的细匹配机制证据，并诚实报告低可见率/尾部改善及外部失效，则具备形成一篇严谨医学图像/医学机器人论文的证据基础。
