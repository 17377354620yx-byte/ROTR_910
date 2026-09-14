# DEPOLL 配准评测与协议核查

## 结论与范围

本入口测试 `trainval.py` 训练的 RTOR+A3 模型，不重新训练，不使用 DEPOLL 标记点挑选权重、初始化模型或筛选预测。论文方法是多特征融合与双曲嵌入模型，本项目的 RTOR+A3 不是该论文模型；评测协议正确不能保证达到论文结果。

证据来源：用户提供的论文 PDF 第 3.2、3.4.2 节与表 3、4；本地 DEPOLL 数据、`dataset_and_evaluation_description.pdf`；本项目训练与模型代码。论文地址：https://doi.org/10.1016/j.cmpb.2026.109451 。本报告的数值对应本地版本，不将 PDF 中的指示作为用户操作要求。

## 两种任务必须分开报告

| 参数 | 源点云 | 目标点云 | 对应论文 |
|---|---|---|---|
| `intraop` | 每例 `surfaces/surface_CT.ply` | 同例 `surfaces/video_reconstruction.ply` | 表 3 |
| `preop` | `reference/referenceModel/segmentedLobes/surface.ply` | 每例视频点云 | 表 4 |

本地分叶文件名为 `surface.ply`；代码也兼容同目录的 `surfaceFull.ply`，不会替换成非分叶肝脏。每种任务都包含 01–13 全部病例，不按注册好坏丢弃病例。术中 CT 是表面区域，不能把它当成完整术前肝脏。

## 已验证的坐标、关联和指标

1. 本地两个 surface PLY 已经处于视频重建坐标系。**本地术中 `markers.yml` 则仍需通过 `coordinate_transforms/XX/M.yml` 转换到视频坐标系。**例如 01 的视频点云质心 z 约 −628 mm，原始 clips 质心 z 约 −371 mm，经过 M 后约 −609 mm。说明 PDF 中“所有实体都在视频坐标系”的概括不能直接套用此版本。
2. 本地 Association 是 **参考标记索引 → 术中标记索引，值从 1 开始**。若按术中 YAML 的顺序评估，参考点必须用 `reference[np.argsort(Association - 1)]` 重排；等价地可用 `intraoperative[Association - 1]` 与原序参考点比较。说明 PDF 文字写的是相反方向。本实现检查 60 点置换、45 balls / 15 clips 分组，防止默默错配。
3. `intraop` 两端评估都使用同一例术中标记点。`preop` 使用关联后的术前点和术中点。两种任务不能混用 marker 来源。
4. 恢复未注册输入：对已经注册的**源表面及其源标记**同时应用 `inv(M)`，目标保留视频坐标。模型估计恢复后的源 → 视频。M、标记、关联表均不传给网络，模型输入中也没有占位 GT transform。
5. 输入跟随训练默认体素化：源和目标各自按半径体素化（0.04），随后各自减质心，**共同除以源半径**；不会各自除以不同半径。没有随机裁剪、点数上限或额外合成噪声。
6. 网络输出的归一化变换 `(R,t)` 转回毫米：`t_mm = scale*t + target_center - R@source_center`。不能只把平移乘以 scale。
7. 每例每组 `TRE = mean(norm(predicted_marker - target_marker))`。对 13 例 TRE 求均值及总体标准差 `ddof=0`。同时保存 RMS-TRE 和 `ddof=1` 以便审计，但不能替代主 TRE。

关键交叉验证：上述处理在 13 例上得到论文表 4 Ground-Truth **clips 28.89380294 ± 8.98281817 mm，balls 27.85076224 ± 10.33541484 mm**，四舍五入与论文完全一致。使用说明文字方向的 `reference[Association-1]`，则变成 clips 47.18 ± 13.68 mm、balls 70.57 ± 7.68 mm。术中任务使用发布变换的 marker 误差约 1e−14 mm，验证了恢复/还原闭环。

这里的 Ground-Truth 是**发布对齐下的标记误差**，不是经过标记点优化的最小刚体误差，也不是模型预测精度。非刚体变形下它不能直接称为刚体误差下界。

## 运行

在项目根目录：

```bash
cd /home/yangx/code/new_deform/ai_worker/RTORv2
bash scripts/test_depoll.sh
```

脚本默认使用 `geo_v2` 环境、GPU 0，以及训练保存的 `rtor_a3_cooperative_v1/snapshots/best.pth.tar`。输出写入带时间戳的新目录。也可完全显式运行：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/home/yangx/miniconda3/envs/geo_v2/bin/python \
  experiments/geotransformer.p2p_liver/test_depoll.py \
  --root /home/yangx/code/new_deform/DEPOLL \
  --snapshot output/geotransformer.p2p_liver.rtor_a3_cooperative_v1/snapshots/best.pth.tar \
  --protocol both --input_voxel 0.04 \
  --output output/depoll_my_run
```

仅核查数据/指标，不需要 GPU 或权重：

```bash
/home/yangx/miniconda3/envs/geo_v2/bin/python \
  experiments/geotransformer.p2p_liver/test_depoll.py \
  --audit_only --output output/depoll_my_audit
```

保留全部原始顶点，检查输入采样敏感性：

```bash
DEPOLL_OUTPUT=output/depoll_my_raw_run bash scripts/test_depoll.sh --input_voxel 0
```

`--input_voxel 0` 会增加 CPU 内存与时间消耗；这是明确的另一项实验，不按结果选更好的设置。论文称域外数据不设点数上限，但没有给出足以完全重建其 DEPOLL 归一化/采样代码的细节。本默认方案采用当前权重的训练预处理；raw 方案取消输入体素化，网络内部多尺度采样仍存在。

单病例调试可加 `--protocol intraop --cases 01`，不能将其输出称作 13 例结果。输出目录必须不存在，避免误覆盖结果。

## 输出格式

- `manifest.json`：命令、权重路径与 SHA256、epoch、训练协议、完整模型配置、邻域上限、预处理、随机种子及设备。
- `cases.csv`：逐例 clips/balls TRE、RMS-TRE、发布对齐基准、输入点数和对应点数。
- `summary.json`：按任务独立汇总。
- `intraop_01.npz` 等：预测矩阵、归一化质心/尺度、逐标记距离和两端坐标。

`estimated_transform_mm` 作用于恢复后的源点云；**直接变换原始源 PLY 时使用 `released_source_to_video_mm`**，其值为 `estimated_transform_mm @ inv(released_transform_mm)`。所有保存的物理点和矩阵平移均使用毫米。

权重按 checkpoint 元数据自动恢复 architecture / dual_encoder / registration_profile / interaction_profile，并 strict 加载。缺元数据时明确报错。邻域上限复用训练保存值，不重新读取仿真训练数据校准。

## 现有代码的问题与可比性限制

- 旧 `test.py` 只支持 in_silico/in_vitro，不是 DEPOLL 入口；其主指标是合并 marker 的 RMS-TRE。
- `trainval.py` 验证阶段的 `RMSE` 是归一化点位移诊断，不能当作 DEPOLL clips/balls TRE。
- 现有 `dataset.py::test_data_loader` 即使设置 checkpoint 邻域上限也重新校准，新入口避开该路径。
- 当前训练 `train_valid_data_loader` 的验证集来自同一训练 dataset 前若干组，并非保存元数据宣称的 “disjoint deformation groups”。本次不改变训练历史；`best` 只是原有验证标准选出的权重，不能称为独立验证最优。
- 本模型使用 LGR；论文第 3.4.1 节描述对应点置信度 >0.05、30,000 次 RANSAC、5 mm 距离阈值等设置。这里没有把 LGR 悄悄替换成 RANSAC，因此这是当前项目模型的 DEPOLL 测试，而非论文端到端复现。
- 即使指标基准精确复现，也不能据此证明网络预处理与论文未发布细节完全一致，更不能保证当前权重跨域性能。

## 验证

```bash
/home/yangx/miniconda3/envs/geo_v2/bin/python -m unittest discover \
  -s tests -p test_depoll_protocol.py -v
```

4 项测试覆盖：非平凡旋转及独立质心的毫米变换、TRE/RMS 区分、ASCII/大小端二进制 PLY、全 13 例论文 Ground-Truth 与术中坐标闭环。已全部通过。

## 已执行的全量模型结果

固定权重：`output/geotransformer.p2p_liver.rtor_a3_cooperative_v1/snapshots/best.pth.tar`，checkpoint epoch 68，architecture=rtor_a3，interaction_profile=cooperative，registration_profile=legacy，dual_encoder=false，neighbor_limits=[7,22,32,39]。并未遍历 epoch 按 DEPOLL 成绩挑选模型。

默认 input_voxel=0.04，全部 13 例：

| 任务 | 当前权重 clips TRE (mm) | 当前权重 balls TRE (mm) | 论文 Ours clips / balls (mm) |
|---|---:|---:|---|
| intraop / 表 3 | 76.66 ± 54.27 | 87.68 ± 65.90 | 11.72 ± 6.32 / 13.10 ± 7.61 |
| preop / 表 4 | 101.86 ± 36.33 | 101.45 ± 35.41 | 39.47 ± 17.36 / 44.38 ± 23.91 |

完整输出：`output/depoll_best68_v1/`；独立数据审计：`output/depoll_audit_v1/`。默认全量测试已完成，结果表明除了潜在评测错误，当前模型也确有跨域注册失败；不能承诺只修评测就接近论文。

复现性复跑：`output/depoll_best68_repeat_v1/`。两次 26 个预测矩阵的最大绝对差为 **0**，全部汇总字段完全一致；记录见该目录 `repeat_verification.json`。耗时字段不作为确定性比较指标。
