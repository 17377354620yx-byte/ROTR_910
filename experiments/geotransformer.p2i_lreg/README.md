# P2I-LReg synthetic rigid registration

This experiment implements the audited `released_corrected_v1` protocol. The
source is the complete preoperative liver, the reference is the released
synthetic partial liver, and every transform maps `src -> ref`. All internal
coordinates are metres.

Both architectures use exactly the same dataset adapter, 8192-point sampling,
1 mm voxel hierarchy, predicted correspondences, top-k selection, metrics and
pose estimators. `geotransformer` disables RTOR and A3; `rtor` enables RTOR and
disables A3. Inference removes `transform` before model forward. There is no
ICP, GT-guided inference, oracle selection, SRSA or non-rigid deformation.

The single-pair liver model uses micro-batch 1 with two-step gradient
accumulation, giving effective batch size 2. The 8192-point/1 mm hierarchy is
memory intensive, so this experiment records and shares `angle_k=1` and
transformer hidden dimension 128 across both architectures.

Run the mandatory direction/unit check first:

```bash
conda run -n geo_v2 python experiments/geotransformer.p2i_lreg/check_dataset.py \
  --num_samples 100 --output output/p2i_lreg_dataset_sanity_100.json
```

Formal training (120 epochs by default):

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
  conda run -n geo_v2 python experiments/geotransformer.p2i_lreg/trainval.py \
  --architecture geotransformer

PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
  conda run -n geo_v2 python experiments/geotransformer.p2i_lreg/trainval.py \
  --architecture rtor
```

Evaluation defaults to all configured top-k values and all three estimators:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n geo_v2 python \
  experiments/geotransformer.p2i_lreg/test.py --architecture rtor \
  --snapshot output/geotransformer.p2i_lreg.rtor/snapshots/epoch-120.pth.tar
```
