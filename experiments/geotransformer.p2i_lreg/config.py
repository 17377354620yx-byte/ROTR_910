"""Configuration for the P2I-LReg synthetic rigid-registration experiment."""

import copy
import os
import os.path as osp
import sys


_ROOT_DIR = osp.realpath(osp.join(osp.dirname(__file__), "..", ".."))
if _ROOT_DIR in sys.path:
    sys.path.remove(_ROOT_DIR)
sys.path.insert(0, _ROOT_DIR)

from geotransformer.config import make_rtor_a3_cfg
from geotransformer.utils.common import ensure_dir


PROTOCOL_NAME = "released_corrected_v1"
DEFAULT_DATA_ROOT = "/mnt/data3/publicData/P2I-LReg/Liver_regis"


def _base_cfg():
    cfg = make_rtor_a3_cfg()
    cfg.root_dir = _ROOT_DIR
    cfg.working_dir = osp.dirname(osp.realpath(__file__))
    cfg.seed = 7351

    cfg.data.root = os.environ.get("P2I_LREG_ROOT", DEFAULT_DATA_ROOT)
    cfg.data.voxel_size = 0.001
    cfg.data.num_points = 8192
    cfg.data.validation_size = 500
    cfg.data.neighbor_calibration_samples = 100
    cfg.data.neighbor_limits = None
    cfg.data.source_surface_samples = 200000
    cfg.data.target_uniform_stride = 8
    cfg.data.target_outlier_neighbors = 20
    cfg.data.target_outlier_std_ratio = 2.0

    # The shared liver model is a single-pair network. Two micro-batches are
    # accumulated to preserve the released effective batch size of two.
    cfg.train.batch_size = 1
    # Open3D mesh sampling is not fork-safe in this dataset adapter.
    cfg.train.num_workers = 0
    cfg.train.point_limit = None
    cfg.train.use_augmentation = False
    cfg.train.augmentation_noise = 0.0
    cfg.train.augmentation_rotation = 0.0
    cfg.test.batch_size = 1
    cfg.test.num_workers = 0
    cfg.test.point_limit = None

    cfg.optim.optimizer = "SGD"
    cfg.optim.lr = 1e-4
    cfg.optim.lr_decay = 0.95
    cfg.optim.lr_decay_steps = 1
    cfg.optim.weight_decay = 1e-6
    cfg.optim.momentum = 0.93
    cfg.optim.max_epoch = 120
    cfg.optim.grad_acc_steps = 2

    # Restrict BF16 autocast to the geometric-transformer core. RTOR,
    # correspondence losses and rigid pose solvers remain in FP32.
    cfg.precision = dict(
        selective_bf16=True,
        autocast_dtype="bfloat16",
    )

    cfg.backbone.init_voxel_size = 0.001
    cfg.backbone.init_radius = cfg.backbone.base_radius * cfg.backbone.init_voxel_size
    cfg.backbone.init_sigma = cfg.backbone.base_sigma * cfg.backbone.init_voxel_size
    # At 8192 points / 1 mm, k=3 angular tensors exceed 32 GiB. This shared
    # experiment-only setting preserves the full inputs and voxel hierarchy.
    cfg.geotransformer.angle_k = 1
    cfg.geotransformer.hidden_dim = 128

    cfg.model.ground_truth_matching_radius = 0.0065
    cfg.fine_loss.positive_radius = 0.0065
    cfg.fine_matching.acceptance_radius = 0.01
    cfg.fine_matching.robust_refinement_radius = None
    cfg.fine_matching.correspondence_limit = None

    cfg.protocol = dict(
        name=PROTOCOL_NAME,
        unit="meter",
        transform="src_to_ref",
        input_points=8192,
        effective_batch_size=2,
        voxel_size_m=0.001,
        geotransformer_angle_k=1,
        geotransformer_hidden_dim=128,
        selective_bf16=True,
        autocast_scope="geometric_transformer_only",
        downstream_dtype="float32",
        invalid_reference_policy="exclude_empty_observation_without_gt",
        topk=[2000, 1500, 1000, 500, 250],
        ir_distance_threshold_m=0.01,
        fmr_inlier_ratio_threshold=0.05,
        rr_mean_displacement_threshold_m=0.01,
        ransac_distance_threshold_m=0.01,
        ransac_iterations=50000,
        ransac_confidence=0.999,
        sanity_min_improved_fraction=0.90,
        sanity_max_median_ratio=0.50,
    )
    return cfg


_C = _base_cfg()


def make_cfg(architecture="rtor", create_dirs=True):
    """Return an isolated config for the baseline or RTOR-only experiment."""
    if architecture not in ("geotransformer", "rtor"):
        raise ValueError(f"Unknown architecture: {architecture}")
    cfg = copy.deepcopy(_C)
    cfg.ablation.architecture = architecture
    cfg.ablation.rtor_enabled = architecture == "rtor"
    cfg.ablation.a3_enabled = False
    run_name = os.environ.get("P2I_LREG_RUN_NAME", architecture).strip()
    if not run_name or "/" in run_name or "\\" in run_name or run_name in (".", ".."):
        raise ValueError("P2I_LREG_RUN_NAME must be a nonempty directory name")
    cfg.exp_name = f"geotransformer.p2i_lreg.{run_name}"
    cfg.output_dir = osp.join(cfg.root_dir, "output", cfg.exp_name)
    cfg.snapshot_dir = osp.join(cfg.output_dir, "snapshots")
    cfg.log_dir = osp.join(cfg.output_dir, "logs")
    cfg.event_dir = osp.join(cfg.output_dir, "events")
    cfg.feature_dir = osp.join(cfg.output_dir, "features")
    cfg.registration_dir = osp.join(cfg.output_dir, "registration")
    if create_dirs:
        for path in (
            cfg.output_dir,
            cfg.snapshot_dir,
            cfg.log_dir,
            cfg.event_dir,
            cfg.feature_dir,
            cfg.registration_dir,
        ):
            ensure_dir(path)
    return cfg


__all__ = ["DEFAULT_DATA_ROOT", "PROTOCOL_NAME", "make_cfg"]
