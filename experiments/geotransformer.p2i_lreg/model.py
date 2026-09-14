"""Model entry points for the two fair P2I-LReg rigid experiments."""

from geotransformer.modules.liver.registration_model import GeoTransformer


def create_model(cfg):
    if cfg.ablation.architecture not in ("geotransformer", "rtor"):
        raise ValueError(f"unsupported P2I-LReg architecture: {cfg.ablation.architecture}")
    if bool(cfg.ablation.a3_enabled):
        raise ValueError("A3 must remain disabled in the P2I-LReg rigid protocol")
    expected_rtor = cfg.ablation.architecture == "rtor"
    if bool(cfg.ablation.rtor_enabled) != expected_rtor:
        raise ValueError("architecture and rtor_enabled disagree")
    return GeoTransformer(cfg)


def forward_without_ground_truth(model, data_dict):
    """Inference-only forward that prevents pose supervision from reaching the model."""
    if model.training:
        raise ValueError("ground-truth-free forward is only valid in eval mode")
    inference_dict = dict(data_dict)
    inference_dict.pop("transform", None)
    return model(inference_dict)


__all__ = ["GeoTransformer", "create_model", "forward_without_ground_truth"]
