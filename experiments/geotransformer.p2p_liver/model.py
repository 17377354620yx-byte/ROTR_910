"""RTOR+A3 model entry point for the P2P liver experiment."""

from geotransformer.modules.liver.registration_model import GeoTransformer


def create_model(config):
    """Build the checkpoint-compatible P2P registration model."""
    return GeoTransformer(config)


__all__ = ['GeoTransformer', 'create_model']
