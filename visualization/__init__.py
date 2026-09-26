"""Publication-quality visualization helpers for point-cloud registration."""

from .error_heatmap import (
    apply_transform,
    compute_registration_error,
    generate_error_heatmap,
)
from .statistics import compute_error_statistics, plot_error_distribution
from .paper_registration_visualizer import (
    compute_partial_registration_error,
    extract_partial_source_indices,
    generate_paper_registration_visualization,
)
from .livermatch_style_comparison import render_dataset_comparison

__all__ = [
    "apply_transform",
    "compute_registration_error",
    "compute_error_statistics",
    "generate_error_heatmap",
    "compute_partial_registration_error",
    "extract_partial_source_indices",
    "generate_paper_registration_visualization",
    "plot_error_distribution",
    "render_dataset_comparison",
]
