"""UI components module."""

from .sidebar import (
    render_dataset_selector,
    render_paths_display,
    render_captions_source_selector,
    render_model_controls,
    render_projection_controls,
    render_caption_controls,
    render_joint_outlier_controls,
    render_outlier_method_controls,
    render_combine_mode_control,
    render_validation_controls,
    render_fusion_controls,
    render_display_controls,
    render_evaluation_controls,
    render_subset_sampling_controls,
)

from .display import (
    display_kpis,
    display_detection_results,
    display_ranked_table,
    display_download_buttons,
    display_embedding_buttons,
    display_status_messages,
)