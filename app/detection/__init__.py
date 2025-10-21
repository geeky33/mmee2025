"""Outlier detection and evaluation module."""

from .outlier_detectors import (
    isolation_forest_single_class,
    lof_single_class,
    knn_quantile_single_class,
    dbscan_single_class,
    run_classwise_detector,
    combine_classwise_predictions,
)

from .fusion_scorer import (
    build_feature_table,
    train_fusion_scorer,
    apply_fusion_scorer,
)

from .evaluation import (
    compute_detection_metrics,
    get_confusion_buckets,
    create_bucket_dataframe,
    compute_per_class_metrics,
    compute_pr_roc_curves,
    create_export_dataframes,
    find_top_n_missed_per_class,
)