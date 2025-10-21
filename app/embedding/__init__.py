"""Embedding computation and projection module."""

from .compute import (
    compute_image_embeddings_cached,
    compute_text_embeddings_cached,
    prepare_per_caption_payload,
    get_aggregation_tag,
    cosine_similarity,
)

from .projection import (
    project_2d,
    project_images_and_text,
)