"""
Embedding computation wrappers for CLIP models.
These functions wrap the utils.clip_utils functions with caching.
"""

import streamlit as st
import numpy as np
from typing import List
from importlib import import_module

# Import CLIP utilities from project root
_clip = import_module("utils.clip_utils")

# Core functions from utils.clip_utils
compute_image_embeddings_raw = _clip.compute_image_embeddings
compute_text_embeddings_raw = _clip.compute_text_embeddings
cosine_similarity = _clip.cosine_similarity


@st.cache_data(show_spinner=False)
def compute_image_embeddings_cached(
    image_paths: List[str],
    model_name: str,
    pretrained: str,
    batch_size: int,
    cache_key: str
) -> np.ndarray:
    """
    Compute image embeddings with Streamlit caching.
    
    Args:
        image_paths: List of image file paths
        model_name: CLIP model name (e.g., 'ViT-B-32')
        pretrained: Pretrained weights name (e.g., 'openai')
        batch_size: Batch size for processing
        cache_key: Unique key for cache invalidation
        
    Returns:
        Image embeddings array of shape (N, embedding_dim)
    """
    return compute_image_embeddings_raw(
        image_paths,
        model_name=model_name,
        pretrained=pretrained,
        batch_size=batch_size
    )


@st.cache_data(show_spinner=False)
def compute_text_embeddings_cached(
    captions: List[List[str]],
    model_name: str,
    pretrained: str,
    aggregate: str,
    cache_key: str
) -> np.ndarray:
    """
    Compute text embeddings with Streamlit caching.
    
    Args:
        captions: List of caption lists, one per image
        model_name: CLIP model name
        pretrained: Pretrained weights name
        aggregate: Aggregation method ('average' or 'first')
        cache_key: Unique key for cache invalidation
        
    Returns:
        Text embeddings array of shape (N, embedding_dim)
    """
    return compute_text_embeddings_raw(
        captions,
        model_name=model_name,
        pretrained=pretrained,
        aggregate=aggregate
    )


def prepare_per_caption_payload(
    all_captions: List[List[str]],
    caps_limit: int = 0
) -> tuple:
    """
    Prepare caption payload for per-caption mode.
    
    Args:
        all_captions: List of caption lists, one per image
        caps_limit: Maximum captions per image (0 = all)
        
    Returns:
        Tuple of:
            - caption_payload: List of single-caption lists
            - caption_texts: Flat list of caption strings
            - caption_image_idx: List mapping each caption to its image index
    """
    caption_payload = []
    caption_texts = []
    caption_image_idx = []
    
    for img_idx, caps in enumerate(all_captions):
        if not caps:
            continue
        
        # Take first K captions or all if limit is 0
        captions_to_use = caps if caps_limit == 0 else caps[:caps_limit]
        
        for caption in captions_to_use:
            caption_payload.append([caption])  # Each item is a single-caption list
            caption_texts.append(caption)
            caption_image_idx.append(img_idx)
    
    return caption_payload, caption_texts, caption_image_idx


def get_aggregation_tag(per_caption: bool, caps_limit: int, text_agg: str) -> str:
    """
    Generate an aggregation tag for cache keys.
    
    Args:
        per_caption: Whether per-caption mode is enabled
        caps_limit: Caption limit in per-caption mode
        text_agg: Text aggregation method ('average' or 'first')
        
    Returns:
        Aggregation tag string
    """
    if per_caption:
        return f"percap_{caps_limit}"
    else:
        return f"agg_{'avg' if text_agg == 'average' else 'first'}"