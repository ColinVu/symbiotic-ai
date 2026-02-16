"""High-level training pipeline for the HTK HMM state detector."""

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .aruco_detection import ArucoDetector
from .config import DEFAULT_HTK_CONFIG, HTKConfig, STATE_CYCLE
from .feature_extraction import FeatureExtractor
from .htk_interface import HTKStateDetector


def _validate_state_sequence(annotations: pd.DataFrame) -> None:
    """Validate that *annotations* follow the required state cycle.

    Rules:
    * States must follow: PICK -> CARRY_WITH -> PLACE -> CARRY_EMPTY (repeat).
    * No state skipping.
    * Timestamps must be monotonically increasing with no gaps.
    """
    states = annotations["state"].tolist()

    # Check all states are valid
    valid = set(STATE_CYCLE)
    for s in states:
        if s not in valid:
            raise ValueError(
                f"Invalid state '{s}' in annotations. "
                f"Valid states: {STATE_CYCLE}"
            )

    # Check cycle order (allow starting at any point in the cycle)
    if len(states) >= 2:
        for i in range(1, len(states)):
            prev_idx = STATE_CYCLE.index(states[i - 1])
            curr_idx = STATE_CYCLE.index(states[i])
            expected_next = (prev_idx + 1) % len(STATE_CYCLE)
            if curr_idx != expected_next and curr_idx != prev_idx:
                raise ValueError(
                    f"State cycle violation at row {i}: "
                    f"'{states[i-1]}' -> '{states[i]}' is not allowed. "
                    f"Expected '{STATE_CYCLE[expected_next]}' or same state."
                )

    # Check monotonically increasing timestamps
    for i in range(1, len(annotations)):
        if annotations.iloc[i]["timestamp_start"] < annotations.iloc[i - 1]["timestamp_end"]:
            raise ValueError(
                f"Timestamps not monotonically increasing at row {i}: "
                f"start {annotations.iloc[i]['timestamp_start']} < "
                f"prev end {annotations.iloc[i-1]['timestamp_end']}"
            )


def train_state_detector(
    video_paths: List[str],
    annotation_paths: List[str],
    output_dir: str,
    aruco_config_path: Optional[str] = None,
    clip_model=None,
    clip_processor=None,
    frame_skip: int = 4,
    blur_threshold: float = 100.0,
    config: Optional[HTKConfig] = None,
    verbose: bool = True,
) -> str:
    """Train an HTK HMM state detector from annotated videos.

    Each video must have a corresponding CSV annotation file with columns
    ``[timestamp_start, timestamp_end, state]``.

    Args:
        video_paths: Paths to training video files.
        annotation_paths: Matching CSV annotation files (same order).
        output_dir: Where to save the trained HMM.
        aruco_config_path: Path to ``aruco_bins.json``.
        clip_model: Loaded CLIP model (optional, for object confidence).
        clip_processor: Loaded CLIP processor.
        frame_skip: Process every N-th frame.
        blur_threshold: Laplacian variance threshold.
        config: ``HTKConfig`` instance (defaults to ``DEFAULT_HTK_CONFIG``).
        verbose: Print progress.

    Returns:
        Path to the final trained model directory.
    """
    if len(video_paths) != len(annotation_paths):
        raise ValueError(
            "Number of videos must equal number of annotation files."
        )

    cfg = config or DEFAULT_HTK_CONFIG
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialise ARUCO detector
    aruco_detector = ArucoDetector(
        aruco_dict_type=cfg.aruco_dict_type,
        distance_decay=cfg.aruco_distance_decay,
    )
    if aruco_config_path and os.path.isfile(aruco_config_path):
        aruco_detector.load_bin_config(aruco_config_path)

    # Initialise feature extractor
    feature_extractor = FeatureExtractor(
        aruco_detector=aruco_detector,
        clip_model=clip_model,
        clip_processor=clip_processor,
    )

    # Extract features from all training videos
    training_data: List[tuple] = []
    for video_path, annotation_path in zip(video_paths, annotation_paths):
        if verbose:
            print(f"\n[HMM Training] Processing: {video_path}")

        annotations = pd.read_csv(annotation_path)
        _validate_state_sequence(annotations)

        features, frame_numbers, fps = feature_extractor.extract_video_features(
            video_path,
            frame_skip=frame_skip,
            blur_threshold=blur_threshold,
            verbose=verbose,
        )

        if features.shape[0] == 0:
            if verbose:
                print(f"  WARNING: No features extracted from {video_path}, skipping.")
            continue

        training_data.append((features, annotations))
        feature_extractor.reset()

    if len(training_data) == 0:
        raise ValueError("No usable training data extracted from any video.")

    # Train HTK HMM
    if verbose:
        print(f"\n[HMM Training] Training on {len(training_data)} video(s) ...")

    htk_detector = HTKStateDetector(output_dir, config=cfg)
    htk_detector.train(training_data, output_dir, verbose=verbose)

    final_dir = os.path.join(output_dir, "models", "hmm_final")
    if verbose:
        print(f"[HMM Training] Done. Model saved to: {final_dir}")

    return final_dir


__all__ = ["train_state_detector"]
