"""Extract 15-D feature vectors from video frames for HTK HMM state detection.

Feature vector layout (15D):
    [0-1]   hand_center (x_norm, y_norm)
    [2-3]   velocity    (vx, vy)
    [4-5]   acceleration (ax, ay)
    [6-9]   bounding box (width_norm, height_norm, delta_width, delta_height)
    [10-12] hand orientation cross product (ox, oy, oz) normalised
    [13]    object confidence (from CLIP classifier)
    [14]    ARUCO weighted bin context weight
    (optional [15] velocity magnitude -- kept for 16D variant)
"""

import os
import sys
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch

# Add lib folder so we can import hand_detection helpers
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
)
from hand_detection import hand_pos, hand_bounding_box

from .aruco_detection import ArucoDetector
from .config import DEFAULT_HTK_CONFIG


class FeatureExtractor:
    """Extract 15-D HMM feature vectors from video frames.

    Reuses the existing ``hand_detection`` library for hand position and
    bounding box, and the existing CLIP pipeline for object confidence.
    """

    FEATURE_DIM = 15

    def __init__(
        self,
        aruco_detector: ArucoDetector,
        clip_model=None,
        clip_processor=None,
        recognizer=None,
    ):
        """Initialise.

        Args:
            aruco_detector: An initialised ``ArucoDetector`` instance.
            clip_model: A loaded CLIP ``AutoModel`` (optional -- needed for
                object-confidence features).
            clip_processor: A loaded CLIP ``AutoProcessor``.
            recognizer: An ``ObjectRecognizer`` instance.  If provided it is
                used for object-confidence instead of the raw CLIP model.
        """
        self.aruco_detector = aruco_detector
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.recognizer = recognizer

        # Temporal state for velocity / acceleration across frames
        self._prev_hand_pos: Optional[np.ndarray] = None
        self._prev_velocity: Optional[np.ndarray] = None
        self._prev_bbox_size: Optional[np.ndarray] = None
        self._prev_frame_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset temporal state (call between videos)."""
        self._prev_hand_pos = None
        self._prev_velocity = None
        self._prev_bbox_size = None
        self._prev_frame_time = None

    def extract_frame_features(
        self,
        frame_rgb: np.ndarray,
        hand_landmarks: List[List[float]],
        segmented_hand: Optional[np.ndarray],
        frame_time: float,
    ) -> np.ndarray:
        """Extract a 15-D feature vector from a single frame.

        Args:
            frame_rgb: Full RGB frame.
            hand_landmarks: 21x3 list of MediaPipe hand landmarks
                (normalised ``[x, y, z]``).
            segmented_hand: Cropped hand image (for CLIP confidence).
                Can be ``None`` if no recogniser is available.
            frame_time: Timestamp of the current frame in seconds.

        Returns:
            ``np.ndarray`` of shape ``(15,)``.
        """
        # 1. Hand centre position (normalised 0-1) --------------------- 2D
        hand_center = self._compute_hand_center(hand_landmarks, frame_rgb.shape)
        hand_center_norm = np.array([
            hand_center[0] / frame_rgb.shape[1],
            hand_center[1] / frame_rgb.shape[0],
        ])

        # 2-3. Velocity & acceleration --------------------------------- 4D
        velocity, acceleration = self._compute_motion(
            hand_center_norm, frame_time
        )

        # 4. Bounding box features ------------------------------------- 4D
        bbox_features = self._compute_bbox_features(hand_landmarks, frame_rgb.shape)

        # 5. Hand orientation cross product ----------------------------- 3D
        orientation = self._compute_hand_orientation(hand_landmarks)

        # 6. Object confidence ------------------------------------------ 1D
        obj_conf = self._compute_object_confidence(segmented_hand)

        # 7. ARUCO weighted bin context --------------------------------- 1D
        bin_weight = self.aruco_detector.compute_bin_context_weight(
            frame_rgb, hand_center
        )

        features = np.concatenate([
            hand_center_norm,       # 2
            velocity,               # 2
            acceleration,           # 2
            bbox_features,          # 4
            orientation,            # 3
            [obj_conf],             # 1
            [bin_weight],           # 1
        ])
        return features  # shape (15,)

    def extract_video_features(
        self,
        video_path: str,
        frame_skip: int = 4,
        blur_threshold: float = 100.0,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, List[int], float]:
        """Extract features for an entire video.

        Returns:
            ``(features, frame_numbers, fps)`` where *features* has shape
            ``(n_frames, 15)``.
        """
        from ..preprocessing.blur_detection import is_blurry

        self.reset()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if verbose:
            print(f"\n[FeatureExtractor] Processing {video_path}")
            print(f"  Total frames: {total_frames}, FPS: {fps:.2f}")
            print(f"  Frame skip: {frame_skip}")

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.3,
            max_num_hands=2,
        )

        all_features: List[np.ndarray] = []
        frame_numbers: List[int] = []
        frame_count = 0

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_time = frame_count / fps if fps > 0 else 0.0

            # Detect hand landmarks
            results = hands.process(frame_rgb)
            if not results.multi_hand_landmarks:
                continue

            # Get right-hand landmarks (leftmost in mirrored view)
            hand_points_list = []
            for hand_landmarks_obj in results.multi_hand_landmarks:
                pts = [
                    [lm.x, lm.y, lm.z]
                    for lm in hand_landmarks_obj.landmark[:21]
                ]
                hand_points_list.append(pts)

            if len(hand_points_list) > 1:
                positions = [
                    hand_pos(hp, frame_rgb) for hp in hand_points_list
                ]
                idx = min(range(len(positions)), key=lambda i: positions[i][0])
                hand_landmarks = hand_points_list[idx]
            else:
                hand_landmarks = hand_points_list[0]

            # Segment hand (for CLIP / blur)
            bbox, bbox_size = hand_bounding_box(hand_landmarks, frame_rgb)
            top, bottom = max(0, bbox[0][1]), min(frame_rgb.shape[0], bbox[2][1])
            left, right = max(0, bbox[0][0]), min(frame_rgb.shape[1], bbox[2][0])
            if top >= bottom or left >= right:
                continue
            segmented = frame_rgb[top:bottom, left:right]

            if segmented.size == 0:
                continue
            if is_blurry(segmented, blur_threshold):
                continue

            features = self.extract_frame_features(
                frame_rgb, hand_landmarks, segmented, frame_time
            )
            all_features.append(features)
            frame_numbers.append(frame_count)

        cap.release()
        hands.close()

        if verbose:
            print(f"  Extracted {len(all_features)} feature vectors")

        if len(all_features) == 0:
            return np.empty((0, self.FEATURE_DIM)), [], fps

        return np.array(all_features), frame_numbers, fps

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_hand_center(
        landmarks: List[List[float]], frame_shape: Tuple[int, ...]
    ) -> Tuple[float, float]:
        """Return (x_px, y_px) hand centre using ``hand_pos``."""
        pos = hand_pos(landmarks, np.zeros(frame_shape, dtype=np.uint8))
        if not pos:
            return (0.0, 0.0)
        return pos

    def _compute_motion(
        self, hand_center_norm: np.ndarray, frame_time: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute velocity and acceleration from normalised position."""
        if self._prev_hand_pos is None or self._prev_frame_time is None:
            self._prev_hand_pos = hand_center_norm.copy()
            self._prev_frame_time = frame_time
            self._prev_velocity = np.zeros(2)
            return np.zeros(2), np.zeros(2)

        dt = frame_time - self._prev_frame_time
        if dt <= 0:
            dt = 1e-6

        velocity = (hand_center_norm - self._prev_hand_pos) / dt
        acceleration = (
            (velocity - self._prev_velocity) / dt
            if self._prev_velocity is not None
            else np.zeros(2)
        )

        self._prev_hand_pos = hand_center_norm.copy()
        self._prev_velocity = velocity.copy()
        self._prev_frame_time = frame_time
        return velocity, acceleration

    def _compute_bbox_features(
        self,
        landmarks: List[List[float]],
        frame_shape: Tuple[int, ...],
    ) -> np.ndarray:
        """Return (width_norm, height_norm, d_width, d_height)."""
        dummy_img = np.zeros(frame_shape, dtype=np.uint8)
        _, (h, w) = hand_bounding_box(landmarks, dummy_img)
        width_norm = w / frame_shape[1]
        height_norm = h / frame_shape[0]
        size = np.array([width_norm, height_norm])

        if self._prev_bbox_size is None:
            self._prev_bbox_size = size.copy()
            return np.array([width_norm, height_norm, 0.0, 0.0])

        delta = size - self._prev_bbox_size
        self._prev_bbox_size = size.copy()
        return np.array([width_norm, height_norm, delta[0], delta[1]])

    @staticmethod
    def _compute_hand_orientation(
        landmarks: List[List[float]],
    ) -> np.ndarray:
        """Cross product of (palm->thumb) x (palm->middle finger).

        Uses landmarks 0 (wrist), 4 (thumb tip), 12 (middle finger tip).
        Returns normalised 3-D vector; zeros if degenerate.
        """
        if len(landmarks) < 21:
            return np.zeros(3)

        wrist = np.array(landmarks[0])
        thumb = np.array(landmarks[4])
        middle = np.array(landmarks[12])

        v1 = thumb - wrist   # palm -> thumb
        v2 = middle - wrist  # palm -> middle finger

        cross = np.cross(v1, v2)
        norm = np.linalg.norm(cross)
        if norm < 1e-8:
            return np.zeros(3)
        return cross / norm

    def _compute_object_confidence(
        self, segmented_hand: Optional[np.ndarray]
    ) -> float:
        """Return max softmax confidence from CLIP classifier.

        Falls back to 0.0 if no recogniser / model is available.
        """
        if segmented_hand is None:
            return 0.0

        if self.clip_model is not None and self.clip_processor is not None:
            try:
                inputs = self.clip_processor(
                    images=[segmented_hand], return_tensors="pt"
                ).to(self.clip_model.device)
                with torch.no_grad():
                    features = self.clip_model.get_image_features(**inputs)
                return float(torch.sigmoid(features.max()).item())
            except Exception:
                return 0.0

        return 0.0


__all__ = ["FeatureExtractor"]
