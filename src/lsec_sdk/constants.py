"""Constants and specifications for OpenPose and How2Sign dataset."""

from typing import Final, List, Tuple

# Resolution and video specifications for How2Sign front camera
DEFAULT_FRAME_WIDTH: Final[int] = 1280
DEFAULT_FRAME_HEIGHT: Final[int] = 720
DEFAULT_FPS: Final[float] = 50.0

# Keypoint counts
POSE_KEYPOINTS_COUNT: Final[int] = 25
FACE_KEYPOINTS_COUNT: Final[int] = 70
HAND_LEFT_KEYPOINTS_COUNT: Final[int] = 21
HAND_RIGHT_KEYPOINTS_COUNT: Final[int] = 21
TOTAL_KEYPOINTS_COUNT: Final[int] = (
    POSE_KEYPOINTS_COUNT + FACE_KEYPOINTS_COUNT + HAND_LEFT_KEYPOINTS_COUNT + HAND_RIGHT_KEYPOINTS_COUNT
)  # 137 keypoints

# Slice ranges in combined tensor (T, 137, C)
POSE_SLICE: Final[slice] = slice(0, 25)
FACE_SLICE: Final[slice] = slice(25, 95)
HAND_LEFT_SLICE: Final[slice] = slice(95, 116)
HAND_RIGHT_SLICE: Final[slice] = slice(116, 137)

# OpenPose BODY_25 Keypoint Names
BODY_25_NAMES: Final[List[str]] = [
    "Nose",             # 0
    "Neck",             # 1
    "RShoulder",        # 2
    "RElbow",           # 3
    "RWrist",           # 4
    "LShoulder",        # 5
    "LElbow",           # 6
    "LWrist",           # 7
    "MidHip",           # 8
    "RHip",             # 9
    "RKnee",            # 10
    "RAnkle",           # 11
    "LHip",             # 12
    "LKnee",            # 13
    "LAnkle",           # 14
    "REye",             # 15
    "LEye",             # 16
    "REar",             # 17
    "LEar",             # 18
    "LBigToe",          # 19
    "LSmallToe",        # 20
    "LHeel",            # 21
    "RBigToe",          # 22
    "RSmallToe",        # 23
    "RHeel"             # 24
]

# Sign Language Upper Body indices (excluding lower body that are not visible in frontal camera)
UPPER_BODY_INDICES: Final[List[int]] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 18
]

