"""
Shared request/response schemas for 3D asset pipeline.
"""
from enum import Enum
from typing import Optional

# Job states for Redis
class JobState(str, Enum):
    QUEUED = "queued"
    GENERATING_IMAGES = "generating_images"
    RECONSTRUCTING_3D = "reconstructing_3d"
    POST_PROCESSING = "post_processing"
    COMPLETED = "completed"
    FAILED = "failed"


class OutputFormat(str, Enum):
    GLB = "glb"
    OBJ = "obj"
    FBX = "fbx"


class QualityTier(str, Enum):
    FAST = "fast"       # 2 views, quickest
    STANDARD = "standard"
    HIGH = "high"


# Canonical view names for multi-view generation
MULTI_VIEW_PROMPTS = {
    "front": "front view, centered, neutral gray background, full object visible",
    "back": "back view, centered, neutral gray background, full object visible",
    "left": "left side view, centered, neutral gray background, full object visible",
    "right": "right side view, centered, neutral gray background, full object visible",
    "front_left": "front-left three-quarter view, neutral gray background, full object visible",
    "front_right": "front-right three-quarter view, neutral gray background, full object visible",
    "back_left": "back-left three-quarter view, neutral gray background, full object visible",
    "back_right": "back-right three-quarter view, neutral gray background, full object visible",
}

FAST_VIEWS = ["front", "back"]  # 2 views — fastest
DEFAULT_VIEWS = ["front", "back", "left", "right"]  # 4 views
HIGH_QUALITY_VIEWS = ["front", "back", "left", "right", "front_left", "front_right", "back_left", "back_right"]
