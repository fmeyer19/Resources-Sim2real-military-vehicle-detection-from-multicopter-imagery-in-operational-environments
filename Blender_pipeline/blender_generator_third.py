"""
Third scene: a terrain with scattered vegetation.

Places vehicles on a fixed grid of spawn slots, settles them with a physics drop,
and renders them from an orbiting camera while the world texture is rotated per
frame.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path

from blender_dataset_core import (
    SCENE_ROOT,
    DatasetGenerator,
    FixedPositions,
    GenerationSettings,
    HdrRotator,
    OrbitCamera,
)

SCENE_FILE = "Third_scene.blend"
"""
Scene file for this run, resolved against :data:`SCENE_ROOT`.
"""

OBJECT_POOL = []

TURRETS = {}

CLASS_MAP = {name: 0 for name in OBJECT_POOL}

HDR_NODE_NAME = "Mapping"

SETTINGS = GenerationSettings(
    scene_path=SCENE_ROOT / SCENE_FILE,
    physics_ignore=["Object_4.010"],
    output_dir=Path(__file__).parent / "output",
    vehicles_collection="Vehicles",
    object_pool=OBJECT_POOL,
    always_visible={
        "Landscape.005",
        "1",
        "Pingzuear",
        "rwr34",
        "Bushes2",
        "ScatterSystem",
        "ScatterSystem.001",
        "ScatterSystem.002",
    },
    turrets=TURRETS,
    class_map=CLASS_MAP,
    terrain_object="Landscape.005",
    use_physics_drop=True,
    physics_sim_frames=60,
    objects_per_group=(8, 8),
    cameras_per_group=1,
    iterations=2,
)

PLACEMENT = FixedPositions([
    (-1.7, -0.75), (-2.5, -1.25), (-3.2, -2.1),
    (-1, 2.7), (-1, 1), (-1, 0),
    (0, 2.7), (0, 1),
    (1, 2.7), (1, 1), (1, 0),
    (0.6, -1), (2, -1), (2, -1.7),
    (-3.2, -1.4), (-3, 1),
])

CAMERA = OrbitCamera(radius_range=(7.0, 15.0), elevation_range=(40, 85))


if __name__ == "__main__":
    DatasetGenerator(SETTINGS, PLACEMENT, CAMERA, effects=[HdrRotator(HDR_NODE_NAME)]).run()
