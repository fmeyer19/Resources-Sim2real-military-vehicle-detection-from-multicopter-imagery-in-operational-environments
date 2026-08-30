"""
Fourth scene: a flat grid terrain.

Places one vehicle per group on a fixed grid of spawn slots, settles it with a
physics drop, and renders it from an orbiting camera while the world texture is
rotated per frame.
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

SCENE_FILE = "Fourth_scene.blend"
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
    always_visible={"Grid", "DeadTrees"},
    turrets=TURRETS,
    class_map=CLASS_MAP,
    terrain_object="Grid",
    use_physics_drop=True,
    physics_sim_frames=60,
    objects_per_group=(1, 1),
    cameras_per_group=1,
    iterations=2,
)

PLACEMENT = FixedPositions([
    (-2.1, 2.7), (-2.1, 1.7), (-2.1, 0.7), (-2.1, -0.3), (-2.1, -1.3), (-2.1, -2.3),
    (-1.1, 2.7), (-1.1, 1.7), (-1.1, 0.7), (-1.1, -0.3), (-1.1, -1.3), (-1.1, -2.3),
    (0, 2.7), (0, 1.7), (0, 0.7), (0, -0.3), (0, -1.3), (0, -2.3),
    (2.1, 2.7), (2.1, 1.7), (2.1, 0.7), (2.1, -0.3), (2.1, -1.3), (2.1, -2.3),
    (1.1, 2.7), (1.1, 1.7), (1.1, 0.7), (1.1, -0.3), (1.1, -1.3), (1.1, -2.3),
])

CAMERA = OrbitCamera(radius_range=(10.0, 10.0), elevation_range=(45, 85))


if __name__ == "__main__":
    DatasetGenerator(SETTINGS, PLACEMENT, CAMERA, effects=[HdrRotator(HDR_NODE_NAME)]).run()
