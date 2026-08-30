"""
First scene: a muddy landscape.

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

SCENE_FILE = "First_scene.blend"
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
        "Landscape.001",
        "Rocks",
        "Trees_for_Mud",
        "Landscape_plane.001",
    },
    turrets=TURRETS,
    class_map=CLASS_MAP,
    terrain_object="Landscape.001",
    use_physics_drop=True,
    physics_sim_frames=60,
    objects_per_group=(8, 8),
    cameras_per_group=1,
    iterations=2,
)

PLACEMENT = FixedPositions([
    (1.7, -6.1), (1.7, -4.3), (1.7, -3.3), (1.7, -2.4), (1.7, -1), (1.7, 0),
    (1.7, 1), (1.7, 2), (1.7, 4), (1.7, 5), (1.7, 6.5), (2, 6.5), (2, 5.5),
    (2, -3), (2, -4), (-1.7, -6.1), (-1.7, -4.3), (-1.7, -3.3), (-1.7, -2.4),
    (-1.7, -1), (-1.7, 0), (-1.7, 1), (-1.7, 2), (-1.7, 4), (-1.7, 5),
    (-1.7, 6.5), (-0.9, -0.4), (-0.9, 3.7),
])

CAMERA = OrbitCamera(radius_range=(7.0, 15.0), elevation_range=(45, 85))


if __name__ == "__main__":
    DatasetGenerator(SETTINGS, PLACEMENT, CAMERA, effects=[HdrRotator(HDR_NODE_NAME)]).run()
