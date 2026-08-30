"""
Second scene: a beach with barbed-wire fences.

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

SCENE_FILE = "Second_scene.blend"
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
        "Cube",
        "Big_Stones_Beach",
        "Grass",
        "Grass.001",
        "Small_Stones_Beach",
        "Trees",
        "Barbed wire fence",
        "Barbed wire fence.001",
        "Barbed wire fence.002",
        "Barbed wire fence.003",
        "Barbed wire fence.004",
        "Barbed wire fence.005",
        "Barbed wire fence.006",
        "Barbed wire fence.007",
        "rock_moss_set_02_rock13.002",
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
    (-3.9, 2.3), (-3.9, 1.7), (-3.9, 0.6), (-3.9, -0.2), (-3.9, -1), (-3.9, -2.7),
    (-2.5, 2.3), (-2.5, 1.7), (-2.5, 0.6), (-2.5, -0.2), (-2.5, -1), (-2.5, -2.7),
    (-1.2, 2.3), (-1.2, 1.7), (-1.2, 0.6), (-1.2, -0.2), (-1.2, -1), (-1.2, -2.7),
    (0.8, 2.3), (0.8, 1.7), (0.8, 0.6), (0.8, -0.2), (0.8, -1), (0.8, -2.7),
    (2.5, 2.3), (2.5, 1.7), (2.5, 0.6), (2.5, -0.2), (2.5, -1), (2.5, -2.7),
])

CAMERA = OrbitCamera(radius_range=(3.0, 7.0), elevation_range=(15, 85))


if __name__ == "__main__":
    DatasetGenerator(SETTINGS, PLACEMENT, CAMERA, effects=[HdrRotator(HDR_NODE_NAME)]).run()
