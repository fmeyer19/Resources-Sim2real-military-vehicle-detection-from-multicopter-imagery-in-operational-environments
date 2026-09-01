"""
First scene: represents uneven muddy terrain with bare autumnal trees, water and rocks.

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
Scene file for this run, resolved against :data:`SCENE_ROOT`. Enter the path to the
scene the run is to render.
"""

OBJECT_POOL = []
"""
Names of the vehicle meshes allowed to spawn. Paste the entries of the vehicle list
belonging to the vehicles images are to be generated for.
"""

TURRETS = {}
"""
Hull-to-turret mapping of the pooled vehicles, taken from the same vehicle list.
"""

CLASS_MAP = {name: 0 for name in OBJECT_POOL}
"""
Class id of every pool entry, taken from the same vehicle list.
"""

HDR_NODE_NAME = "Mapping"

SETTINGS = GenerationSettings(
    scene_path=SCENE_ROOT / SCENE_FILE,
    
    # Objects that do not tolerate physics-based placement well.
    physics_ignore=["Object_4.010"],
    
    output_dir=Path(__file__).parent / "output",
    
    # Collection of the Blender scene holding the models of the vehicle lists.
    vehicles_collection="Vehicles",
    
    object_pool=OBJECT_POOL,
    
    # Scene elements that stay visible throughout the run.
    always_visible={
        "Landscape.001",
        "Rocks",
        "Trees_for_Mud",
        "Landscape_plane.001",
    },
    
    turrets=TURRETS,
    class_map=CLASS_MAP,
    
    # Terrain of the scene.
    terrain_object="Landscape.001",
    
    use_physics_drop=True,
    physics_sim_frames=60,
    
    # Replaceable values shaping how a run is composed, see the documentation.
    objects_per_group=(1, 8),
    cameras_per_group=2,
    iterations=2,
)

# Fixed spawn positions of this scene.
PLACEMENT = FixedPositions([
    (1.7, -6.1), (1.7, -4.3), (1.7, -3.3), (1.7, -2.4), (1.7, -1), (1.7, 0),
    (1.7, 1), (1.7, 2), (1.7, 4), (1.7, 5), (1.7, 6.5), (2, 6.5), (2, 5.5),
    (2, -3), (2, -4), (-1.7, -6.1), (-1.7, -4.3), (-1.7, -3.3), (-1.7, -2.4),
    (-1.7, -1), (-1.7, 0), (-1.7, 1), (-1.7, 2), (-1.7, 4), (-1.7, 5),
    (-1.7, 6.5), (-0.9, -0.4), (-0.9, 3.7),
])

# Adjustable camera parameters.
CAMERA = OrbitCamera(radius_range=(3.0, 15.0), elevation_range=(15, 85))


if __name__ == "__main__":
    DatasetGenerator(SETTINGS, PLACEMENT, CAMERA, effects=[HdrRotator(HDR_NODE_NAME)]).run()
