"""
Fifth scene: backdrop planes, an orbiting distractor, and a single camera that
slides along a fixed path.

Defines the two scene-specific helpers :class:`PathLookAtCamera` and
:class:`OrbitingDistractor` alongside the ``SETTINGS``, ``PLACEMENT``, ``CAMERA``,
and ``EFFECTS`` this scene runs with.
"""

import math
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path

import bpy
from mathutils import Euler, Vector

from blender_dataset_core import (
    SCENE_ROOT,
    BackgroundRandomizer,
    BoundedRandomPositions,
    DatasetGenerator,
    GenerationSettings,
    OFFSCREEN,
    SceneEffect,
)


class PathLookAtCamera:
    """
    Camera placement that moves along a fixed poly-line path and always looks at the
    object.

    Each frame draws a point uniformly along the path (segments weighted by length),
    picks a height from *height_range*, and aims at the object. A stable angular offset
    is applied in the camera's local frame so pitch, yaw, and roll mean the same thing
    everywhere on the path. *zoom_range* scales the base focal length to fake a change
    in distance without moving the camera.
    """

    def __init__(
        self,
        path_xy: list[tuple[float, float]],
        height_range: tuple[float, float],
        pitch_offset_range_deg: tuple[float, float] = (0.0, 0.0),
        yaw_offset_range_deg: tuple[float, float] = (0.0, 0.0),
        roll_offset_range_deg: tuple[float, float] = (0.0, 0.0),
        zoom_range: tuple[float, float] = (1.0, 1.0),
    ):
        """
        :param path_xy: Ordered ``(x, y)`` vertices of the path the camera follows.
        :param height_range: Inclusive ``(min, max)`` camera height, drawn per frame.
        :param pitch_offset_range_deg: Degree range for the local pitch offset.
        :param yaw_offset_range_deg: Degree range for the local yaw offset.
        :param roll_offset_range_deg: Degree range for the local roll offset.
        :param zoom_range: Multiplier range applied to the base focal length.
        """
        self._path = [tuple(p) for p in path_xy]
        self._height_range = height_range
        self._pitch_offset_range = pitch_offset_range_deg
        self._yaw_offset_range = yaw_offset_range_deg
        self._roll_offset_range = roll_offset_range_deg
        self._zoom_range = zoom_range
        self._base_lens: float | None = None

        self._segments: list[tuple[tuple[float, float], tuple[float, float], float]] = []
        self._total_len = 0.0
        for (ax, ay), (bx, by) in zip(self._path, self._path[1:]):
            length = math.hypot(bx - ax, by - ay)
            self._segments.append(((ax, ay), (bx, by), length))
            self._total_len += length

    def __len__(self) -> int:
        """
        Return the number of objects this camera frames per placement, which is one.
        """
        return 1

    def _sample_xy(self) -> tuple[float, float]:
        """
        Return a random ``(x, y)`` point drawn uniformly along the path by arc length.
        """
        distance = random.uniform(0.0, self._total_len)
        for (ax, ay), (bx, by), length in self._segments:
            if length == 0.0:
                continue
            if distance <= length:
                f = distance / length
                return ax + (bx - ax) * f, ay + (by - ay) * f
            distance -= length
        return self._path[-1]

    def place(self, camera: bpy.types.Object, target: Vector, aim_jitter_ratio: float = 0.0) -> None:  # noqa: ARG002
        """
        Put *camera* on a random point of the path, aim it at *target*, and apply the
        configured local angular offset and zoom.
        """
        x, y = self._sample_xy()
        camera.location = Vector((x, y, random.uniform(*self._height_range)))

        base = (target - camera.location).to_track_quat("-Z", "Y")
        offset = Euler((
            math.radians(random.uniform(*self._pitch_offset_range)),
            math.radians(random.uniform(*self._yaw_offset_range)),
            math.radians(random.uniform(*self._roll_offset_range)),
        ), "XYZ").to_quaternion()

        camera.rotation_mode = "XYZ"
        camera.rotation_euler = (base @ offset).to_euler()

        if self._base_lens is None:
            self._base_lens = camera.data.lens
        camera.data.lens = self._base_lens * random.uniform(*self._zoom_range)


class OrbitingDistractor(SceneEffect):
    """
    Scene effect that places one clutter object on the target's orbit, between the
    camera and the target, so it partially occludes the subject.

    The distractor's bearing is taken from the object towards the camera, so it stays
    in front of the subject everywhere on the camera path. Its radius is capped just
    short of the camera distance, and its height rides the camera-to-target sight line
    so the occlusion holds as the camera height changes. The pool is gathered from the
    named collections at run time, with a random yaw and scale.
    """

    def __init__(
        self,
        collections: list[str],
        radius_range: tuple[float, float],
        angle_offset_range_deg: tuple[float, float] = (0.0, 0.0),
        z_offset_range: tuple[float, float] = (0.0, 0.0),
        z_min: float = 0.0,
        scale_range: tuple[float, float] = (0.8, 1.2),
    ):
        """
        :param collections: Names of the collections the distractor pool is drawn from.
        :param radius_range: Inclusive ``(min, max)`` XY distance from the target.
        :param angle_offset_range_deg: Degree range added to the target-to-camera
            bearing.
        :param z_offset_range: Height offset range added to the sight-line height.
        :param z_min: Lowest height the distractor is allowed to sit at.
        :param scale_range: Multiplier range applied to the distractor's base scale.
        """
        self._collections = list(collections)
        self._radius_range = radius_range
        self._angle_offset_range = angle_offset_range_deg
        self._z_offset_range = z_offset_range
        self._z_min = z_min
        self._scale_range = scale_range
        self._base_scales: dict[str, Vector] = {}

    def _orbit_xy(self, camera_location: Vector, target: Vector) -> tuple[float, float]:
        """
        Return an ``(x, y)`` point on the target's orbit, on the camera's side and
        never past the camera.
        """
        towards_camera = Vector((camera_location.x - target.x, camera_location.y - target.y))
        bearing = math.atan2(towards_camera.y, towards_camera.x)
        angle = bearing + math.radians(random.uniform(*self._angle_offset_range))

        radius = random.uniform(*self._radius_range)
        radius = min(radius, 0.9 * towards_camera.length)

        return target.x + radius * math.cos(angle), target.y + radius * math.sin(angle)

    @staticmethod
    def _sightline_z(camera_location: Vector, target: Vector, x: float, y: float) -> float:
        """
        Return the height of the camera-to-target segment above the point ``(x, y)``.
        """
        span = Vector((target.x - camera_location.x, target.y - camera_location.y))
        if span.length_squared == 0.0:
            return target.z
        offset = Vector((x - camera_location.x, y - camera_location.y))
        fraction = min(1.0, max(0.0, offset.dot(span) / span.length_squared))
        return camera_location.z + (target.z - camera_location.z) * fraction

    def _objects(self) -> list[bpy.types.Object]:
        """
        Return the mesh objects found across the configured distractor collections.
        """
        objects: dict[str, bpy.types.Object] = {}
        for name in self._collections:
            collection = bpy.data.collections.get(name)
            if collection is None:
                print(f"Distractor collection '{name}' not found")
                continue
            for obj in collection.all_objects:
                if obj.type == "MESH":
                    objects[obj.name] = obj
        return list(objects.values())

    def managed_names(self) -> set[str]:
        """
        Return the names of every object in the distractor pool.
        """
        return {obj.name for obj in self._objects()}

    def on_frame(self, camera: bpy.types.Object, target: Vector) -> None:
        """
        Hide the whole pool, then place one random distractor between *camera* and
        *target* with a random yaw and scale.
        """
        distractors = self._objects()
        for obj in distractors:
            obj.hide_render = True
            obj.location = OFFSCREEN.copy()

        if not distractors:
            return

        chosen = random.choice(distractors)

        base = self._base_scales.get(chosen.name)
        if base is None:
            base = chosen.scale.copy()
            self._base_scales[chosen.name] = base
        chosen.scale = base * random.uniform(*self._scale_range)

        x, y = self._orbit_xy(camera.location, target)
        z = self._sightline_z(camera.location, target, x, y) + random.uniform(*self._z_offset_range)
        chosen.location = Vector((x, y, max(self._z_min, z)))
        chosen.rotation_mode = "XYZ"
        chosen.rotation_euler = (0.0, 0.0, random.uniform(0.0, 2.0 * math.pi))
        chosen.hide_render = False


SCENE_FILE = "Fifth_scene.blend"
"""
Scene file for this run, resolved against :data:`SCENE_ROOT`.
"""

OBJECT_POOL = []

TURRETS = {}

CLASS_MAP = {name: 0 for name in OBJECT_POOL}

SETTINGS = GenerationSettings(
    scene_path=SCENE_ROOT / SCENE_FILE,
    output_dir=Path(__file__).parent / "output",
    vehicles_collection="Vehicles",
    object_pool=OBJECT_POOL,
    always_visible={"Plane", "Plane.002", "Plane.003"},
    turrets=TURRETS,
    class_map=CLASS_MAP,
    terrain_object="Plane",
    use_physics_drop=False,
    physics_sim_frames=60,
    objects_per_group=(1, 1),
    cameras_per_group=1,
    iterations=2,
)

PLACEMENT = BoundedRandomPositions(
    x_range=(-3.5, -3.5),
    y_range=(1.5, 1.5),
    min_spacing=0.0,
)

CAMERA = PathLookAtCamera(
    path_xy=[(0.0, 1.8), (0.0, -1.8), (-3.5, -1.8)],
    height_range=(2.0, 5.0),
    pitch_offset_range_deg=(-5.0, 5.0),
    yaw_offset_range_deg=(-12.0, 12.0),
    roll_offset_range_deg=(0.0, 0.0),
    zoom_range=(0.8, 1.6),
)

BACKGROUND = BackgroundRandomizer(
    target_planes=["Plane", "Plane.002", "Plane.003"],
    source_planes=[f"Plane.{i:03d}" for i in range(4, 14)],
)

EFFECTS = [
    BACKGROUND,
    OrbitingDistractor(
        collections=[
            "Realistic_Distractors",
            "Unreal_Shape_Distractors_Pattern",
            "Unreal_Shape_Distractors_Rand.Color",
        ],
        radius_range=(0.3, 0.8),
        angle_offset_range_deg=(-10.0, 10.0),
        z_offset_range=(-0.05, 0.05),
        z_min=0.0,
        scale_range=(0.8, 1.2),
    ),
]


if __name__ == "__main__":
    DatasetGenerator(SETTINGS, PLACEMENT, CAMERA, EFFECTS).run()
