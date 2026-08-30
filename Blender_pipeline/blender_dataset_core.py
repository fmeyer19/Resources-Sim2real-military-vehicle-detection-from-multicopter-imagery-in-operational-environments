"""
Reusable dataset-generation core shared by every scene script.

Provides the placement strategies, camera placements, and scene effects, plus the
:class:`DatasetGenerator` that renders a loaded scene from many viewpoints and writes
YOLO-format bounding-box labels.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

import bpy
from mathutils import Vector


SCENE_ROOT = Path(__file__).parent / "resources"
"""
Directory that holds the ``.blend`` scene files.

Set this once to wherever the scene files live on this machine; every scene script
names its own file relative to this directory.
"""

OFFSCREEN = Vector((9999.0, 9999.0, 9999.0))
"""
Parking position far outside any scene, used to hide an object from a render.
"""

SINGLE_OBJECT_OFFSET_RATIO = 1.0 / 2.0
"""
Look-at jitter applied when a group holds a single object, as a fraction of the
camera-to-object distance.
"""

Position = tuple[float, float]
"""
An ``(x, y)`` coordinate pair in scene units.
"""

BoundingBox = tuple[float, float, float, float]
"""
A pixel-space box as ``(x_min, y_min, x_max, y_max)``.
"""

Annotation = tuple[int, BoundingBox]
"""
A labelled detection as ``(class_id, bounding_box)``.
"""


@dataclass
class RenderSettings:
    """
    Render engine and output resolution parameters.
    """

    width: int = 640
    """
    Rendered image width in pixels.
    """

    height: int = 640
    """
    Rendered image height in pixels.
    """

    samples: int = 64
    """
    Cycles path-tracing samples per pixel.
    """

    compute_device_type: str = "CUDA"
    """
    Cycles compute backend, for example ``"CUDA"``, ``"OPTIX"``, ``"HIP"``, or
    ``"ONEAPI"``.
    """


@dataclass
class GenerationSettings:
    """
    Full description of a dataset-generation run for one scene.
    """

    scene_path: str | Path
    """
    Path to the ``.blend`` scene file to load.
    """

    output_dir: Path
    """
    Directory that receives the ``images``, ``labels``, and ``classes.txt`` output.
    """

    vehicles_collection: str
    """
    Name of the Blender collection holding the spawnable vehicle meshes.
    """

    object_pool: list[str] | None
    """
    Names of the objects allowed to spawn, or ``None`` to use the whole collection.
    """

    class_map: dict[str, int]
    """
    Maps an object's original name to its integer class id.
    """

    terrain_object: str
    """
    Name of the surface object that placed vehicles are dropped onto.
    """

    turrets: dict[str, str] = field(default_factory=dict)
    """
    Maps a body object name to the turret object name mounted on it.
    """

    always_visible: set[str] = field(default_factory=set)
    """
    Names of objects that are never hidden between frames.
    """

    objects_per_group: tuple[int, int] = (8, 8)
    """
    Inclusive ``(min, max)`` range for how many objects are placed together per setup.
    """

    cameras_per_group: int = 4
    """
    Number of camera shots rendered for each placement.
    """

    iterations: int = 2
    """
    How many times the whole pool is reshuffled and rendered.
    """

    reject_overlaps: bool = False
    """
    Whether to drop an annotation whose box overlaps an already accepted one.
    """

    overlap_threshold: float = 0.5
    """
    Overlap ratio above which a box counts as overlapping when *reject_overlaps* is set.
    """

    class_names: list[str] | None = None
    """
    Explicit class names for ``classes.txt``, or ``None`` to generate ``class_0``,
    ``class_1``, and so on.
    """

    render: RenderSettings = field(default_factory=RenderSettings)
    """
    Render engine and resolution settings.
    """

    use_physics_drop: bool = False
    """
    Whether to settle placed bodies with a rigid-body simulation before rendering.
    """

    physics_sim_frames: int = 60
    """
    Number of simulation frames to step when *use_physics_drop* is set.
    """

    physics_ignore: list[str] = field(default_factory=list)
    """
    Original names of bodies excluded from the physics drop.
    """


class PlacementStrategy(Protocol):
    """
    Protocol for objects that decide where a group of vehicles is placed.
    """

    def positions(self, count: int) -> list[Position]:
        """
        Return up to *count* placement positions as ``(x, y)`` pairs.
        """
        ...


class CameraPlacement(Protocol):
    """
    Protocol for objects that position and aim the camera for one frame.
    """

    def place(self, camera: bpy.types.Object, target: Vector, aim_jitter_ratio: float = 0.0) -> None:
        """
        Move and orient *camera* for a frame that looks at *target*.

        :param camera: The camera object to move.
        :param target: The point the camera should look at.
        :param aim_jitter_ratio: Fraction of the target distance by which to offset the
            look-at point so the subject sits off-centre.
        """
        ...


class SceneEffect:
    """
    Base class for optional per-run behaviour at group and frame boundaries.

    Subclasses override only the hooks they need; the defaults do nothing.
    """

    def managed_names(self) -> set[str]:
        """
        Return the names of objects this effect controls so they are not hidden as
        scenery.
        """
        return set()

    def on_group_start(self) -> None:
        """
        Run once before each new group of objects is placed.
        """
        return None

    def on_frame(self, camera: bpy.types.Object, target: Vector) -> None:
        """
        Run before each frame is rendered.

        :param camera: The camera used for this frame.
        :param target: The point the camera is aimed at.
        """
        return None


def require_object(name: str) -> bpy.types.Object:
    """
    Return the scene object named *name*.

    :raises RuntimeError: if no object with that name exists.
    """
    obj = bpy.data.objects.get(name)
    if obj is None:
        available = [o.name for o in bpy.data.objects]
        raise RuntimeError(f"Object '{name}' not found. Available: {available}")
    return obj


def get_or_create_camera() -> bpy.types.Object:
    """
    Return the scene's first camera, adding one if the scene has none.
    """
    for obj in bpy.data.objects:
        if obj.type == "CAMERA":
            return obj
    bpy.ops.object.camera_add()
    return bpy.context.active_object


def place_object(obj: bpy.types.Object, x: float, y: float, height: float = 2.0) -> None:
    """
    Move *obj* to ``(x, y, height)``, give it a random yaw, and make it visible.
    """
    obj.rotation_mode = 'XYZ'
    obj.rotation_euler.z = random.uniform(0, 2 * math.pi)
    obj.location = Vector((x, y, height))
    obj.hide_viewport = False
    obj.hide_render = False


def partition_pool(pool: Sequence[bpy.types.Object], group_size: int) -> list[list[bpy.types.Object]]:
    """
    Shuffle *pool* and split it into consecutive groups of at most *group_size*.
    """
    shuffled = list(pool)
    random.shuffle(shuffled)
    return [shuffled[i:i + group_size] for i in range(0, len(shuffled), group_size)]


class SceneIO:
    """
    Loads a ``.blend`` file into the current Blender session.
    """

    @staticmethod
    def load(path: str | Path) -> None:
        """
        Open the ``.blend`` file at *path* as the current scene.
        """
        bpy.ops.wm.open_mainfile(filepath=str(path))
        print(f"Loaded scene: {path}")


class RenderConfigurator:
    """
    Applies :class:`RenderSettings` to the active scene and enables GPU rendering.
    """

    def __init__(self, settings: RenderSettings):
        self._settings = settings

    def apply(self) -> None:
        """
        Configure the active scene for GPU Cycles rendering at the requested
        resolution and sample count.
        """
        scene = bpy.context.scene
        scene.render.engine = "CYCLES"
        scene.render.resolution_x = self._settings.width
        scene.render.resolution_y = self._settings.height
        scene.render.image_settings.file_format = "PNG"
        scene.render.use_motion_blur = False
        scene.cycles.samples = self._settings.samples
        preferences = bpy.context.preferences.addons["cycles"].preferences
        preferences.compute_device_type = self._settings.compute_device_type
        preferences.get_devices()
        scene.cycles.device = "GPU"


class Visibility:
    """
    Hides objects by parking them off-screen and disabling their render, while
    keeping an allow-list of objects always visible.
    """

    def __init__(self, always_visible: set[str]):
        self._always_visible = always_visible

    def hide(self, obj: bpy.types.Object) -> None:
        """
        Park *obj* off-screen and disable its render, unless it is on the allow-list.
        """
        if obj.name in self._always_visible:
            return
        obj.location = OFFSCREEN.copy()
        obj.hide_render = True

    def hide_many(self, objects: Sequence[bpy.types.Object]) -> None:
        """
        Hide every object in *objects*.
        """
        for obj in objects:
            self.hide(obj)

    def hide_unprotected_meshes(self, protected: set[str]) -> None:
        """
        Hide every mesh object whose name is not in *protected*.
        """
        for obj in bpy.data.objects:
            if obj.type == "MESH" and obj.name not in protected:
                self.hide(obj)


class PoolSelector:
    """
    Resolves the spawnable mesh objects from a collection and an optional name filter.
    """

    def __init__(self, collection_name: str, object_pool: list[str] | None, turret_names: set[str]):
        self._collection_name = collection_name
        self._object_pool = object_pool
        self._turret_names = turret_names

    def select(self) -> list[bpy.types.Object]:
        """
        Return the mesh objects to spawn, keeping only the configured names and
        excluding turrets.

        :raises RuntimeError: if the collection does not exist.
        """
        collection = bpy.data.collections.get(self._collection_name)
        if collection is None:
            available = [c.name for c in bpy.data.collections]
            raise RuntimeError(
                f"Collection '{self._collection_name}' not found. Available: {available}"
            )
        meshes = [obj for obj in collection.all_objects if obj.type == "MESH"]
        if self._object_pool is None:
            return meshes
        wanted = set(self._object_pool)
        return [obj for obj in meshes if obj.name in wanted and obj.name not in self._turret_names]


class TurretMounter:
    """
    Clones body/turret pairs, spins each turret randomly, and joins them into a
    single object ready for placement.
    """

    def __init__(self, turrets: dict[str, str]):
        self._turrets = turrets
        self._body_original_locations: dict[str, Vector] = {}
        self._turret_original_locations: dict[str, Vector] = {}

    def turret_names(self) -> set[str]:
        """
        Return the set of turret object names.
        """
        return set(self._turrets.values())

    def all_turret_objects(self) -> list[bpy.types.Object]:
        """
        Return the turret objects that currently exist in the scene.
        """
        return [obj for name in self._turrets.values() if (obj := bpy.data.objects.get(name)) is not None]

    def initialize(self) -> None:
        """
        Record the original location of every body and turret.

        .. note:: Must run before any object is moved.
        """
        for body_name, turret_name in self._turrets.items():
            body = bpy.data.objects.get(body_name)
            turret = bpy.data.objects.get(turret_name)
            if body is None or turret is None:
                print(f"Missing turret pair: {body_name} -> {turret_name}")
                continue
            self._body_original_locations[body_name] = body.location.copy()
            self._turret_original_locations[turret_name] = turret.location.copy()

    def make_copy(self, body: bpy.types.Object, x: float, y: float, height: float) -> bpy.types.Object:
        """
        Return a placed copy of *body*.

        If the body has a turret, the turret is copied too, spun to a random yaw, and
        joined into the body copy so the result is one object.

        :param body: The source body object to clone.
        :param x: Target X location for the copy.
        :param y: Target Y location for the copy.
        :param height: Target Z location for the copy.
        """
        body_copy = body.copy()
        body_copy.data = body.data.copy()
        bpy.context.scene.collection.objects.link(body_copy)
        body_copy["original_name"] = body.name

        turret_name = self._turrets.get(body.name)
        turret = bpy.data.objects.get(turret_name) if turret_name else None

        if turret is not None:
            turret_copy = turret.copy()
            turret_copy.data = turret.data.copy()
            bpy.context.scene.collection.objects.link(turret_copy)

            body_copy.location = self._body_original_locations.get(body.name, Vector((0.0, 0.0, 0.0))).copy()
            body_copy.rotation_mode = 'XYZ'
            body_copy.hide_render = False
            body_copy.hide_viewport = False

            turret_copy.location = self._turret_original_locations.get(turret_name, Vector((0.0, 0.0, 0.0))).copy()
            turret_copy.rotation_mode = 'XYZ'
            turret_copy.hide_render = False
            turret_copy.hide_viewport = False

            turret_copy.rotation_euler.z = random.uniform(0, 2 * math.pi)

            with bpy.context.temp_override(
                active_object=body_copy,
                selected_editable_objects=[body_copy, turret_copy],
            ):
                bpy.ops.object.join()
            print(f"Joined turret copy into hull copy of '{body.name}'")

        place_object(body_copy, x, y, height=height)
        return body_copy

    @staticmethod
    def delete_copy(body_copy: bpy.types.Object) -> None:
        """
        Remove a copy made by :meth:`make_copy` along with its now-unused mesh data.
        """
        mesh = body_copy.data
        bpy.data.objects.remove(body_copy, do_unlink=True)
        if mesh and mesh.users == 0:
            bpy.data.meshes.remove(mesh)


class TerrainSnapper:
    """
    Rests objects on a terrain mesh by ray-casting straight down onto its surface.
    """

    def __init__(self, terrain: bpy.types.Object):
        self._terrain = terrain

    def snap(self, obj: bpy.types.Object) -> None:
        """
        Shift *obj* along Z so its lowest point sits on the terrain directly below it,
        falling back to the terrain origin height when the ray misses.
        """
        depsgraph = bpy.context.evaluated_depsgraph_get()
        matrix_inverse = self._terrain.matrix_world.inverted()
        origin = Vector((obj.location.x, obj.location.y, obj.location.z + 1000.0))
        direction = Vector((0, 0, -1))
        hit, location, _, _ = self._terrain.evaluated_get(depsgraph).ray_cast(
            matrix_inverse @ origin,
            (matrix_inverse.to_3x3() @ direction).normalized(),
        )
        if hit:
            terrain_z = (self._terrain.matrix_world @ location).z
            world_corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
            min_z = min(c.z for c in world_corners)
            obj.location.z += terrain_z - min_z
        else:
            obj.location.z = self._terrain.location.z

    def snap_all(self, objects: Sequence[bpy.types.Object]) -> None:
        """
        Snap every object in *objects* to the terrain.
        """
        for obj in objects:
            self.snap(obj)


class PhysicsDropper:
    """
    Settles placed bodies onto the terrain with a short rigid-body simulation.
    """

    def __init__(self, terrain: bpy.types.Object, sim_frames: int = 60, ignore: list[str] | None = None):
        self._terrain = terrain
        self._sim_frames = sim_frames
        self._ignore: set[str] = set(ignore) if ignore else set()

    def _make_active(self, obj: bpy.types.Object) -> None:
        """
        Make *obj* the active object for the next operator call.
        """
        bpy.context.view_layer.objects.active = obj

    def setup(self) -> None:
        """
        Create the rigid-body world and mark the terrain as a passive collider.
        """
        scene = bpy.context.scene
        if scene.rigidbody_world is None:
            bpy.ops.rigidbody.world_add()
        scene.rigidbody_world.enabled = True
        if self._terrain.rigid_body is None:
            self._make_active(self._terrain)
            bpy.ops.rigidbody.object_add()
        self._terrain.rigid_body.type = 'PASSIVE'
        self._terrain.rigid_body.collision_shape = 'MESH'

    def drop(self, bodies: list[bpy.types.Object]) -> None:
        """
        Simulate *bodies* falling for the configured number of frames, then freeze each
        at its settled pose.

        Bodies whose original name is in the ignore list are left where they are.
        """
        scene = bpy.context.scene
        active = [obj for obj in bodies if obj.get("original_name") not in self._ignore]
        for obj in active:
            self._make_active(obj)
            bpy.ops.rigidbody.object_add()
            obj.rigid_body.type = 'ACTIVE'
            obj.rigid_body.collision_shape = 'CONVEX_HULL'
            obj.rigid_body.collision_margin = 0.001
            obj.rigid_body.restitution = 0.1
            obj.rigid_body.friction = 0.8
            obj.rigid_body.linear_damping = 0.5
            obj.rigid_body.angular_damping = 0.8
        scene.frame_set(1)
        for frame in range(2, self._sim_frames + 1):
            scene.frame_set(frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        final_matrices = {
            obj: obj.evaluated_get(depsgraph).matrix_world.copy()
            for obj in active
        }
        for obj in active:
            self._make_active(obj)
            bpy.ops.rigidbody.object_remove()
            obj.matrix_world = final_matrices[obj]
        scene.frame_set(1)


class FixedPositions:
    """
    Placement strategy that samples without replacement from a fixed list of ``(x, y)``
    slots.
    """

    def __init__(self, positions: Sequence[Position]):
        self._positions = list(positions)

    def positions(self, count: int) -> list[Position]:
        """
        Return up to *count* positions drawn at random from the fixed list.
        """
        sample_size = min(count, len(self._positions))
        return random.sample(self._positions, sample_size)


class BoundedRandomPositions:
    """
    Placement strategy that draws random ``(x, y)`` points inside a rectangle while
    keeping a minimum spacing between them.
    """

    def __init__(self, x_range: Position, y_range: Position, min_spacing: float, max_attempts: int = 50):
        self._x_range = x_range
        self._y_range = y_range
        self._min_spacing = min_spacing
        self._max_attempts = max_attempts

    def positions(self, count: int) -> list[Position]:
        """
        Return up to *count* positions inside the rectangle, each at least
        *min_spacing* from the others.

        Fewer than *count* points are returned when the spacing constraint cannot be
        met within the attempt budget.
        """
        placed: list[Vector] = []
        result: list[Position] = []
        for _ in range(count):
            candidate = self._sample(placed)
            if candidate is None:
                continue
            placed.append(candidate)
            result.append((candidate.x, candidate.y))
        return result

    def _sample(self, placed: list[Vector]) -> Vector | None:
        """
        Return a random point that clears every point in *placed* by *min_spacing*, or
        ``None`` if none was found within *max_attempts* tries.
        """
        for _ in range(self._max_attempts):
            candidate = Vector((
                random.uniform(*self._x_range),
                random.uniform(*self._y_range),
                0.0,
            ))
            if all((candidate - other).length >= self._min_spacing for other in placed):
                return candidate
        return None


def _jittered_aim_point(target: Vector, camera_location: Vector, aim_jitter_ratio: float) -> Vector:
    """
    Return a look-at point offset from *target* by a random amount proportional to the
    camera-to-target distance, so the subject is framed off-centre.

    Returns *target* unchanged when *aim_jitter_ratio* is zero.
    """
    if not aim_jitter_ratio:
        return target
    distance = (camera_location - target).length
    magnitude = distance * aim_jitter_ratio
    return target + Vector((
        random.uniform(-magnitude, magnitude),
        random.uniform(-magnitude, magnitude),
        0.0,
    ))


class OrbitCamera:
    """
    Camera placement that orbits the target at a random radius, elevation, and azimuth.
    """

    def __init__(self, radius_range: Position, elevation_range: Position):
        self._radius_range = radius_range
        self._elevation_range = elevation_range

    def place(self, camera: bpy.types.Object, target: Vector, aim_jitter_ratio: float = 0.0) -> None:
        """
        Put *camera* on a random point of an orbit around *target* and aim it back at
        the (optionally jittered) target.

        :param camera: The camera object to move.
        :param target: The point the orbit is centred on.
        :param aim_jitter_ratio: Fraction of the target distance by which to offset the
            look-at point.
        """
        radius = random.uniform(*self._radius_range)
        elevation = math.radians(random.uniform(*self._elevation_range))
        azimuth = math.radians(random.uniform(0, 360))
        camera.location = Vector((
            target.x + radius * math.cos(elevation) * math.cos(azimuth),
            target.y + radius * math.cos(elevation) * math.sin(azimuth),
            target.z + radius * math.sin(elevation),
        ))
        aim_point = _jittered_aim_point(target, camera.location, aim_jitter_ratio)
        camera.rotation_euler = (aim_point - camera.location).to_track_quat("-Z", "Y").to_euler()


class BackgroundRandomizer(SceneEffect):
    """
    Scene effect that swaps each target plane's materials for those of a randomly
    chosen donor plane at the start of every group.
    """

    def __init__(self, target_planes: Sequence[str], source_planes: Sequence[str]):
        self._target_planes = list(target_planes)
        self._source_planes = list(source_planes)

    def on_group_start(self) -> None:
        """
        Replace each target plane's materials with those of a random donor plane.
        """
        sources = [bpy.data.objects.get(name) for name in self._source_planes]
        sources = [obj for obj in sources if obj is not None and len(obj.data.materials) > 0]
        if not sources:
            print("No donor planes with materials found")
            return
        for target_name in self._target_planes:
            target = bpy.data.objects.get(target_name)
            if target is None:
                continue
            source = random.choice(sources)
            target.data.materials.clear()
            for material in source.data.materials:
                target.data.materials.append(material)


class HdrRotator(SceneEffect):
    """
    Scene effect that sets a random X rotation on the mapping node feeding the world
    environment texture, changing the lighting direction each frame.
    """

    def __init__(self, node_name: str):
        self._node_name = node_name
        self._mapping: bpy.types.Node | None = None

    def _find_mapping(self) -> bpy.types.Node | None:
        """
        Return the named world-shader node, or ``None`` if the world has no node tree.
        """
        world = bpy.context.scene.world
        if world is None or not world.use_nodes:
            return None
        return world.node_tree.nodes.get(self._node_name)

    def on_frame(self, camera: bpy.types.Object, target: Vector) -> None:
        """
        Set a random X rotation on the world texture's mapping node.
        """
        if self._mapping is None:
            self._mapping = self._find_mapping()
        if self._mapping is not None:
            self._mapping.inputs["Rotation"].default_value[0] = math.radians(random.uniform(0, 180))


class RayCastProjector:
    """
    Projects an object to its tight 2-D bounding box by casting rays against the
    object's mesh geometry.

    A coarse box from the projected vertices bounds the search; rays are then scanned
    inward from each edge to find the exact pixel extent.

    .. note:: Because it tests mesh geometry directly, it ignores Blender's viewport
        and render visibility flags.
    """

    def __init__(self, camera: bpy.types.Object, width: int, height: int, margin: int = 15):
        self._camera = camera
        self._width = width
        self._height = height
        self._margin = margin

    def _pixel_to_ray(self, px: int, py: int) -> tuple[Vector, Vector]:
        """
        Return the world-space origin and direction of the viewing ray through pixel
        ``(px, py)``.
        """
        cam = self._camera
        focal_pixels = (cam.data.lens / cam.data.sensor_width) * self._width
        dir_cam = Vector((
            ((px + 0.5) - self._width * 0.5) / focal_pixels,
            (self._height * 0.5 - (py + 0.5)) / focal_pixels,
            -1.0,
        ))
        dir_cam.normalize()
        return cam.matrix_world.translation.copy(), cam.matrix_world.to_3x3() @ dir_cam

    def _coarse_bbox(self, obj: bpy.types.Object) -> BoundingBox | None:
        """
        Return the pixel box spanning the object's projected vertices, used as the
        search region, or ``None`` if no vertex is in front of the camera.
        """
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        cam_inv = self._camera.matrix_world.inverted()
        focal_pixels = (self._camera.data.lens / self._camera.data.sensor_width) * self._width
        xs: list[float] = []
        ys: list[float] = []
        for v in mesh.vertices:
            local = cam_inv @ (obj.matrix_world @ v.co)
            if local.z >= 0:
                continue
            xs.append((local.x / -local.z) * focal_pixels + self._width * 0.5)
            ys.append(-(local.y / -local.z) * focal_pixels + self._height * 0.5)
        evaluated.to_mesh_clear()
        if not xs:
            return None
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    def project(self, obj: bpy.types.Object) -> BoundingBox | None:
        """
        Return the tight pixel bounding box of *obj*, or ``None`` if it is off-screen
        or fully behind the camera.
        """
        coarse = self._coarse_bbox(obj)
        if coarse is None:
            return None

        m = self._margin
        cx0, _cy0, cx1, _cy1 = coarse
        sx0 = max(0, int(cx0) - m)
        sx1 = min(self._width - 1, int(cx1) + m)

        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated = obj.evaluated_get(depsgraph)
        mat_inv = obj.matrix_world.inverted()
        mat_inv_3x3 = mat_inv.to_3x3()

        def hits(px: int, py: int) -> bool:
            origin_w, dir_w = self._pixel_to_ray(px, py)
            result, _, _, _ = evaluated.ray_cast(
                mat_inv @ origin_w,
                (mat_inv_3x3 @ dir_w).normalized(),
            )
            return result

        x0 = None
        for x in range(sx0, sx1 + 1):
            if any(hits(x, y) for y in range(self._height)):
                x0 = x
                break
        if x0 is None:
            return None

        x1 = x0
        for x in range(sx1, x0 - 1, -1):
            if any(hits(x, y) for y in range(self._height)):
                x1 = x
                break

        y0 = None
        for y in range(self._height):
            if any(hits(x, y) for x in range(x0, x1 + 1)):
                y0 = y
                break
        if y0 is None:
            return None

        y1 = y0
        for y in range(self._height - 1, y0 - 1, -1):
            if any(hits(x, y) for x in range(x0, x1 + 1)):
                y1 = y
                break

        return (float(x0), float(y0), float(x1 + 1), float(y1 + 1))

    def in_frame(self, bbox: BoundingBox) -> bool:
        """
        Return whether *bbox* lies fully within the image.
        """
        x0, y0, x1, y1 = bbox
        return x0 >= 0 and y0 >= 0 and x1 <= self._width and y1 <= self._height


def bbox_overlap_ratio(a: BoundingBox, b: BoundingBox) -> float:
    """
    Return the intersection area of *a* and *b* divided by the smaller box's area.
    """
    inter_x0 = max(a[0], b[0])
    inter_y0 = max(a[1], b[1])
    inter_x1 = min(a[2], b[2])
    inter_y1 = min(a[3], b[3])
    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return 0.0
    intersection = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / min(area_a, area_b)


class Annotator:
    """
    Turns projected bounding boxes into class-tagged annotations, dropping boxes that
    fall outside the frame, lack a class mapping, or overlap an accepted box.
    """

    def __init__(self, projector: RayCastProjector, class_map: dict[str, int],
                 reject_overlaps: bool, overlap_threshold: float):
        self._projector = projector
        self._class_map = class_map
        self._reject_overlaps = reject_overlaps
        self._overlap_threshold = overlap_threshold

    def annotate(self, bodies: Sequence[bpy.types.Object]) -> list[Annotation]:
        """
        Return one ``(class_id, bbox)`` per body that projects into frame and has a
        known class, skipping overlapping boxes when overlap rejection is enabled.
        """
        annotations: list[Annotation] = []
        accepted: list[BoundingBox] = []
        for body in bodies:
            bbox = self._projector.project(body)
            if bbox is None or not self._projector.in_frame(bbox):
                continue
            if self._reject_overlaps and any(
                bbox_overlap_ratio(bbox, other) > self._overlap_threshold for other in accepted
            ):
                continue
            class_id = self._class_map.get(body.get("original_name", body.name))
            if class_id is None:
                print(f"No class mapping for '{body.name}'")
                continue
            annotations.append((class_id, bbox))
            accepted.append(bbox)
        return annotations


class YoloLabelWriter:
    """
    Writes the per-frame image paths, YOLO label files, and the run's ``classes.txt``.
    """

    def __init__(self, output_dir: Path, width: int, height: int, scene_name: str):
        self._images_dir = output_dir / "images"
        self._labels_dir = output_dir / "labels"
        self._root = output_dir
        self._width = width
        self._height = height
        self._scene_name = scene_name
        self._images_dir.mkdir(parents=True, exist_ok=True)
        self._labels_dir.mkdir(parents=True, exist_ok=True)

    def _base_name(self, frame_index: int) -> str:
        """
        Return the ``<scene>_<index>`` stem shared by a frame's image and label files.
        """
        return f"{self._scene_name}_{frame_index:06d}"

    def image_path(self, frame_index: int) -> str:
        """
        Return the path the renderer should write frame *frame_index* to.
        """
        return str(self._images_dir / f"{self._base_name(frame_index)}.png")

    def write(self, frame_index: int, annotations: Sequence[Annotation]) -> None:
        """
        Write the YOLO label file for one frame.

        :param frame_index: Index of the frame being labelled.
        :param annotations: ``(class_id, bbox)`` pairs in pixel coordinates.
        """
        lines = []
        for class_id, (x0, y0, x1, y1) in annotations:
            center_x = ((x0 + x1) / 2) / self._width
            center_y = ((y0 + y1) / 2) / self._height
            box_width = (x1 - x0) / self._width
            box_height = (y1 - y0) / self._height
            lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}")

        path = self._labels_dir / f"{self._base_name(frame_index)}.txt"
        path.write_text("\n".join(lines) + "\n")

    def write_classes(self, class_names: list[str]) -> None:
        """
        Write ``classes.txt`` with one class name per line.
        """
        path = self._root / "classes.txt"
        path.write_text("\n".join(class_names) + "\n")


def derive_class_names(class_map: dict[str, int], provided: list[str] | None) -> list[str]:
    """
    Return *provided* if given, otherwise ``class_0`` up to the highest id in
    *class_map*.
    """
    if provided is not None:
        return provided
    highest = max(class_map.values(), default=-1)
    return [f"class_{index}" for index in range(highest + 1)]


class DatasetGenerator:
    """
    Runs a full dataset-generation pass: load the scene, place groups of objects,
    render them from several cameras, and write YOLO labels.
    """

    def __init__(self, settings: GenerationSettings, placement: PlacementStrategy,
                 camera: CameraPlacement, effects: Sequence[SceneEffect] = ()):
        """
        :param settings: What to render and where to write it.
        :param placement: Strategy deciding where objects are placed.
        :param camera: Strategy positioning and aiming the camera per frame.
        :param effects: Optional per-run effects run at group and frame boundaries.
        """
        self._settings = settings
        self._placement = placement
        self._camera_placement = camera
        self._effects = list(effects)

        scene_name = Path(settings.scene_path).stem

        self._writer = YoloLabelWriter(
            settings.output_dir,
            settings.render.width,
            settings.render.height,
            scene_name
        )

    def run(self) -> None:
        """
        Render the whole dataset and write its labels and class list.
        """
        self._setup()
        frame_index = 0
        min_group_size, max_group_size = self._settings.objects_per_group
        avg_group_size = (min_group_size + max_group_size) / 2
        groups_per_iteration = math.ceil(len(self._pool) / avg_group_size)
        total = self._settings.iterations * groups_per_iteration * self._settings.cameras_per_group
        print(f"Planned frames (approx): {total}")
        for iteration in range(self._settings.iterations):
            group_size = random.randint(min_group_size, max_group_size)
            print(f"Iteration {iteration + 1}/{self._settings.iterations} (group size {group_size})")
            for group in partition_pool(self._pool, group_size):
                frame_index = self._process_group(group, frame_index)
        self._writer.write_classes(
            derive_class_names(self._settings.class_map, self._settings.class_names)
        )
        print(f"Done. Rendered {frame_index} frames to {self._settings.output_dir}")

    def _setup(self) -> None:
        """
        Load the scene, configure rendering, and build the placement, physics, and
        annotation helpers.
        """
        settings = self._settings
        SceneIO.load(settings.scene_path)
        RenderConfigurator(settings.render).apply()
        self._camera = get_or_create_camera()
        bpy.context.scene.camera = self._camera
        self._terrain = require_object(settings.terrain_object)
        self._snapper = TerrainSnapper(self._terrain)
        self._dropper = PhysicsDropper(self._terrain, settings.physics_sim_frames, settings.physics_ignore) if settings.use_physics_drop else None
        if self._dropper is not None:
            self._dropper.setup()
        self._visibility = Visibility(settings.always_visible)
        self._turrets = TurretMounter(settings.turrets)
        self._projector = RayCastProjector(self._camera, settings.render.width, settings.render.height)
        self._annotator = Annotator(
            self._projector, settings.class_map, settings.reject_overlaps, settings.overlap_threshold
        )
        self._pool = PoolSelector(
            settings.vehicles_collection, settings.object_pool, self._turrets.turret_names()
        ).select()
        print(f"Pool size: {len(self._pool)}")
        protected = self._protected_names()
        self._visibility.hide_unprotected_meshes(protected)
        self._turrets.initialize()
        self._reset_dynamic_objects()

    def _protected_names(self) -> set[str]:
        """
        Return the names that must stay visible: scenery, the pool, turrets, and
        effect-managed objects.
        """
        protected = set(self._settings.always_visible)
        protected |= {obj.name for obj in self._pool}
        protected |= self._turrets.turret_names()
        for effect in self._effects:
            protected |= effect.managed_names()
        return protected

    def _reset_dynamic_objects(self) -> None:
        """
        Hide every pool object and turret so the next group starts from a clean slate.
        """
        self._visibility.hide_many(self._pool)
        self._visibility.hide_many(self._turrets.all_turret_objects())

    def _process_group(self, group: list[bpy.types.Object], frame_index: int) -> int:
        """
        Place one *group*, render it from several cameras, and delete the copies.

        :param group: The objects to place for this batch of frames.
        :param frame_index: The first free frame index.
        :return: The next free frame index after this group.
        """
        for effect in self._effects:
            effect.on_group_start()
        self._reset_dynamic_objects()
        copies = self._place_group(group)
        if not copies:
            return frame_index
        for _ in range(self._settings.cameras_per_group):
            self._render_frame(copies, frame_index)
            frame_index += 1
        for copy in copies:
            TurretMounter.delete_copy(copy)
        return frame_index

    def _place_group(self, group: list[bpy.types.Object]) -> list[bpy.types.Object]:
        """
        Create and place a copy of every body in *group*, optionally settle them with
        physics, and snap them to the terrain.

        :return: The placed body copies.
        """
        coordinates = self._placement.positions(len(group))
        copies: list[bpy.types.Object] = []
        drop_height = 5.0 if self._dropper is not None else 2.0
        for body, (x, y) in zip(group, coordinates):
            copies.append(self._turrets.make_copy(body, x, y, height=drop_height))
        if self._dropper is not None:
            self._dropper.drop(copies)
        self._snapper.snap_all(copies)
        return copies

    def _render_frame(self, bodies: list[bpy.types.Object], frame_index: int) -> None:
        """
        Aim the camera at the group's centre, run the effects, render the image, and
        write the frame's labels.
        """
        target = sum((body.location for body in bodies), Vector()) / len(bodies)
        aim_jitter_ratio = SINGLE_OBJECT_OFFSET_RATIO if len(bodies) == 1 else 0.0
        self._camera_placement.place(self._camera, target, aim_jitter_ratio)
        for effect in self._effects:
            effect.on_frame(self._camera, target)
        bpy.context.scene.render.filepath = self._writer.image_path(frame_index)
        bpy.ops.render.render(write_still=True)
        annotations = self._annotator.annotate(bodies)
        if annotations:
            self._writer.write(frame_index, annotations)
        print(f"Frame {frame_index:06d}: {len(annotations)} labels")
