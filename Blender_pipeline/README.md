# Synthetic Dataset Generator (Blender)

Generates YOLO-format object-detection datasets by placing vehicle models into five different Blender scenes, rendering them from many camera angles, and writing bounding-box labels automatically.

This pipeline was developed for the master's thesis <THESIS_TITLE> and produced the **Blender-raw** dataset, which is available on [Roboflow](<ROBOFLOW_URL>). The parameter values used for that dataset are documented in the [technical documentation (PDF)](<PDF_PATH>). This README covers setup and usage. The internal mechanisms (generation loop, turret mounting, bounding-box computation, camera models and controlled occlusion) are described in detail in the [technical documentation (PDF)](<PDF_PATH>).

This guide is written for **Windows** users.

---

## What's in this folder

| File | Purpose |
| --- | --- |
| `blender_dataset_core.py` | Shared engine. Contains all the reusable logic. **Do not run this directly.** |
| `blender_generator_first.py` | Scene 1 — uneven muddy terrain with bare autumnal trees, water and rocks |
| `blender_generator_second.py` | Scene 2 — tall-grass meadow with a stream, bordered by sand, gravel and boulders |
| `blender_generator_third.py` | Scene 3 — dense woodland with rock formations, a beaten path and old asphalt |
| `blender_generator_fourth.py` | Scene 4 — hilly snow landscape crossed by a railway line |
| `blender_generator_fifth.py` | Scene 5 — studio scene with textured backdrop planes, a camera on a fixed path and an occluding distractor |
| `list_collections_objects.py` | Helper for listing object names / building config dictionaries |
| `visualizer.py` | Draws YOLO boxes on rendered images for a quick sanity check |
| `<POOL_FILES>` | Vehicle pools, class maps and turret rosters used for the published dataset. They are pasted into the scene scripts before a run. |

> **Important:** All five scene scripts import `blender_dataset_core.py`. They must stay **in the same folder** as the core file. Each scene script adds its own folder to Python's path automatically, so no extra setup is needed as long as they sit together.

---

## Requirements

- **Windows 10 or 11**
- **Blender 5.1.1**. Other versions have not been tested. Blender ships with its own Python, so you do **not** need to install Python separately.
- A **GPU** is strongly recommended. The GPU backend is set in `RenderSettings` (see GPU section below).
- The scene `.blend` files referenced by each script. They are **not included** in this repository because of their size and the licenses of the assets they contain. They are available from the author upon reasonable request.

---

## One-time setup

### 1. Place the scripts

Put all the `.py` files in one folder, for example:

```
C:\Dev\bpy_vehicles\scripts\
```

### 2. Point the scripts at your scene files

All five scene scripts resolve their `.blend` file relative to one setting. Open `blender_dataset_core.py` and set `SCENE_ROOT` to the folder that holds your scene files:

```python
SCENE_ROOT = Path(r"C:\Dev\bpy_vehicles\resources")
```

The default is a `resources` folder next to the scripts. Each scene script then names its own file near the top, so you normally don't touch these:

```python
SCENE_FILE = "First_scene.blend"
```

The script builds `scene_path` as `SCENE_ROOT / SCENE_FILE`. If a scene file has a different name or lives elsewhere, change that script's `SCENE_FILE` (a bare filename is joined to `SCENE_ROOT`; a full path is used as-is).

### 3. Fill in the vehicle pool

In the repository, every scene script starts with an empty pool:

```python
OBJECT_POOL = []
TURRETS = {}
CLASS_MAP = {name: 0 for name in OBJECT_POOL}
```

Before a run, paste the entries for the vehicles you want to render from the pool files into `OBJECT_POOL`, `TURRETS` and `CLASS_MAP`. The default `CLASS_MAP` assigns class 0 to every vehicle, so it must be replaced by the real class map.

### 4. (Optional) Add Blender to your PATH

Blender's executable is usually here:

```
C:\Program Files\Blender Foundation\Blender 5.1\blender.exe
```

If you add that folder to your Windows `PATH` environment variable, you can type `blender` instead of the full path. Otherwise, use the full path in the commands below.

---

## Running a scene

The scene script loads its own `.blend` file, so you only point Blender at the script. Open **Command Prompt** or **PowerShell**, then:

**If Blender is on your PATH:**

```bat
blender --background --python blender_generator_first.py
```

**If it is not on your PATH (use the full path to blender.exe):**

```bat
"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe" --background --python blender_generator_first.py
```

- `--background` runs Blender with no UI (faster, no window).
- `--python` runs the script.

Swap `first` for `second`, `third`, `fourth`, or `fifth` to run a different scene.

### Run from the script folder

It's easiest to `cd` into the folder first:

```bat
cd C:\Dev\bpy_vehicles\scripts
blender --background --python blender_generator_fifth.py
```

---

## Output

Each run writes into an `output` folder created next to the scripts. File names start with the name of the scene file, so the output of different scenes can share one folder:

```
output\
  images\        First_scene_000000.png, First_scene_000001.png, ...
  labels\        First_scene_000000.txt, First_scene_000001.txt, ...   (YOLO format)
  classes.txt    one class name per line
```

Each label line is YOLO format: `class_id center_x center_y width height` (all normalized 0–1). Frames in which no vehicle could be labeled get an image but no label file.

> Re-running the same scene reuses the same file names and **overwrites** frames starting from `000000`. Rename or move `output` between runs if you want to keep previous results.

---

## About the labels

Bounding boxes are computed from the scene geometry, not from the rendered image. The vertices of each vehicle are projected into the image to obtain a coarse box, which is then tightened to pixel accuracy by ray casts against the vehicle mesh. Since the rays test only the vehicle itself, a box always covers the **full extent of the vehicle, including parts hidden behind distractors or scenery**. Vehicles reaching beyond the image edge are clipped to the frame, and vehicles lying entirely outside it are dropped. Distractors are never labeled.

For the published Blender-raw dataset, these labels served as pre-annotations only. They were reviewed and corrected by hand in Roboflow. Boxes were reduced to the visible part of the vehicle, and vehicles that were almost entirely occluded or no longer recognizable were removed. **Labels from a fresh run therefore do not follow the convention of the published dataset** until they have been corrected in the same way. Use `visualizer.py` to inspect the raw labels.

---

## How a run works

Each run cycles through the vehicle pool in iterations. At the start of every iteration, a group size is drawn from `objects_per_group`, and the shuffled pool is split into groups of that size, so that every vehicle appears exactly once per iteration. Each group is placed in the scene and rendered `cameras_per_group` times before the next group follows.

The scripts never move the pooled vehicles themselves. Every placement works on copies, which are deleted after the group has been rendered. Turreted vehicles receive a random turret azimuth and an independent random heading, both drawn over the full circle. Hull and turret are then joined, so the label covers both as one object. The camera aims at the center of the placed group. If the group holds a single vehicle, the aim point is shifted by a random offset, so that the vehicle does not always appear in the image center.

The five scenes use two configurations of the same engine:

| | Pipeline A (Scenes 1–4) | Pipeline B (Scene 5) |
| --- | --- | --- |
| Placement | Hand-picked spawn slots, a random subset per group (`FixedPositions`) | One vehicle at a fixed position (`BoundedRandomPositions` with coinciding bounds) |
| Camera | Random orbit around the group, with radius and elevation drawn from configured ranges (`OrbitCamera`) | Camera on a fixed path at a random height, with small pitch and yaw offsets and a random zoom (`PathLookAtCamera`) |
| Ground alignment | Short rigid-body drop onto the terrain, then snapping to the surface | Snapping to the flat floor only |
| Effects | `HdrRotator`, which rotates the lighting HDRI before every frame and thereby changes the direction of the light | `BackgroundRandomizer` and `OrbitingDistractor` |

Pipeline A varies the environment, lighting and viewpoint. Pipeline B holds the lighting fixed and varies background, viewpoint and occlusion instead.

---

## Customizing a scene

Everything you'd normally tweak lives at the top of each scene script.

### `SETTINGS` — what to render

```python
SETTINGS = GenerationSettings(
    scene_path=SCENE_ROOT / SCENE_FILE,
    output_dir=Path(__file__).parent / "output",
    vehicles_collection="Vehicles",  # collection holding the spawnable vehicles
    object_pool=OBJECT_POOL,         # which objects can be spawned
    always_visible={...},            # scenery that should always render
    turrets=TURRETS,                 # body -> turret pairs
    class_map=CLASS_MAP,             # object name -> class id
    terrain_object="...",            # surface objects are dropped onto
    use_physics_drop=True,           # settle vehicles with a rigid-body drop
    physics_sim_frames=60,           # simulation frames for the drop
    physics_ignore=[...],            # vehicles excluded from the drop
    objects_per_group=(1, 8),        # inclusive (min, max) vehicles per group
    cameras_per_group=2,             # renders taken per placement
    iterations=2,                    # how many times to reshuffle and repeat
)
```

Total frames ≈ `iterations × (pool size ÷ mean group size, rounded up) × cameras_per_group`. The script prints this estimate at the start of a run.

`objects_per_group` is an inclusive range. A new group size is drawn from it at the start of every iteration, and `(1, 1)` produces images with exactly one vehicle. Scene 5 should always run with `(1, 1)`, because its distractor is placed relative to a single vehicle.

Further optional settings:

- `class_names=["ifv", "mbt", ...]` sets human-readable class names in `classes.txt`. If omitted, names default to `class_0`, `class_1`, and so on.
- `reject_overlaps=True` drops a box that overlaps an already accepted box by more than `overlap_threshold`. It is off by default.
- `render=RenderSettings(...)` changes resolution, sample count or GPU backend.

### `PLACEMENT` — where objects go

- `FixedPositions([...])` — choose from a fixed list of (x, y) slots (scenes 1–4).
- `BoundedRandomPositions(x_range, y_range, min_spacing)` — random spots inside a rectangle (scene 5).

### `CAMERA` — how the camera moves

- `OrbitCamera(radius_range, elevation_range)` — orbits around the objects (scenes 1–4).
- `PathLookAtCamera(...)` — a single camera that slides along a fixed poly-line path and always looks at the object. Defined in `blender_generator_fifth.py` (scene 5).

### `EFFECTS` — optional extras (scenes 1–5)

- `HdrRotator(node_name)` — before every frame, rotates the mapping node that feeds the world texture by a random angle, which changes the direction of the light (scenes 1–4). The node name is set in the scene script (`HDR_NODE_NAME`).
- `BackgroundRandomizer(...)` — randomizes the backdrop planes each group (scene 5).
- `OrbitingDistractor(...)` — places a clutter object on the object's orbit, between the camera and the target, for occlusion. Never labeled. Defined in `blender_generator_fifth.py` (scene 5).

---

## Setting up a new scene

The engine does not need to know what a scene contains, as long as the `.blend` file follows these conventions:

- Each scene is one self-contained `.blend` file.
- All spawnable vehicle meshes sit in one collection, by convention named `Vehicles`, placed outside the visible part of the scene. They stay hidden until they are spawned.
- Static scenery such as terrain, rocks and vegetation consists of ordinary objects outside that collection. At startup, every mesh that is neither pooled nor a turret nor managed by an effect is hidden, so scenery that should render must be listed in `always_visible`.
- One mesh acts as the terrain onto which vehicles are placed (`terrain_object`).
- Vehicles with a rotatable turret consist of two meshes, a hull and a turret, positioned correctly relative to each other. The origin of the turret mesh must lie on its rotation axis, otherwise the turret does not rotate in place.
- For `HdrRotator`, the world shader needs a mapping node feeding the lighting HDRI, named as passed to the effect.

Hull and turret are both listed in the pool and in the class map with the same class id. The turret roster maps each hull to its turret. A turret named there is never spawned on its own, and a hull without an entry is placed as a single mesh:

```python
OBJECT_POOL = [
    "vehicle_a_hull",
    "vehicle_a_turret",
    "vehicle_b",
]

CLASS_MAP = {
    "vehicle_a_hull": 3,
    "vehicle_a_turret": 3,
    "vehicle_b": 4,
}

TURRETS = {
    "vehicle_a_hull": "vehicle_a_turret",
}
```

---

## Helper: listing object names

When setting up a new scene you often need the exact object names from a Blender collection. The helper prints them, builds a ready-to-paste object list, detects body/turret pairs, and emits a class map. A sub-collection holding exactly two meshes is proposed as a body/turret pair, with the larger mesh taken as the body.

```bat
blender "C:\Dev\bpy_vehicles\resources\Fifth_scene.blend" --background --python list_collections_objects.py -- "Vehicles" --mesh-only --class-name mbt --class-id 0
```

Notes:
- The collection name (`"Vehicles"`) and all flags come **after** the `--` separator.
- `--mesh-only` skips empties, lights, and cameras.
- `--class-name` / `--class-id` set the single category for the run.
- `--out names.txt` also writes the names to a file.
- Run it with no collection name to print every available collection.

---

## Troubleshooting

**`Collection 'Vehicles' not found`**
The collection name in `vehicles_collection` doesn't match the scene. Run `list_collections_objects.py` with no collection name to see the available collections.

**`Object '...' not found`**
The `terrain_object` doesn't exist in that `.blend`. Names are case-sensitive and must match Blender exactly, including suffixes like `.001`.

**Fewer vehicles than expected**
Names in `object_pool` and `always_visible` that don't exist in the scene are skipped without an error. Compare the printed `Pool size: N` with the number of vehicles you expect. A hull/turret pair with a missing partner is reported as `Missing turret pair`, and a vehicle without a class id as `No class mapping`.

**Renders are extremely slow / using the CPU**
GPU rendering is configured in `blender_dataset_core.py` inside `RenderConfigurator`. The backend is set by `compute_device_type` in `RenderSettings`. Choose the one that matches your GPU:
- NVIDIA RTX: `"OPTIX"`
- NVIDIA older: `"CUDA"`
- AMD: `"HIP"`
- Intel Arc: `"ONEAPI"`

You can override it per scene:

```python
from blender_dataset_core import RenderSettings
SETTINGS = GenerationSettings(
    ...,
    render=RenderSettings(compute_device_type="CUDA"),
)
```

**`'blender' is not recognized`**
Blender isn't on your PATH. Use the full quoted path to `blender.exe`, or add its folder to PATH.

**Paths with spaces**
Always wrap paths containing spaces in double quotes in the command line, e.g. `"C:\Program Files\..."`.

**Nothing appears in `output`**
Confirm the script printed `Done. Rendered N frames`. If `N` is 0, the object pool may be empty or all objects fell outside the camera frame — check the console output for warnings.

---

## Quick start

```bat
cd C:\Dev\bpy_vehicles\scripts
"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe" --background --python blender_generator_first.py
```

Open `output\images` to see the renders and `output\labels` for the matching YOLO labels.

---

## Reproducibility

A run is fully determined by its scene file and the values in its scene script. Bit-identical reproduction of individual images is not a goal. A repeated run with the same settings reproduces the distribution of the dataset, not the exact images.

---

## Asset credits

The scenes and vehicles are built from third-party assets: vehicle models from Sketchfab, HDRI panoramas, assets and textures from BlenderKit, and tile and camouflage textures created with the Camo Pattern Generator. All sources with their authors and licenses are listed in the [technical documentation (PDF)](<PDF_PATH>).

---

## Citation

If you use this pipeline or the Blender-raw dataset, please cite:

```
<CITATION>
```

---

## License

<LICENSE>
