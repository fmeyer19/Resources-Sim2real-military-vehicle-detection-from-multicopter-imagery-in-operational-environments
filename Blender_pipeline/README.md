# Synthetic Dataset Generator (Blender)

Generates YOLO-format object-detection datasets by placing vehicle models into five different Blender scenes, rendering them from many camera angles, and writing bounding-box labels automatically.

This guide is written for **Windows** users.

---

## What's in this folder

| File | Purpose |
| --- | --- |
| `blender_dataset_core.py` | Shared engine. Contains all the reusable logic. **Do not run this directly.** |
| `blender_generator_first.py` | Scene 1 — muddy landscape |
| `blender_generator_second.py` | Scene 2 — beach with fences |
| `blender_generator_third.py` | Scene 3 — scattered vegetation |
| `blender_generator_fourth.py` | Scene 4 — grid terrain, large turret roster, single category |
| `blender_generator_fifth.py` | Scene 5 — backdrop planes, orbiting distractor, single camera on a fixed path |
| `list_collections_objects.py` | Helper for listing object names / building config dictionaries |
| `visualizer.py` | Draws YOLO boxes on rendered images for a quick sanity check |

> **Important:** All five scene scripts import `blender_dataset_core.py`. They must stay **in the same folder** as the core file. Each scene script adds its own folder to Python's path automatically, so no extra setup is needed as long as they sit together.

---

## Requirements

- **Windows 10 or 11**
- **Blender 5.x or newer** (5.x recommended). Blender ships with its own Python, so you do **not** need to install Python separately.
- A **GPU** is strongly recommended. The scripts are configured for NVIDIA CUDA by default (see GPU section below).
- The scene `.blend` files referenced by each script.

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

### 3. (Optional) Add Blender to your PATH

Blender's executable is usually here:

```
C:\Program Files\Blender Foundation\Blender 4.2\blender.exe
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
"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe" --background --python blender_generator_first.py
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

Each run writes into an `output` folder created next to the scripts:

```
output\
  images\        000000.png, 000001.png, ...
  labels\        000000.txt, 000001.txt, ...   (YOLO format)
  classes.txt    one class name per line
```

Each label line is YOLO format: `class_id center_x center_y width height` (all normalized 0–1).

> Re-running a scene reuses the same `output` folder and **overwrites** frames starting from `000000`. Rename or move `output` between runs if you want to keep previous results.

---

## Customizing a scene

Everything you'd normally tweak lives at the top of each scene script in three objects.

### `SETTINGS` — what to render

```python
SETTINGS = GenerationSettings(
    scene_path=SCENE_ROOT / SCENE_FILE,
    object_pool=[...],          # which objects can be spawned
    always_visible={...},       # scenery that should always render
    turrets={...},              # body -> turret pairs
    class_map={...},            # object name -> class id
    terrain_object="...",       # surface objects are dropped onto
    objects_per_group=8,        # objects placed per scene setup
    cameras_per_group=4,        # renders taken per placement
    iterations=2,               # how many times to reshuffle and repeat
)
```

Total frames = `iterations × (pool size ÷ objects_per_group, rounded up) × cameras_per_group`.

### `PLACEMENT` — where objects go

- `FixedPositions([...])` — choose from a fixed list of (x, y) slots (scenes 1–4).
- `BoundedRandomPositions(x_range, y_range, min_spacing)` — random spots inside a rectangle (scene 5).

### `CAMERA` — how the camera moves

- `OrbitCamera(radius_range, elevation_range)` — orbits around the objects (scenes 1–4).
- `PathLookAtCamera(...)` — a single camera that slides along a fixed poly-line path and always looks at the object. Defined in `blender_generator_fifth.py` (scene 5).

### `EFFECTS` — optional extras (scene 5)

- `BackgroundRandomizer(...)` — randomizes the backdrop planes each group.
- `OrbitingDistractor(...)` — places a clutter object on the object's orbit, between the camera and the target, for occlusion. Never labeled. Defined in `blender_generator_fifth.py`.

To set human-readable class names in `classes.txt`, pass `class_names=["ifv", "mbt", ...]` inside `GenerationSettings`. If omitted, names default to `class_0`, `class_1`, and so on.

---

## Helper: listing object names

When setting up a new scene you often need the exact object names from a Blender collection. The helper prints them, builds a ready-to-paste object list, detects body/turret pairs, and emits a class map.

```bat
blender "C:\Dev\bpy_vehicles\resources\Fifth-scene.blend" --background --python list_collections_objects.py -- "Vehicles" --mesh-only --class-name mbt --class-id 0
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
A name in `object_pool`, `turrets`, `always_visible`, or `terrain_object` doesn't exist in that `.blend`. Names are case-sensitive and must match Blender exactly, including suffixes like `.001`.

**Renders are extremely slow / using the CPU**
GPU rendering is configured in `blender_dataset_core.py` inside `RenderConfigurator`. The default device type is `CUDA` (NVIDIA). If you have a different GPU, change `compute_device_type` in your `RenderSettings`:
- NVIDIA RTX: `"OPTIX"`
- NVIDIA older: `"CUDA"`
- AMD: `"HIP"`
- Intel Arc: `"ONEAPI"`

You can override it per scene:

```python
from blender_dataset_core import RenderSettings
SETTINGS = GenerationSettings(
    ...,
    render=RenderSettings(compute_device_type="OPTIX"),
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
"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe" --background --python blender_generator_first.py
```

Open `output\images` to see the renders and `output\labels` for the matching YOLO labels.
