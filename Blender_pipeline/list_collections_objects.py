"""
List all object names inside a Blender collection ("directory").

Scene structure assumed:
    <Collection>                e.g. one of the 5 top-level dirs
        - <object dir>          per-object sub-collection
              - <object>        the actual mesh
        - <object dir>
              - <object>
        ...

`collection.all_objects` walks that nesting recursively, so passing the
top-level collection name returns every object beneath it.

Usage (from a terminal):
    blender path/to/scene.blend --background \
        --python list_collections_objects.py -- "CollectionName"

Optional flags (after the collection name):
    --mesh-only          only list MESH objects (skip empties, lights, etc.)
    --out names.txt      also write the names to a file (one per line)
    --class-id N         class id for every object (default 0)
    --class-name NAME    category label, e.g. mbt / ifv / transport (default "object")

Because each run covers a single category, every object - bodies and turrets
alike - is mapped to the same class id in the emitted CLASS_MAP.

If no collection name is given, all available collections are printed.
"""

import sys
import bpy


def get_script_args() -> list[str]:
    """
    Return the arguments passed after the ``--`` separator, per Blender convention.
    """
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1:]
    return []


def iter_collections(col):
    """
    Yield *col* and every collection nested beneath it.
    """
    yield col
    for child in col.children:
        yield from iter_collections(child)


def _volume(obj) -> float:
    """
    Return the bounding-box volume of *obj*.
    """
    d = obj.dimensions
    return d.x * d.y * d.z


def build_turret_map(collection_name: str, mesh_only: bool = True):
    """
    Pair the two objects of every two-object sub-collection as body and turret.

    The object with the larger bounding-box volume is treated as the body and the
    other as its turret. Returns ``(pairs, info)`` where *pairs* maps body name to
    turret name and *info* maps body name to
    ``(body_volume, turret_volume, body_z, turret_z)`` for manual verification.
    """
    top = bpy.data.collections.get(collection_name)
    if top is None:
        return {}, {}

    pairs: dict[str, str] = {}
    info: dict[str, tuple] = {}

    for sub in iter_collections(top):
        objs = [o for o in sub.objects if (o.type == "MESH" or not mesh_only)]
        if len(objs) != 2:
            continue
        a, b = objs
        body, turret = (a, b) if _volume(a) >= _volume(b) else (b, a)
        pairs[body.name] = turret.name
        info[body.name] = (_volume(body), _volume(turret),
                           body.matrix_world.translation.z,
                           turret.matrix_world.translation.z)
    return pairs, info


def list_objects(collection_name: str, mesh_only: bool = False) -> list[str]:
    """
    Return the de-duplicated object names beneath *collection_name*, printing the
    available collections and returning an empty list when it does not exist.

    :param collection_name: Name of the collection to walk.
    :param mesh_only: Whether to skip non-mesh objects.
    """
    col = bpy.data.collections.get(collection_name)
    if col is None:
        print(f"Collection '{collection_name}' not found.")
        print("Available collections:")
        for c in bpy.data.collections:
            print(f"  - {c.name}")
        return []

    names: list[str] = []
    seen: set[str] = set()
    for obj in col.all_objects:
        if mesh_only and obj.type != "MESH":
            continue
        if obj.name not in seen:
            seen.add(obj.name)
            names.append(obj.name)
    return names


def main():
    """
    Print the objects, a paste-ready name list, the detected turret map, and a class
    map for the collection named on the command line.
    """
    args = get_script_args()

    if not args:
        print("No collection name given. Available collections:")
        for c in bpy.data.collections:
            print(f"  - {c.name}  ({len(c.all_objects)} objects)")
        return

    collection_name = args[0]
    mesh_only = "--mesh-only" in args

    out_path = None
    if "--out" in args:
        idx = args.index("--out")
        if idx + 1 < len(args):
            out_path = args[idx + 1]

    class_id = 0
    if "--class-id" in args:
        idx = args.index("--class-id")
        if idx + 1 < len(args):
            class_id = int(args[idx + 1])

    class_name = "object"
    if "--class-name" in args:
        idx = args.index("--class-name")
        if idx + 1 < len(args):
            class_name = args[idx + 1]

    names = list_objects(collection_name, mesh_only=mesh_only)
    if not names:
        return

    kind = "MESH objects" if mesh_only else "objects"
    print(f"\n{len(names)} {kind} in '{collection_name}':\n")
    for n in names:
        print(n)

    print("\n# --- as a Python list ---")
    print("[")
    for n in names:
        print(f'    "{n}",')
    print("]")

    pairs, info = build_turret_map(collection_name, mesh_only=True)
    print(f"\n# --- TURRETS (body: turret) — {len(pairs)} two-object collection(s) ---")
    if pairs:
        print("# Heuristic: larger volume = body. Verify and swap if a pair looks wrong.")
        print("{")
        for body, turret in pairs.items():
            bvol, tvol, bz, tz = info[body]
            print(f'    "{body}": "{turret}",'
                  f'  # body vol={bvol:.3f} z={bz:.2f} | turret vol={tvol:.3f} z={tz:.2f}')
        print("}")
    else:
        print("# (none found)")

    print(f"\n# --- CLASS_MAP (single category '{class_name}' -> id {class_id}) ---")
    print("{")
    for n in names:
        print(f'    "{n}": {class_id},')
    print("}")
    print(f"\n# classes.txt (one line): {class_name}")

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(names) + "\n")
        print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
