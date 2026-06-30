import argparse
import json
import os
import queue
import shutil
import threading

import cv2
import numpy as np
import torch as th
import yaml

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.objects import DatasetObject
from omnigibson.utils.ui_utils import KeyboardEventHandler


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
DATASET_SCENES_DIR = os.path.join(REPO_ROOT, "datasets", "behavior-1k-assets", "scenes")
DATASET_OBJECTS_DIR = os.path.join(REPO_ROOT, "datasets", "behavior-1k-assets", "objects")
BASE_SCENE_MODEL = "Pomaria_2_int_data_collection_2"
OUTPUT_SCENE_MODEL = f"{BASE_SCENE_MODEL}_3"
BASE_SCENE_DIR = os.path.join(DATASET_SCENES_DIR, BASE_SCENE_MODEL)
BASE_SCENE_PATH = os.path.join(BASE_SCENE_DIR, "json", f"{BASE_SCENE_MODEL}_best.json")
SCENE_LOAD_EXCLUDE_OBJECTS = {
    # Pomaria_2_int: this object crashes during FIRE emitter initialization:
    # AssertionError: .../armchair_bslhmj_0/base_link/emitter local transform is not orthogonal.
    "armchair_bslhmj_0",
}

DEFAULT_ADD_POSITION = [0.0, 0.0, 0.6]
DEFAULT_ADD_ORIENTATION = [0, 0, 0, 1]
STRUCTURE_CATEGORIES = {"floors", "ceilings", "walls"}
TASK_ROLE_KEYS = ("objects", "assets")
TASK_ROLE_ALIASES = {
    "object": "objects",
    "obj": "objects",
    "pickable": "objects",
    "asset": "assets",
    "fixed": "assets",
    "surface": "assets",
    "table": "assets",
}
LEGACY_TASK_ROLE_MAP = {
    "pickable_objects": "objects",
    "pick_surfaces": "assets",
    "place_surfaces": "assets",
    "background_objects": "assets",
}


def build_config():
    config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    config["scene"]["scene_model"] = BASE_SCENE_MODEL
    config["scene"]["scene_file"] = scene_load_path()
    # Load the original scene as-is. This keeps the creator generic when switching
    # between BEHAVIOR scenes instead of maintaining a per-scene category whitelist.
    config["scene"]["load_object_categories"] = None
    # config["scene"]["load_object_categories"] = ["floors", "ceilings", "walls"]
    config["objects"] = []
    config["robots"] = []
    return config


def output_scene_dir(output_scene_model):
    return os.path.join(DATASET_SCENES_DIR, output_scene_model)


def scene_save_path(output_scene_model):
    return os.path.join(output_scene_dir(output_scene_model), "json", f"{output_scene_model}_best.json")


def task_config_path(save_path):
    return os.path.join(os.path.dirname(os.path.abspath(save_path)), "task_config.json")


def base_task_config_path():
    return os.path.join(os.path.dirname(BASE_SCENE_PATH), "task_config.json")


def scene_load_path():
    if not SCENE_LOAD_EXCLUDE_OBJECTS:
        return BASE_SCENE_PATH

    with open(BASE_SCENE_PATH, "r") as f:
        scene_data = json.load(f)

    removed = []
    object_registry = scene_data.get("state", {}).get("registry", {}).get("object_registry", {})
    init_info = scene_data.get("objects_info", {}).get("init_info", {})
    for name in SCENE_LOAD_EXCLUDE_OBJECTS:
        if object_registry.pop(name, None) is not None:
            removed.append(name)
        init_info.pop(name, None)

    if not removed:
        return BASE_SCENE_PATH

    filtered_dir = os.path.join("/tmp", "behavior_scene_creator")
    os.makedirs(filtered_dir, exist_ok=True)
    filtered_path = os.path.join(filtered_dir, f"{BASE_SCENE_MODEL}_filtered_best.json")
    with open(filtered_path, "w") as f:
        json.dump(scene_data, f, indent=4)
    print(f"[scene] Loading original scene with excluded object(s): {removed}")
    print(f"[scene] Temporary filtered scene: {filtered_path}")
    return filtered_path


def prepare_output_scene_directory(save_path):
    scene_dir = os.path.dirname(os.path.dirname(os.path.abspath(save_path)))
    base_layout_dir = os.path.join(BASE_SCENE_DIR, "layout")
    output_layout_dir = os.path.join(scene_dir, "layout")
    os.makedirs(os.path.join(scene_dir, "json"), exist_ok=True)
    if os.path.isdir(base_layout_dir):
        shutil.copytree(base_layout_dir, output_layout_dir, dirs_exist_ok=True)
    return scene_dir


def original_map_pixel_from_world(xy, image_size):
    xy = th.as_tensor(xy, dtype=th.float32)
    point_wrt_map = xy / 0.01 + image_size / 2.0
    dims = 0 if point_wrt_map.dim() == 1 else 1
    return th.flip(point_wrt_map, dims=(dims,)).int()


def should_rasterize_object_on_trav_map(obj, task_config):
    category = getattr(obj, "category", None)
    if category in STRUCTURE_CATEGORIES:
        return False
    if obj.name in task_config["objects"]:
        return False
    if obj.name in task_config["assets"]:
        return True
    return bool(getattr(obj, "fixed_base", False))


def rasterize_object_footprint(trav_img, obj, padding_px=3):
    try:
        aabb_min, aabb_max = obj.aabb
    except Exception as exc:
        print(f"[trav-map-warn] Could not read AABB for {obj.name}: {exc}")
        return False

    height, width = trav_img.shape
    corners_world = th.tensor(
        [
            [aabb_min[0], aabb_min[1]],
            [aabb_min[0], aabb_max[1]],
            [aabb_max[0], aabb_min[1]],
            [aabb_max[0], aabb_max[1]],
        ],
        dtype=th.float32,
    )
    corners_map = original_map_pixel_from_world(corners_world, height)
    rows = corners_map[:, 0].cpu().numpy()
    cols = corners_map[:, 1].cpu().numpy()
    r0 = max(0, int(np.min(rows)) - padding_px)
    r1 = min(height - 1, int(np.max(rows)) + padding_px)
    c0 = max(0, int(np.min(cols)) - padding_px)
    c1 = min(width - 1, int(np.max(cols)) + padding_px)
    if r0 > r1 or c0 > c1:
        return False
    cv2.rectangle(trav_img, (c0, r0), (c1, r1), color=0, thickness=-1)
    return True


def save_current_traversability_maps(scene_dir, scene, task_config):
    layout_dir = os.path.join(scene_dir, "layout")
    os.makedirs(layout_dir, exist_ok=True)

    n_floors = getattr(scene.trav_map, "n_floors", 1) or 1
    rasterized_names = []
    for floor in range(n_floors):
        no_obj_path = os.path.join(layout_dir, f"floor_trav_no_obj_{floor}.png")
        if not os.path.exists(no_obj_path):
            source_no_obj_path = os.path.join(BASE_SCENE_DIR, "layout", f"floor_trav_no_obj_{floor}.png")
            if os.path.exists(source_no_obj_path):
                shutil.copy2(source_no_obj_path, no_obj_path)

        base_img = cv2.imread(no_obj_path, cv2.IMREAD_GRAYSCALE)
        if base_img is None:
            source_trav_path = os.path.join(BASE_SCENE_DIR, "layout", f"floor_trav_{floor}.png")
            base_img = cv2.imread(source_trav_path, cv2.IMREAD_GRAYSCALE)
        if base_img is None:
            print(f"[trav-map-warn] Could not find base traversability image for floor {floor}; skipping map save.")
            continue

        trav_img = base_img.copy()
        for obj in scene.objects:
            if not should_rasterize_object_on_trav_map(obj, task_config):
                continue
            if rasterize_object_footprint(trav_img, obj):
                rasterized_names.append(obj.name)

        trav_path = os.path.join(layout_dir, f"floor_trav_{floor}.png")
        cv2.imwrite(trav_path, trav_img)

        # Keep the common variants synchronized with the current object-aware map.
        for variant in ("floor_trav_open_door", "floor_trav_no_door"):
            variant_path = os.path.join(layout_dir, f"{variant}_{floor}.png")
            if os.path.exists(variant_path):
                cv2.imwrite(variant_path, trav_img)

        print(f"[trav-map] Saved current object-aware traversability map: {trav_path}")

    if rasterized_names:
        print(f"[trav-map] Rasterized {len(set(rasterized_names))} fixed object(s): {sorted(set(rasterized_names))}")
    else:
        print("[trav-map] No fixed objects were rasterized into floor_trav maps.")


def print_controls(save_path):
    print("")
    print("=" * 72)
    print("Simple scene creator")
    print("Viewport:")
    print("  Shift + left mouse drag: move objects after adding them")
    print("  S: save current scene layout")
    print("  L: reload saved scene layout")
    print("  ESC: quit")
    print("")
    print("Asset Browser window:")
    print("  Search asset categories / model ids")
    print("  Click Add to insert an object at the default position")
    print("Scene Roles window:")
    print("  Mark scene objects as Object or Asset, then click Save")
    print("")
    print("Terminal commands:")
    print("  add <category> <model> [name] [x y z] [scale]")
    print("    e.g. add coffee_table cjjayg pick_table 0 -1.3 0.2 0.5")
    print("    e.g. add bottle_of_lemon_sauce iyijeb sauce 0 -1.3 0.55")
    print("  categories [keyword]")
    print("  models <category>")
    print("  search <keyword>")
    print("  addidx <search_index> [name] [x y z] [scale]")
    print("  pose <name> <x> <y> <z> [qx qy qz qw]")
    print("  delete <name>")
    print("  list")
    print("  mark <role> <name>      roles: object, asset")
    print("  unmark <role|all> <name>")
    print("  task")
    print("  save")
    print("  load")
    print("  quit")
    print("")
    print(f"Base scene: {BASE_SCENE_PATH}")
    print(f"Output scene dir: {os.path.dirname(os.path.dirname(os.path.abspath(save_path)))}")
    print(f"Save path: {save_path}")
    print(f"Task config: {task_config_path(save_path)}")
    print("=" * 72)
    print("")


def safe_keyboard_reset():
    if KeyboardEventHandler._CALLBACK_ID is None:
        KeyboardEventHandler.KEYBOARD_CALLBACKS = dict()
    else:
        KeyboardEventHandler.reset()


def set_camera_pose():
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([0.0, -4.2, 3.2], dtype=th.float32),
        orientation=th.tensor([0.3827, 0.0, 0.0, 0.9239], dtype=th.float32),
    )


def scan_asset_index():
    asset_index = {}
    if not os.path.isdir(DATASET_OBJECTS_DIR):
        print(f"[warn] Object asset directory not found: {DATASET_OBJECTS_DIR}")
        return asset_index

    for category_entry in sorted(os.scandir(DATASET_OBJECTS_DIR), key=lambda entry: entry.name):
        if not category_entry.is_dir():
            continue
        models = []
        for model_entry in sorted(os.scandir(category_entry.path), key=lambda entry: entry.name):
            if not model_entry.is_dir():
                continue
            usd_path = os.path.join(model_entry.path, "usd", f"{model_entry.name}.usd")
            encrypted_usd_path = os.path.join(model_entry.path, "usd", f"{model_entry.name}.encrypted.usd")
            if os.path.exists(usd_path) or os.path.exists(encrypted_usd_path):
                models.append(model_entry.name)
        if models:
            asset_index[category_entry.name] = models
    return asset_index


def print_categories(asset_index, keyword=None, limit=80):
    categories = sorted(asset_index)
    if keyword:
        categories = [category for category in categories if keyword.lower() in category.lower()]
    print(f"[categories] showing {min(len(categories), limit)}/{len(categories)}")
    for category in categories[:limit]:
        print(f"  {category} ({len(asset_index[category])} models)")


def print_models(asset_index, category, limit=80):
    models = asset_index.get(category)
    if not models:
        print(f"[warn] Category not found: {category}")
        return
    print(f"[models] {category}: showing {min(len(models), limit)}/{len(models)}")
    for model in models[:limit]:
        print(f"  {model}")


def search_assets(asset_index, keyword, limit=40):
    keyword = keyword.lower()
    results = []
    for category, models in sorted(asset_index.items()):
        category_match = keyword in category.lower()
        for model in models:
            if category_match or keyword in model.lower():
                results.append((category, model))
    print(f"[search] '{keyword}': showing {min(len(results), limit)}/{len(results)}")
    for idx, (category, model) in enumerate(results[:limit]):
        print(f"  {idx}: {category} / {model}")
    return results[:limit]


def search_asset_results(asset_index, keyword, limit=80):
    keyword = keyword.lower().strip()
    results = []
    for category, models in sorted(asset_index.items()):
        category_match = not keyword or keyword in category.lower()
        for model in models:
            if category_match or keyword in model.lower():
                results.append((category, model))
                if len(results) >= limit:
                    return results
    return results


def create_asset_browser(asset_index, command_queue):
    ui = lazy.omni.ui
    window = ui.Window("Asset Browser", width=520, height=620)
    search_model = ui.SimpleStringModel("")
    result_limit = 80
    pending_rebuild = {"value": False}

    def enqueue_add(category, model, role=None):
        command = f"addrole {role} {category} {model}" if role is not None else f"add {category} {model}"
        command_queue.put(command)

    def request_rebuild():
        # Do not rebuild the UI directly from a click callback; omni.ui does not
        # allow clearing / adding children during event dispatch or draw.
        pending_rebuild["value"] = True

    def rebuild():
        pending_rebuild["value"] = False
        query = search_model.get_value_as_string()
        results = search_asset_results(asset_index, query, limit=result_limit)
        window.frame.clear()
        with window.frame:
            with ui.VStack(spacing=6, height=0):
                ui.Label("BEHAVIOR Asset Browser")
                with ui.HStack(height=28, spacing=4):
                    ui.StringField(model=search_model)
                    ui.Button("Search", width=70, clicked_fn=request_rebuild)
                ui.Label(f"{len(results)} result(s). Add as Object or Asset, then drag it in the viewport.")
                with ui.ScrollingFrame(height=500):
                    with ui.VStack(spacing=3, height=0):
                        if not results:
                            ui.Label("No results. Try e.g. bottle, table, sofa, apple.", height=24)
                        for category, model in results:
                            with ui.HStack(height=28, spacing=4):
                                ui.Label(f"{category} / {model}", width=285, height=24)
                                ui.Button(
                                    "Object",
                                    width=72,
                                    clicked_fn=lambda c=category, m=model: enqueue_add(c, m, "object"),
                                )
                                ui.Button(
                                    "Asset",
                                    width=64,
                                    clicked_fn=lambda c=category, m=model: enqueue_add(c, m, "asset"),
                                )

    rebuild()
    return {"window": window, "pending_rebuild": pending_rebuild, "rebuild": rebuild}


def process_asset_browser(asset_browser):
    if asset_browser["pending_rebuild"]["value"]:
        asset_browser["rebuild"]()


def create_scene_roles_window(get_scene, task_config, save_path, save_fn):
    ui = lazy.omni.ui
    window = ui.Window("Scene Roles", width=560, height=620)
    search_model = ui.SimpleStringModel("")
    pending_rebuild = {"value": False}

    def request_rebuild():
        pending_rebuild["value"] = True

    def set_role(name, role):
        scene = get_scene()
        unmark_task_object(task_config, "all", name)
        mark_task_object(task_config, scene, role, name)
        request_rebuild()

    def clear_role(name):
        unmark_task_object(task_config, "all", name)
        request_rebuild()

    def role_label(name):
        labels = []
        if name in task_config["objects"]:
            labels.append("Object")
        if name in task_config["assets"]:
            labels.append("Asset")
        return "/".join(labels) if labels else "-"

    def rebuild():
        pending_rebuild["value"] = False
        scene = get_scene()
        cleanup_task_config(task_config, scene)
        query = search_model.get_value_as_string().lower().strip()
        objects = sorted(scene.objects, key=lambda item: item.name)
        if query:
            objects = [
                obj
                for obj in objects
                if query in obj.name.lower()
                or query in str(getattr(obj, "category", "")).lower()
                or query in str(getattr(obj, "model", "")).lower()
            ]

        window.frame.clear()
        with window.frame:
            with ui.VStack(spacing=6, height=0):
                ui.Label("Scene Roles")
                with ui.HStack(height=28, spacing=4):
                    ui.StringField(model=search_model)
                    ui.Button("Refresh", width=70, clicked_fn=request_rebuild)
                    ui.Button("Save", width=58, clicked_fn=save_fn)
                ui.Label(
                    f"{len(objects)} scene item(s). Object = movable/graspable, Asset = fixed/support/place target."
                )
                with ui.ScrollingFrame(height=500):
                    with ui.VStack(spacing=3, height=0):
                        for obj in objects:
                            name = obj.name
                            category = getattr(obj, "category", "")
                            model = getattr(obj, "model", "")
                            title = f"{name}  [{category}/{model}]"
                            with ui.HStack(height=28, spacing=4):
                                ui.Label(role_label(name), width=56, height=24)
                                ui.Label(title, width=290, height=24)
                                ui.Button(
                                    "Object",
                                    width=72,
                                    clicked_fn=lambda n=name: set_role(n, "object"),
                                )
                                ui.Button(
                                    "Asset",
                                    width=62,
                                    clicked_fn=lambda n=name: set_role(n, "asset"),
                                )
                                ui.Button(
                                    "Clear",
                                    width=56,
                                    clicked_fn=lambda n=name: clear_role(n),
                                )

    rebuild()
    return {"window": window, "pending_rebuild": pending_rebuild, "rebuild": rebuild}


def process_scene_roles_window(scene_roles_window):
    if scene_roles_window["pending_rebuild"]["value"]:
        scene_roles_window["rebuild"]()


def unique_object_name(scene, requested_name):
    if scene.object_registry("name", requested_name) is None:
        return requested_name

    idx = 1
    while scene.object_registry("name", f"{requested_name}_{idx}") is not None:
        idx += 1
    return f"{requested_name}_{idx}"


def add_dataset_object(scene, category, model, name=None, position=None, orientation=None, scale=None):
    obj_name = unique_object_name(scene, name or f"{category}_{model}")
    obj = DatasetObject(
        name=obj_name,
        category=category,
        model=model,
        scale=scale,
    )
    scene.add_object(obj)
    obj.set_position_orientation(
        position=th.tensor(position or DEFAULT_ADD_POSITION, dtype=th.float32),
        orientation=th.tensor(orientation or DEFAULT_ADD_ORIENTATION, dtype=th.float32),
        frame="scene",
    )
    for _ in range(5):
        og.sim.step()
    print(f"[add] {obj_name}: category={category}, model={model}, position={position or DEFAULT_ADD_POSITION}")
    return obj


def print_scene_objects(scene):
    print("[objects]")
    for obj in sorted(scene.objects, key=lambda item: item.name):
        pos, orn = obj.get_position_orientation()
        print(
            f"  {obj.name}: category={getattr(obj, 'category', None)}, "
            f"model={getattr(obj, 'model', None)}, "
            f"pos={[round(v, 3) for v in pos.tolist()]}, "
            f"orn={[round(v, 3) for v in orn.tolist()]}"
        )


def default_task_config():
    return {key: [] for key in TASK_ROLE_KEYS}


def normalize_task_role(role):
    role = role.lower()
    if role in TASK_ROLE_KEYS:
        return role
    return TASK_ROLE_ALIASES.get(role)


def load_task_config_file(path):
    config = default_task_config()

    with open(path, "r") as f:
        loaded = json.load(f)
    for key in TASK_ROLE_KEYS:
        values = loaded.get(key, [])
        config[key] = list(dict.fromkeys(values))
    for old_key, new_key in LEGACY_TASK_ROLE_MAP.items():
        for name in loaded.get(old_key, []):
            if name not in config[new_key]:
                config[new_key].append(name)
    return config


def load_task_config(save_path):
    path = task_config_path(save_path)
    if os.path.exists(path):
        config = load_task_config_file(path)
        print(f"[task] Loaded task config: {path}")
        return config

    inherited_path = base_task_config_path()
    if os.path.exists(inherited_path):
        config = load_task_config_file(inherited_path)
        print(f"[task] Output task config does not exist yet: {path}")
        print(f"[task] Inherited task config from base scene: {inherited_path}")
        return config

    print(f"[task] No task config found yet: {path}")
    return default_task_config()


def cleanup_task_config(task_config, scene):
    valid_names = {obj.name for obj in scene.objects}
    for key in TASK_ROLE_KEYS:
        task_config[key] = [name for name in task_config[key] if name in valid_names]


def save_task_config(save_path, task_config, scene):
    cleanup_task_config(task_config, scene)
    path = task_config_path(save_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(task_config, f, indent=2)
    print(f"[save] Task config saved to: {path}")


def print_task_config(task_config, scene=None):
    if scene is not None:
        cleanup_task_config(task_config, scene)
    print("[task]")
    for key in TASK_ROLE_KEYS:
        print(f"  {key}:")
        if not task_config[key]:
            print("    []")
            continue
        for name in task_config[key]:
            print(f"    - {name}")


def mark_task_object(task_config, scene, role, name):
    key = normalize_task_role(role)
    if key is None:
        print(f"[usage] Unknown role '{role}'. Use: object, asset")
        return
    if scene.object_registry("name", name) is None:
        print(f"[warn] Object not found: {name}")
        return
    for other_key in TASK_ROLE_KEYS:
        if other_key != key and name in task_config[other_key]:
            task_config[other_key].remove(name)
    if name not in task_config[key]:
        task_config[key].append(name)
    print(f"[task] Marked {name} as {key}")


def unmark_task_object(task_config, role, name):
    keys = TASK_ROLE_KEYS if role.lower() == "all" else (normalize_task_role(role),)
    if None in keys:
        print(f"[usage] Unknown role '{role}'. Use: object, asset, all")
        return
    removed = False
    for key in keys:
        if name in task_config[key]:
            task_config[key].remove(name)
            removed = True
            print(f"[task] Removed {name} from {key}")
    if not removed:
        print(f"[task] {name} was not marked under {role}")


def start_command_thread(command_queue):
    def read_commands():
        while True:
            try:
                command = input("scene> ").strip()
            except EOFError:
                break
            if command:
                command_queue.put(command)
            if command.lower() in {"quit", "q", "exit"}:
                break

    thread = threading.Thread(target=read_commands, daemon=True)
    thread.start()
    return thread


def handle_command(command, scene, save_path, should_quit, asset_index, last_search_results, task_config, request_save):
    parts = command.split()
    if not parts:
        return

    op = parts[0].lower()
    try:
        if op == "add":
            if len(parts) < 3:
                print("[usage] add <category> <model> [name] [x y z] [scale]")
                return
            category, model = parts[1], parts[2]
            name = parts[3] if len(parts) >= 4 else None
            position = [float(v) for v in parts[4:7]] if len(parts) >= 7 else None
            scale = [float(parts[7])] * 3 if len(parts) >= 8 else None
            add_dataset_object(scene, category=category, model=model, name=name, position=position, scale=scale)
        elif op == "addrole":
            if len(parts) < 4:
                print("[usage] addrole <object|asset> <category> <model> [name] [x y z] [scale]")
                return
            role, category, model = parts[1], parts[2], parts[3]
            if normalize_task_role(role) is None:
                print("[usage] addrole <object|asset> <category> <model> [name] [x y z] [scale]")
                return
            name = parts[4] if len(parts) >= 5 else None
            position = [float(v) for v in parts[5:8]] if len(parts) >= 8 else None
            scale = [float(parts[8])] * 3 if len(parts) >= 9 else None
            obj = add_dataset_object(scene, category=category, model=model, name=name, position=position, scale=scale)
            mark_task_object(task_config, scene, role, obj.name)
        elif op == "addidx":
            if len(parts) < 2:
                print("[usage] addidx <search_index> [name] [x y z] [scale]")
                return
            idx = int(parts[1])
            if idx < 0 or idx >= len(last_search_results["value"]):
                print(f"[warn] Search index out of range: {idx}. Run search <keyword> first.")
                return
            category, model = last_search_results["value"][idx]
            name = parts[2] if len(parts) >= 3 else None
            position = [float(v) for v in parts[3:6]] if len(parts) >= 6 else None
            scale = [float(parts[6])] * 3 if len(parts) >= 7 else None
            add_dataset_object(scene, category=category, model=model, name=name, position=position, scale=scale)
        elif op in {"categories", "cats"}:
            keyword = parts[1] if len(parts) >= 2 else None
            print_categories(asset_index, keyword=keyword)
        elif op == "models":
            if len(parts) != 2:
                print("[usage] models <category>")
                return
            print_models(asset_index, parts[1])
        elif op == "search":
            if len(parts) != 2:
                print("[usage] search <keyword>")
                return
            last_search_results["value"] = search_assets(asset_index, parts[1])
        elif op == "pose":
            if len(parts) not in {5, 9}:
                print("[usage] pose <name> <x> <y> <z> [qx qy qz qw]")
                return
            obj = scene.object_registry("name", parts[1])
            if obj is None:
                print(f"[warn] Object not found: {parts[1]}")
                return
            position = th.tensor([float(v) for v in parts[2:5]], dtype=th.float32)
            orientation = (
                th.tensor([float(v) for v in parts[5:9]], dtype=th.float32)
                if len(parts) == 9
                else obj.get_position_orientation()[1]
            )
            obj.set_position_orientation(position=position, orientation=orientation, frame="scene")
            print(f"[pose] {obj.name}: pos={position.tolist()}, orn={orientation.tolist()}")
        elif op in {"delete", "remove", "rm"}:
            if len(parts) != 2:
                print("[usage] delete <name>")
                return
            obj = scene.object_registry("name", parts[1])
            if obj is None:
                print(f"[warn] Object not found: {parts[1]}")
                return
            scene.remove_object(obj)
            unmark_task_object(task_config, "all", parts[1])
            print(f"[delete] {parts[1]}")
        elif op == "list":
            print_scene_objects(scene)
        elif op == "mark":
            if len(parts) != 3:
                print("[usage] mark <role> <name>")
                return
            mark_task_object(task_config, scene, parts[1], parts[2])
        elif op == "unmark":
            if len(parts) != 3:
                print("[usage] unmark <role|all> <name>")
                return
            unmark_task_object(task_config, parts[1], parts[2])
        elif op == "task":
            print_task_config(task_config, scene=scene)
        elif op == "save":
            request_save()
        elif op == "load":
            print("[info] Use keyboard L to reload safely; command reload would replace the active scene object.")
        elif op in {"quit", "q", "exit"}:
            should_quit["value"] = True
        elif op in {"help", "h", "?"}:
            print_controls(save_path)
        else:
            print(f"[warn] Unknown command: {op}. Type help.")
    except Exception as exc:
        print(f"[error] Command failed: {command}\n  {exc}")


def save_scene(save_path, scene, task_config):
    scene_dir = prepare_output_scene_directory(save_path)
    # Dragging / adding objects can invalidate PhysX tensor views. Refresh handles
    # and step a few frames before saving so dump_state reads a valid backend view.
    for _ in range(3):
        og.sim.step()
    og.sim.update_handles()
    for _ in range(2):
        og.sim.step()
    try:
        og.sim.save(json_paths=[save_path])
    except Exception as exc:
        print(f"[warn] First save attempt failed after refresh: {exc}")
        print("[warn] Rebuilding physics handles and retrying once.")
        og.sim.update_handles()
        for _ in range(5):
            og.sim.step()
        og.sim.save(json_paths=[save_path])
    save_task_config(save_path, task_config, scene)
    save_current_traversability_maps(scene_dir, scene, task_config)
    print(f"[save] Scene saved to: {save_path}")
    print(f"[save] Scene directory prepared at: {scene_dir}")


def restore_scene(save_path):
    if not os.path.exists(save_path):
        print(f"[warn] No saved scene found: {save_path}")
        return False

    og.clear()
    og.sim.restore(scene_files=[save_path])
    og.sim.play()
    set_camera_pose()
    print(f"[load] Scene restored from: {save_path}")
    return True


def main(save_path=None, output_scene_model=OUTPUT_SCENE_MODEL, load_existing=False, short_exec=False):
    """
    A tiny visual scene creator for data-collection layouts.

    Run it, drag objects in the viewport, press S to save the layout, and reuse the
    generated JSON with og.sim.restore([...]) in data collection scripts.
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    save_path = os.path.abspath(save_path if save_path is not None else scene_save_path(output_scene_model))
    should_quit = {"value": False}
    pending_save = {"steps_remaining": -1}
    asset_index = scan_asset_index()
    last_search_results = {"value": []}
    task_config = load_task_config(save_path)

    if load_existing and os.path.exists(save_path):
        og.sim.restore(scene_files=[save_path])
        og.sim.play()
        env = None
    else:
        env = og.Environment(configs=build_config())

    for _ in range(30):
        og.sim.step()

    og.sim.enable_viewer_camera_teleoperation()
    set_camera_pose()
    command_queue = queue.Queue()
    start_command_thread(command_queue)

    def active_scene():
        return og.sim.scenes[0] if env is None else env.scene

    def request_save():
        pending_save["steps_remaining"] = 3
        print("[save] Save requested; waiting a few sim steps for PhysX to settle.")

    def on_save():
        request_save()

    def on_load():
        nonlocal env, task_config
        if restore_scene(save_path):
            env = None
            loaded_task_config = load_task_config(save_path)
            task_config.clear()
            task_config.update(loaded_task_config)
            install_keyboard_callbacks()

    def on_quit():
        should_quit["value"] = True

    def install_keyboard_callbacks():
        safe_keyboard_reset()
        KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.S, on_save)
        KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.L, on_load)
        KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.ESCAPE, on_quit)

    asset_browser = create_asset_browser(asset_index, command_queue)
    scene_roles_window = create_scene_roles_window(active_scene, task_config, save_path, on_save)

    install_keyboard_callbacks()
    print_controls(save_path)
    print(f"[assets] Indexed {len(asset_index)} object categories from {DATASET_OBJECTS_DIR}")
    print_task_config(task_config, scene=active_scene())

    steps = 0
    max_steps = 300 if short_exec else -1
    while not should_quit["value"] and steps != max_steps:
        while not command_queue.empty():
            handle_command(
                command_queue.get(),
                active_scene(),
                save_path,
                should_quit,
                asset_index,
                last_search_results,
                task_config,
                request_save,
            )
            scene_roles_window["pending_rebuild"]["value"] = True
        process_asset_browser(asset_browser)
        process_scene_roles_window(scene_roles_window)
        og.sim.step()
        if pending_save["steps_remaining"] >= 0:
            pending_save["steps_remaining"] -= 1
            if pending_save["steps_remaining"] < 0:
                save_scene(save_path, active_scene(), task_config)
                scene_roles_window["pending_rebuild"]["value"] = True
        steps += 1

    safe_keyboard_reset()
    asset_browser["window"].visible = False
    scene_roles_window["window"].visible = False
    og.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", default=None, help="Full path to save / load the scene JSON.")
    parser.add_argument(
        "--output-scene-model",
        default=OUTPUT_SCENE_MODEL,
        help=(
            "Name of the new scene directory created under datasets/behavior-1k-assets/scenes. "
            "Ignored when --save-path is set."
        ),
    )
    parser.add_argument("--load-existing", action="store_true", help="Load --save-path at startup if it exists.")
    args = parser.parse_args()
    main(save_path=args.save_path, output_scene_model=args.output_scene_model, load_existing=args.load_existing)
