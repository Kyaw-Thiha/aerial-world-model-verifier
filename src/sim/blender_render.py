"""Blender renderer adapter (Track I, Step 1 candidate D).

Runs **inside Blender's Python** (``blender -b -P src/sim/blender_render.py -- <job.json>``),
so it imports only ``bpy``, ``numpy`` and the standard library. The scene is procedural and
seeded, so a scene is a small integer plus a seed and the frames can be regenerated exactly.

Why this adapter exists: the verifier needs (i) a pinhole camera whose pose is *known*, not
inferred, and (ii) image content with enough texture and horizon for a gravity estimator to
work on. Both are constructed here, no physics engine involved: the flight's truth comes
from ``src/sim/flight.py`` and this file only draws it.

Frame conventions: Blender cameras look along local -Z with +Y up, while the project uses
OpenCV RDF (x right, y down, z optical axis). The conversion is therefore
``R_blender = R_wc @ diag(1, -1, -1)``, and a point on the optical axis lands on the pixel
grid centre.

Job file (JSON)::

    {"poses": "poses.npy",      # (M,4,4) camera-to-world, OpenCV RDF
     "out_dir": "frames",
     "width": 640, "height": 360, "hfov_deg": 90.0,
     "scene_seed": 7, "scene": "outdoor",
     "engine": "CYCLES", "device": "GPU", "samples": 24,
     "frame_names": ["0000.png", ...]}   # optional; default %04d.png
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _cv_to_blender(R_wc: np.ndarray) -> np.ndarray:
    D = np.diag([1.0, -1.0, -1.0])
    return np.asarray(R_wc, dtype=np.float64) @ D


def _matrix_world(pose: np.ndarray):
    """4x4 camera-to-world in OpenCV RDF -> Blender ``mathutils.Matrix``.

    Assigning a *numpy* array to ``object.matrix_world`` does not round-trip (Blender reads
    it column-major, so the translation ends up in the rotation block and the camera renders
    from the origin). Building a ``mathutils.Matrix`` from nested rows does round-trip, and
    the assignment below asserts it.
    """
    from mathutils import Matrix

    T = np.eye(4)
    T[:3, :3] = _cv_to_blender(pose[:3, :3])
    T[:3, 3] = pose[:3, 3]
    return Matrix([list(row) for row in T])


# --------------------------------------------------------------------------- scene
def build_scene(seed: int = 7, kind: str = "outdoor", sun_energy: float | None = None,
                sky_strength: float | None = None, view_transform: str = "Filmic",
                layout_dict: dict | None = None):
    """Procedural outdoor scene from ``src/sim/scene.py::layout`` (shared with Isaac Sim).

    Deterministic in ``seed``. Content is chosen for a gravity estimator, not for beauty:
    a checker ground gives a perspective texture gradient, blocks give vertical edges
    (vanishing points), a hill ring gives a crisp horizon silhouette and the sky gives an
    unambiguous up.
    """
    import bpy
    import math as _m

    if layout_dict is None:            # local use: build from the shared module
        from src.sim.scene import layout
        lay = layout(seed)
    else:                              # in a renderer container: the job carries the layout
        lay = layout_dict
    sun_energy = lay["sun_energy"] if sun_energy is None else sun_energy
    sky_strength = lay["sky_strength"] if sky_strength is None else sky_strength

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    # --- ground: checker texture, large enough that the horizon is always inside frame
    bpy.ops.mesh.primitive_plane_add(size=lay["ground_m"], location=(0.0, 0.0, 0.0))
    ground = bpy.context.object
    ground.name = "ground"
    mat = bpy.data.materials.new("ground_mat")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.9
    checker = nt.nodes.new("ShaderNodeTexChecker")
    checker.inputs["Scale"].default_value = 1.0 / lay["checker_m"]
    checker.inputs["Color1"].default_value = (0.10, 0.13, 0.08, 1.0)
    checker.inputs["Color2"].default_value = (0.26, 0.24, 0.18, 1.0)
    coord = nt.nodes.new("ShaderNodeTexCoord")
    nt.links.new(coord.outputs["Object"], checker.inputs["Vector"])
    nt.links.new(checker.outputs["Color"], bsdf.inputs["Base Color"])
    ground.data.materials.append(mat)

    block_mat = bpy.data.materials.new("block_mat")
    block_mat.use_nodes = True
    b = block_mat.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.42, 0.40, 0.36, 1.0)
    b.inputs["Roughness"].default_value = 0.8
    leaf = bpy.data.materials.new("leaf_mat")
    leaf.use_nodes = True
    leaf.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.10, 0.22, 0.08, 1.0)
    hill = bpy.data.materials.new("hill_mat")
    hill.use_nodes = True
    hill.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.16, 0.18, 0.20, 1.0)

    for ob in lay["objects"]:
        if ob["kind"] == "block":
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(ob["x"], ob["y"], ob["h"] / 2.0))
            o = bpy.context.object
            o.scale = (ob["w"], ob["d"], ob["h"])
            o.data.materials.append(block_mat)
        elif ob["kind"] == "tree":
            bpy.ops.mesh.primitive_cone_add(radius1=ob["w"] * 1.4, depth=ob["h"],
                                            location=(ob["x"], ob["y"], ob["h"] / 2.0))
            bpy.context.object.data.materials.append(leaf)
        else:  # hill: a shallow cone reads as a ridge at 300 m
            bpy.ops.mesh.primitive_cone_add(radius1=ob["w"] / 2.0, depth=ob["h"],
                                            location=(ob["x"], ob["y"], ob["h"] / 2.0))
            bpy.context.object.data.materials.append(hill)

    # --- sun + sky
    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 60.0))
    sun = bpy.context.object
    sun.data.energy = float(sun_energy)
    sun.data.angle = _m.radians(0.6)
    sun.rotation_euler = (_m.radians(90.0 - lay["sun_elevation_deg"]), 0.0,
                          _m.radians(lay["sun_azimuth_deg"]))
    world = bpy.data.worlds.new("sky")
    scene.world = world
    world.use_nodes = True
    wnt = world.node_tree
    bg = wnt.nodes["Background"]
    sky = wnt.nodes.new("ShaderNodeTexSky")
    try:
        sky.sky_type = "NISHITA"
        sky.sun_elevation = _m.radians(lay["sun_elevation_deg"])
        sky.sun_rotation = _m.radians(lay["sun_azimuth_deg"])
    except Exception:
        sky.sky_type = "HOSEK_WILKIE"
    wnt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = float(sky_strength)

    # --- camera
    cam_data = bpy.data.cameras.new("cam")
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = 36.0
    cam_data.lens = 18.0
    cam_data.clip_start = 0.05
    cam_data.clip_end = 4000.0
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    try:
        scene.view_settings.view_transform = str(view_transform)
        scene.view_settings.look = "None"
    except Exception:
        scene.view_settings.view_transform = "Standard"
    return scene, cam


def set_intrinsics(cam, width: int, height: int, hfov_deg: float) -> None:
    """Square-pixel pinhole, principal point at the image centre (Blender's convention).

    ``lens = (sensor_width / 2) / tan(hfov / 2)`` with ``sensor_fit = HORIZONTAL``; the
    vertical field follows from the aspect ratio, and ``fx == fy`` in pixels because the
    sensor is square-pixel by construction here. ``width``/``height`` are recorded by the
    caller's intrinsics; Blender's own pixel centre is at ``(W/2, H/2)`` in continuous
    coordinates, which is ``(W-1)/2`` in pixel indices, the same ``cx`` ``pinhole_K`` uses.
    """
    cam.data.sensor_fit = "HORIZONTAL"
    cam.data.sensor_width = 36.0
    cam.data.lens = (cam.data.sensor_width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    cam.data.shift_x = 0.0
    cam.data.shift_y = 0.0


def configure_render(scene, width: int, height: int, engine: str, samples: int,
                     device: str = "GPU", job_exposure: float = 0.0,
                     denoise: bool = False) -> None:
    import bpy

    try:
        scene.view_settings.exposure = float(job_exposure)
    except Exception:
        pass
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    engine = engine.upper()
    if engine == "CYCLES":
        scene.render.engine = "CYCLES"
        scene.cycles.samples = samples
        scene.cycles.use_denoising = bool(denoise)
        scene.cycles.max_bounces = 4
        scene.cycles.device = "GPU" if device.upper() == "GPU" else "CPU"
        if scene.cycles.device == "GPU":
            prefs = bpy.context.preferences.addons["cycles"].preferences
            for backend in ("OPTIX", "CUDA"):
                try:
                    prefs.compute_device_type = backend
                    prefs.get_devices()
                    devs = [d for d in prefs.devices]
                    for d in devs:
                        d.use = d.type != "CPU"
                    if any(d.use for d in devs):
                        break
                except Exception:
                    continue
    else:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
        try:
            scene.eevee.taa_render_samples = samples
        except Exception:
            pass


def verify_projection(job: dict, n_frames: int = 5, n_points: int = 40) -> dict:
    """Check the render skipped: does Blender's camera project exactly like our pinhole model?

    The ground truth of the whole track is "this camera pose produced this image", so the
    pose->pixel model has to be the renderer's own, not a plausible-looking one. Blender's
    ``world_to_camera_view`` is the renderer's answer; ``src/sim/flight.pinhole_K`` plus the
    pose is ours. They are compared on the scene's own object tops, which are never at
    infinity and span the whole image.

    The residuals are expected to be at most half a pixel, which is the difference between
    Blender's pixel-centre convention (``(W/2, H/2)`` in continuous coordinates) and the
    pixel-index convention ``K`` uses (``(W-1)/2``).
    """
    import bpy
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector

    poses = np.load(job["poses"])
    if hasattr(poses, "files"):
        poses = poses["poses"]
    scene, cam = build_scene(seed=int(job.get("scene_seed", 7)),
                             layout_dict=job.get("scene") if isinstance(job.get("scene"), dict) else None)
    W, H = int(job["width"]), int(job["height"])
    hfov = float(job.get("hfov_deg", 90.0))
    set_intrinsics(cam, W, H, hfov)
    # the repo's pinhole model, written out here because Blender's Python has no scipy and
    # the point of this check is to compare against *this* formula
    f = 0.5 * W / np.tan(np.radians(hfov) / 2.0)
    K = np.array([[f, 0.0, (W - 1) / 2.0], [0.0, f, (H - 1) / 2.0], [0.0, 0.0, 1.0]])

    def project(pose, point):
        R, p0 = pose[:3, :3], pose[:3, 3]
        pc = R.T @ (np.asarray(point) - p0)
        if pc[2] <= 1e-6:
            return np.array([np.nan, np.nan])
        return (K @ pc)[:2] / pc[2]
    objs = scene["objects"] if isinstance(scene, dict) else None  # not used; layout is in job
    lay = job.get("scene") or {}
    pts = np.array([[o["x"], o["y"], o["h"]] for o in lay.get("objects", [])][:n_points])
    if len(pts) == 0:
        pts = np.array([[10.0, 0.0, 5.0], [-10.0, 5.0, 15.0], [0.0, 20.0, 2.0]])
    out = []
    for i in range(min(n_frames, len(poses))):
        cam.matrix_world = _matrix_world(poses[i])
        bpy.context.view_layer.update()
        blender_px, ours_px, keep = [], [], []
        for j, p in enumerate(pts):
            co = world_to_camera_view(scene, cam, Vector((float(p[0]), float(p[1]), float(p[2]))))
            if co.z <= 0:
                continue
            bx, by = co.x * W, (1.0 - co.y) * H
            px = project(poses[i], p)
            if not np.isfinite(px).all():
                continue
            blender_px.append((bx, by))
            ours_px.append(px)
            keep.append(j)
        d = np.linalg.norm(np.asarray(blender_px) - np.asarray(ours_px), axis=1)
        out.append({"frame": i, "n": int(len(d)),
                    "max_px": float(d.max()) if len(d) else None,
                    "median_px": float(np.median(d)) if len(d) else None})
    return {"per_frame": out,
            "max_px": max((o["max_px"] or 0.0) for o in out) if out else None,
            "n_frames": len(out)}


def render_job(job: dict) -> dict:
    import bpy

    poses = np.load(job["poses"])
    if hasattr(poses, "files"):          # npz with the GT bundle: the poses are one key
        poses = poses["poses"]
    out_dir = job["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    scene, cam = build_scene(seed=int(job.get("scene_seed", 7)),
                             layout_dict=job.get("scene") if isinstance(job.get("scene"), dict) else None,
                             sun_energy=job.get("sun_energy"),
                             sky_strength=job.get("sky_strength"),
                             view_transform=job.get("view_transform", "Filmic"))
    set_intrinsics(cam, int(job["width"]), int(job["height"]), float(job.get("hfov_deg", 90.0)))
    configure_render(scene, int(job["width"]), int(job["height"]),
                     job.get("engine", "CYCLES"), int(job.get("samples", 24)),
                     job.get("device", "GPU"), float(job.get("exposure", -3.0)),
                     bool(job.get("denoise", False)))
    names = job.get("frame_names")
    times = []
    for i, pose in enumerate(poses):
        mw = _matrix_world(pose)
        cam.matrix_world = mw
        if i == 0:
            got = np.array(cam.matrix_world)
            if not np.allclose(got, np.array(mw), atol=1e-6):
                raise RuntimeError("camera matrix_world did not round-trip; refusing to render")
        name = names[i] if names else f"{i:04d}.png"
        scene.render.filepath = os.path.join(out_dir, name)
        t0 = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        times.append(time.perf_counter() - t0)
    dt = np.asarray(times)
    return {
        "n_frames": int(len(poses)),
        "width": int(job["width"]), "height": int(job["height"]),
        "engine": scene.render.engine,
        "device": job.get("device", "GPU"),
        "samples": int(job.get("samples", 24)),
        "seconds_total": float(dt.sum()),
        "seconds_per_frame_median": float(np.median(dt)),
        "fps": float(len(poses) / dt.sum()),
        "blender_version": bpy.app.version_string,
        "gpu": _gpu_name(),
    }


def _gpu_name() -> str:
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip().splitlines()[0] if out.stdout.strip() else "unknown"
    except Exception:
        return "unknown"


def main() -> int:
    argv = sys.argv
    idx = argv.index("--") if "--" in argv else len(argv)
    args = argv[idx + 1:]
    if not args:
        print("usage: blender -b -P src/sim/blender_render.py -- job.json")
        return 2
    with open(args[0]) as fh:
        job = json.load(fh)
    if job.get("verify_projection"):
        result = verify_projection(job, n_frames=int(job.get("verify_frames", 5)))
        print("VERIFY_RESULT " + json.dumps(result))
        return 0
    result = render_job(job)
    out = job.get("result_json") or os.path.join(job["out_dir"], "..", "render_timing.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2)
    print("RENDER_RESULT " + json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
