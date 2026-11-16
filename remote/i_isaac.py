"""Isaac Sim headless render on Modal (Track I spike, candidate B).

The spike question is narrow: can Isaac Sim, on a cloud GPU we can rent, render a *known*
camera pose sequence headlessly, in a way that fits the track's budget? Not "what is the
prettiest scene Isaac Sim can make".

Two stages, deliberately separated because the API surface moves between releases:

* ``probe``  -- start the app, import the render paths in turn, print versions and the
  attributes that actually exist, and render one frame with whichever path works.
* ``render`` -- run the full pose sequence through the path ``probe`` found and return
  per-frame times.

Both write to the Modal Volume ``awmv-i-data`` (prefix ``spike/``) and return their logs, so
a failure is diagnosable without a second run.

    pixi run modal run remote/i_isaac.py::probe
    pixi run modal run remote/i_isaac.py::render --poses <npz> --tag isaac
"""
from __future__ import annotations

import json
import pathlib

import modal

app = modal.App("awmv-i-isaac")
volume = modal.Volume.from_name("awmv-i-data", create_if_missing=True)

ISAACSIM_VERSION = "6.2.0.0"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("curl", "git", "libgl1", "libglib2.0-0", "libx11-6", "libxext6",
                 "libxrender1", "libxi6", "libxxf86vm1", "libsm6", "libice6",
                 "libegl1", "libvulkan1", "vulkan-tools", "libatomic1", "libxt6",
                 "libxcursor1", "libxrandr2", "libxinerama1", "libxkbcommon0")
    .pip_install("numpy<2", "pillow")
    .run_commands(
        f"pip install 'isaacsim[extscache,all]=={ISAACSIM_VERSION}' "
        "--extra-index-url https://pypi.nvidia.com"
    )
    .env({"ACCEPT_EULA": "Y", "OMNI_KIT_ACCEPT_EULA": "YES"})
)

PROBE = r'''
import json, os, sys, time, traceback
out = {"steps": [], "python": sys.version}
def note(k, v):
    out["steps"].append({k: v})

from isaacsim import SimulationApp
t0 = time.time()
sim_app = SimulationApp({"headless": True, "width": 640, "height": 360})
note("app_started_s", round(time.time() - t0, 2))

import carb, omni.usd
note("kit_version", carb.__file__)
try:
    import omni.kit.app
    note("kit_app_version", omni.kit.app.get_app().get_build_version())
except Exception as e:
    note("kit_app_version_error", repr(e))

# what exists in the replicator module
import omni.replicator.core as rep
api = {}
for name in ("create", "functional", "orchestrator", "writers", "backends", "render_product"):
    obj = getattr(rep, name, None)
    if obj is None:
        api[name] = None
    else:
        api[name] = [a for a in dir(obj) if not a.startswith("_")][:40]
note("replicator_api", api)

sensor_api = {}
for modname in ("isaacsim.sensors.camera", "isaacsim.core.experimental.objects"):
    try:
        mod = __import__(modname, fromlist=["*"])
        sensor_api[modname] = [a for a in dir(mod) if not a.startswith("_")][:40]
    except Exception as e:
        sensor_api[modname] = repr(e)
note("sensor_api", sensor_api)

# build a minimal stage with pxr only: ground, one box, sun, camera
from pxr import Usd, UsdGeom, UsdLux, Gf, Sdf
stage = omni.usd.get_context().get_stage()
if stage is None:
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

def add_cube(path, size, loc):
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    c.CreateExtentAttr([(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)])
    x = UsdGeom.Xformable(c)
    x.AddTranslateOp().Set(Gf.Vec3d(*loc))
    x.AddScaleOp().Set(Gf.Vec3f(*size))
    m = UsdGeom.MaterialBindingAPI.Apply(c.GetPrim())
    return c

def add_plane(path, size, z=0.0):
    p = UsdGeom.Mesh.Define(stage, path)
    h = size / 2.0
    p.CreatePointsAttr([(-h, -h, z), (h, -h, z), (h, h, z), (-h, h, z)])
    p.CreateFaceVertexCountsAttr([4])
    p.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    return p

add_plane("/World/ground", 800.0)
for i, (x, y, z, sx, sy, sz) in enumerate([(20, 5, 5, 8, 8, 10), (-25, 10, 4, 6, 6, 8),
                                           (5, -30, 6, 10, 10, 12), (-40, -20, 3, 5, 5, 6)]):
    add_cube(f"/World/box{i}", (sx, sy, sz), (x, y, z))

sun = UsdLux.DistantLight.Define(stage, "/World/sun")
sun.CreateIntensityAttr(1500.0)
sun.AddRotateXYZOp().Set(Gf.Vec3f(45.0, 0.0, 35.0))
dome = UsdLux.DomeLight.Define(stage, "/World/dome")
dome.CreateIntensityAttr(400.0)

cam = UsdGeom.Camera.Define(stage, "/World/cam")
cam.CreateFocalLengthAttr(18.0)
cam.CreateHorizontalApertureAttr(36.0)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 4000.0))
xf = UsdGeom.Xformable(cam.GetPrim())
xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 12.0))
xf.AddRotateXYZOp().Set(Gf.Vec3f(90.0, 0.0, 0.0))

result = {"stage_ok": True}
# --- attempt 1: replicator + BasicWriter (stable across 4.x-6.x)
try:
    rep.orchestrator.set_capture_on_play(False)
    rp = rep.create.render_product("/World/cam", (640, 360))
    outdir = "/tmp/isaac_probe"
    os.makedirs(outdir, exist_ok=True)
    writer = rep.writers.get("BasicWriter")
    writer.initialize(output_dir=outdir, rgb=True)
    writer.attach(rp)
    t0 = time.time()
    for i in range(3):
        rep.orchestrator.step(rt_subframes=4)
    rep.orchestrator.wait_until_complete()
    result["replicator_basicwriter"] = {
        "ok": True, "seconds_for_3": round(time.time() - t0, 2),
        "files": sorted(os.listdir(outdir))[:10]}
    writer.detach(); rp.destroy()
except Exception as e:
    result["replicator_basicwriter"] = {"ok": False, "error": repr(e),
                                        "trace": traceback.format_exc()[-1500:]}

# --- attempt 2: replicator get_data (in-memory)
try:
    rp2 = rep.create.render_product("/World/cam", (640, 360))
    from omni.replicator.core import annotator_lookup  # noqa: F401
    rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annot.attach([rp2])
    rep.orchestrator.step(rt_subframes=4)
    data = rgb_annot.get_data()
    result["replicator_annotator"] = {"ok": True, "shape": list(getattr(data, "shape", [])),
                                      "dtype": str(getattr(data, "dtype", None))}
    rp2.destroy()
except Exception as e:
    result["replicator_annotator"] = {"ok": False, "error": repr(e),
                                      "trace": traceback.format_exc()[-1200:]}

# --- attempt 3: isaacsim.sensors.camera.Camera
try:
    from isaacsim.sensors.camera import Camera
    c = Camera(prim_path="/World/cam2", resolution=(640, 360), translation=(0, 0, 12))
    c.initialize()
    rgb = c.get_rgba()
    result["sensors_camera"] = {"ok": True, "shape": list(rgb.shape)}
except Exception as e:
    result["sensors_camera"] = {"ok": False, "error": repr(e),
                                "trace": traceback.format_exc()[-1200:]}

print("PROBE_RESULT " + json.dumps({**out, **result}))
sim_app.close()
'''


@app.function(image=image, gpu="L40S", timeout=3600,
              volumes={"/vol": volume})
def probe() -> dict:
    import subprocess
    import sys

    p = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, timeout=3300)
    log = (p.stdout or "") + "\n--stderr--\n" + (p.stderr or "")
    pathlib.Path("/vol/spike").mkdir(parents=True, exist_ok=True)
    pathlib.Path("/vol/spike/isaac_probe.log").write_text(log[-200000:])
    volume.commit()
    out = {"returncode": p.returncode, "log_tail": log[-6000:]}
    for line in log.splitlines():
        if line.startswith("PROBE_RESULT "):
            out["probe"] = json.loads(line[len("PROBE_RESULT "):])
    return out


RENDER = r'''
import json, os, sys, time, traceback
import numpy as np
job = json.load(open(sys.argv[1]))
_npz = np.load(job["poses"])
poses = _npz["poses"] if hasattr(_npz, "files") else _npz
outdir = job["out_dir"]; os.makedirs(outdir, exist_ok=True)
from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": True, "width": int(job["width"]), "height": int(job["height"])})
import omni.usd
from pxr import Usd, UsdGeom, UsdLux, Gf
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
h = job["ground_m"] / 2.0
g = UsdGeom.Mesh.Define(stage, "/World/ground")
g.CreatePointsAttr([(-h,-h,0),(h,-h,0),(h,h,0),(-h,h,0)])
g.CreateFaceVertexCountsAttr([4]); g.CreateFaceVertexIndicesAttr([0,1,2,3])
for i, ob in enumerate(job["objects"]):
    kind = ob["kind"]
    if kind == "block":
        c = UsdGeom.Cube.Define(stage, f"/World/obj{i}")
        c.CreateSizeAttr(1.0)
        x = UsdGeom.Xformable(c); x.AddTranslateOp().Set(Gf.Vec3d(ob["x"], ob["y"], ob["h"]/2.0))
        x.AddScaleOp().Set(Gf.Vec3f(ob["w"], ob["d"], ob["h"]))
    else:
        # cones are not a USD primitive: use a cube silhouette for trees/hills (reported)
        c = UsdGeom.Cube.Define(stage, f"/World/obj{i}")
        c.CreateSizeAttr(1.0)
        x = UsdGeom.Xformable(c); x.AddTranslateOp().Set(Gf.Vec3d(ob["x"], ob["y"], ob["h"]/2.0))
        x.AddScaleOp().Set(Gf.Vec3f(ob["w"], ob["d"], ob["h"]))
sun = UsdLux.DistantLight.Define(stage, "/World/sun")
sun.CreateIntensityAttr(1500.0)
sun.AddRotateXYZOp().Set(Gf.Vec3f(90.0 - job["sun_elevation_deg"], 0.0, job["sun_azimuth_deg"]))
dome = UsdLux.DomeLight.Define(stage, "/World/dome")
dome.CreateIntensityAttr(400.0)
cam = UsdGeom.Camera.Define(stage, "/World/cam")
cam.CreateFocalLengthAttr((36.0/2.0)/np.tan(np.radians(job["hfov_deg"])/2.0))
cam.CreateHorizontalApertureAttr(36.0)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 4000.0))
xf = UsdGeom.Xformable(cam.GetPrim())
mx = xf.AddTransformOp()

import omni.replicator.core as rep
rep.orchestrator.set_capture_on_play(False)
size = (int(job["width"]), int(job["height"]))
rp = None
for maker in (lambda: rep.create.render_product("/World/cam", size),
              lambda: rep.functional.create.render_product("/World/cam", size)):
    try:
        rp = maker(); break
    except Exception:
        continue
if rp is None:
    raise RuntimeError("no render_product constructor found")
from PIL import Image
get = None
mode = None
for reg in (getattr(rep, "AnnotatorRegistry", None), getattr(rep, "annotators", None)):
    if reg is None:
        continue
    try:
        rgb = reg.get_annotator("rgb")
        rgb.attach([rp])
        get = lambda: np.asarray(rgb.get_data())
        mode = "annotator"
        break
    except Exception:
        continue
if get is None:
    mode = "basicwriter"
    writer = rep.writers.get("BasicWriter")
    writer.initialize(output_dir=outdir, rgb=True)
    writer.attach(rp)

D = np.diag([1.0, -1.0, -1.0])
times = []
for i, T in enumerate(poses):
    R = T[:3, :3] @ D
    m = Gf.Matrix4d(float(R[0,0]), float(R[0,1]), float(R[0,2]), 0.0,
                    float(R[1,0]), float(R[1,1]), float(R[1,2]), 0.0,
                    float(R[2,0]), float(R[2,1]), float(R[2,2]), 0.0,
                    float(T[0,3]), float(T[1,3]), float(T[2,3]), 1.0)
    mx.Set(m)
    t0 = time.time()
    rep.orchestrator.step(rt_subframes=int(job.get("rt_subframes", 4)))
    times.append(time.time() - t0)
    if get is not None:
        img = get()
        if img.ndim == 3 and img.shape[-1] >= 3:
            Image.fromarray(img[..., :3]).save(os.path.join(outdir, f"{i:04d}.png"))
rep.orchestrator.wait_until_complete()
dt = np.asarray(times)
print("RENDER_RESULT " + json.dumps({
    "mode": mode, "n_frames": len(poses), "seconds_total": float(dt.sum()),
    "fps": float(len(poses)/dt.sum()), "seconds_per_frame_median": float(np.median(dt)),
    "files": sorted(os.listdir(outdir))[:5]}))
sim_app.close()
'''


@app.function(image=image, gpu="L40S", timeout=2 * 3600, volumes={"/vol": volume})
def render_from_volume(tag: str = "isaac", rt_subframes: int = 4) -> dict:
    """Render the pose file at /vol/spike/{tag}_poses.npy with the shared scene layout."""
    import subprocess
    import sys

    sys.path.insert(0, "/root")
    job_path = f"/vol/spike/{tag}_job.json"
    with open(job_path) as fh:
        job = json.load(fh)
    job["rt_subframes"] = rt_subframes
    job["poses"] = f"/vol/spike/{tag}_poses.npy"
    job["out_dir"] = f"/vol/spike/frames/{tag}"
    script = "/tmp/isaac_render.py"
    pathlib.Path(script).write_text(RENDER)
    p = subprocess.run([sys.executable, script, job_path], capture_output=True, text=True,
                       timeout=7000)
    log = (p.stdout or "") + "\n--stderr--\n" + (p.stderr or "")
    pathlib.Path(f"/vol/spike/{tag}_render.log").write_text(log[-200000:])
    volume.commit()
    out = {"returncode": p.returncode, "log_tail": log[-4000:]}
    for line in log.splitlines():
        if line.startswith("RENDER_RESULT "):
            out["render"] = json.loads(line[len("RENDER_RESULT "):])
    return out
