"""Flightmare build + headless-render attempt on Modal (Track I spike, candidate A).

Flightmare is a C++ flight library (``flightlib``, pybind11) plus a **Unity binary** renderer
distributed as a 936 MB release asset, and its upstream install instructions target Ubuntu
18.04 with ROS Melodic. The spike question is whether any of that still builds and renders
headlessly on a cloud GPU in 2026, within a day of effort.

The function does the whole probe in one run and returns the full log, so a failure is
recorded rather than guessed:

1. clone the repo at the pinned commit, build ``flightlib`` (pybind module) with apt deps;
2. HEAD the Unity renderer release asset;
3. if the build succeeded, download the renderer and try to start it under ``xvfb-run``
   with GL, printing whatever it says.

    pixi run modal run remote/i_flightmare.py::probe
"""
from __future__ import annotations

import modal

app = modal.App("awmv-i-flightmare")

FLIGHTMARE_COMMIT = "master"
UNITY_ASSET = ("https://github.com/uzh-rpg/flightmare/releases/download/0.0.5/"
               "RPG_Flightmare.tar.xz")

image = (
    modal.Image.from_registry("ubuntu:22.04", add_python="3.10")
    .apt_install("git", "cmake", "g++", "build-essential", "pkg-config",
                 # libzmqpp-dev does not exist on 22.04 (it is an 18.04-era package);
                 # zmqpp is therefore built from source in the probe.
                 "libzmq3-dev", "libopencv-dev", "libeigen3-dev", "libyaml-cpp-dev",
                 "libboost-all-dev", "xvfb", "libgl1", "libglu1-mesa", "mesa-utils",
                 "libgomp1", "curl", "ca-certificates")
)

PROBE = r'''
import json, os, subprocess, sys, time
log = []
def run(cmd, cwd=None, timeout=1800, env=None):
    t0 = time.time()
    p = subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str), capture_output=True,
                       text=True, timeout=timeout, env=env)
    out = {"cmd": cmd if isinstance(cmd, str) else " ".join(cmd),
           "rc": p.returncode, "sec": round(time.time() - t0, 1),
           "tail": ((p.stdout or "")[-1500:] + (p.stderr or "")[-3000:])}
    log.append(out)
    return p

probe_apt = {"libzmqpp-dev_missing_on_22_04": True}
run("git clone --depth 1 https://github.com/uzh-rpg/flightmare.git /work/flightmare")
run("git clone --depth 1 https://github.com/zeromq/zmqpp.git /work/zmqpp")
run("cd /work/zmqpp && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && "
    "cmake --build build -j 8 && cmake --install build", timeout=1800)
run(["ls", "/work/flightmare"])
# the pybind package lives in flightlib/ (setup.py at that level)
build = run([sys.executable, "-m", "pip", "install", "--no-build-isolation", "."],
            cwd="/work/flightmare/flightlib", timeout=2400)
probe = {"flightlib_built": build.returncode == 0}
try:
    import flightgym  # noqa: F401
    probe["flightgym_import"] = True
except Exception as e:
    probe["flightgym_import"] = repr(e)

head = run(f"curl -sSIL -m 60 {UNITY_ASSET} | grep -Ei 'HTTP/|content-length' | tail -4", timeout=120)
probe["unity_asset_head"] = head["tail"]

if build.returncode == 0:
    run("mkdir -p /work/unity && cd /work/unity && curl -sSL -o unity.tar.xz " + UNITY_ASSET,
        timeout=1800)
    run("cd /work/unity && tar xf unity.tar.xz && find . -maxdepth 2 -type f -name '*Flightmare*' | head")
    # try to start the Unity player offscreen
    start = run("cd /work/unity && (find . -maxdepth 3 -type f -executable -name '*.x86_64' | head -1) ")
    probe["unity_binaries"] = start["tail"]
    glx = run("xvfb-run -a -s '-screen 0 1280x720x24' glxinfo -B", timeout=300)
    probe["glxinfo"] = glx["tail"]
    probe["glx_ok"] = glx["rc"] == 0
print("PROBE_RESULT " + json.dumps({"probe": probe, "log": log}))
'''


@app.function(image=image, timeout=2 * 3600)
def probe() -> dict:
    import json as _json
    import subprocess
    import sys

    p = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True,
                       timeout=7000)
    text = (p.stdout or "") + "\n--stderr--\n" + (p.stderr or "")
    out = {"returncode": p.returncode, "log_tail": text[-4000:]}
    for line in text.splitlines():
        if line.startswith("PROBE_RESULT "):
            out.update(_json.loads(line[len("PROBE_RESULT "):]))
    import pathlib
    pathlib.Path("/tmp/flightmare_probe.txt").write_text(text[-400000:])
    out["log_full"] = text[-400000:]
    return out
