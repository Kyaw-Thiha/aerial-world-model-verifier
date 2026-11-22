"""H03 (optional): build the 60-frame human-labelling tool.

Copies 60 sampled frames into ``data/h03_label/frames`` and writes a self-contained
``data/h03_label/label.html``: drag along the horizon (or a long vertical edge) on each frame,
the angle is stored in the browser and exported as JSON.  The labels are never decisive; if
the user labels them they are reported as exploratory.

    PYTHONPATH=. pixi run -e gpu python experiments/h03_label_tool.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/h03_frames"
OUT = ROOT / "data/h03_label"
PER_SOURCE = {"anwm": 15, "lingbot_bf16": 15, "solarwm": 15, "sim_control": 8,
              "real_control": 7}

HTML = """<!doctype html>
<meta charset="utf-8">
<title>H03 horizon labelling</title>
<style>
 body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:1rem}
 #wrap{position:relative;display:inline-block;max-width:100%}
 canvas{position:absolute;left:0;top:0;cursor:crosshair}
 img{max-width:100%;display:block}
 #bar{display:flex;gap:1rem;align-items:center;margin:.6rem 0;flex-wrap:wrap}
 button{padding:.3rem .8rem;font-size:1rem}
</style>
<div id="bar">
  <b id="name">-</b>
  <span id="ang">no line</span>
  <button id="clear">clear</button>
  <button id="prev">prev</button>
  <button id="next">next</button>
  <button id="export">export JSON</button>
  <span id="count"></span>
</div>
<div id="wrap"><img id="im"><canvas id="cv"></canvas></div>
<p>Drag along the horizon (or a long edge you believe is level). The angle is measured with
image y pointing down, the same convention as the report's roll.</p>
<script>
const FRAMES = __FRAMES__;
let i = 0, start = null, line = null;
const store = JSON.parse(localStorage.getItem("h03labels") || "{}");
const im = document.getElementById("im"), cv = document.getElementById("cv");
function fit(){
  const w = im.clientWidth, h = im.clientHeight;
  cv.width = w; cv.height = h; cv.style.width = w+"px"; cv.style.height = h+"px";
  draw();
}
function draw(){
  const ctx = cv.getContext("2d");
  ctx.clearRect(0,0,cv.width,cv.height);
  const l = line;
  if(l){
    ctx.strokeStyle = "#00ff66"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(l[0][0],l[0][1]); ctx.lineTo(l[1][0],l[1][1]); ctx.stroke();
  }
}
function angleOf(l){
  const dx = l[1][0]-l[0][0], dy = l[1][1]-l[0][1];
  let a = Math.atan2(dy, dx) * 180/Math.PI;
  if(a > 90) a -= 180; if(a < -90) a += 180;
  return a;
}
function show(){
  const f = FRAMES[i];
  im.src = "frames/" + f;
  im.onload = () => { fit(); line = store[f] ? store[f].line : null; draw(); update(); };
}
function update(){
  document.getElementById("name").textContent = FRAMES[i];
  document.getElementById("ang").textContent = line ? angleOf(line).toFixed(2)+" deg" : "no line";
  document.getElementById("count").textContent = `${i+1} / ${FRAMES.length}, labelled ${Object.keys(store).length}`;
  if(line) store[FRAMES[i]] = {line, deg: angleOf(line)};
  localStorage.setItem("h03labels", JSON.stringify(store));
}
cv.addEventListener("mousedown", e => {
  const r = cv.getBoundingClientRect();
  start = [e.clientX-r.left, e.clientY-r.top]; line = [start, start]; draw();
});
cv.addEventListener("mousemove", e => {
  if(!start) return;
  const r = cv.getBoundingClientRect();
  line = [start, [e.clientX-r.left, e.clientY-r.top]]; draw();
});
cv.addEventListener("mouseup", () => { if(line) update(); start = null; });
document.getElementById("clear").onclick = () => { delete store[FRAMES[i]]; line = null; update(); draw(); };
document.getElementById("prev").onclick = () => { i = Math.max(0, i-1); show(); };
document.getElementById("next").onclick = () => { i = Math.min(FRAMES.length-1, i+1); show(); };
document.getElementById("export").onclick = () => {
  const blob = new Blob([JSON.stringify(store, null, 1)], {type:"application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "h03_labels.json"; a.click();
};
show();
</script>
"""


def main() -> int:
    rows_dir = OUT / "frames"
    rows_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for src, n in PER_SOURCE.items():
        files = sorted((SRC / src).glob("*/*.png"))
        if not files:
            continue
        idx = np.linspace(0, len(files) - 1, n).round().astype(int)
        for k in idx:
            p = files[int(k)]
            name = f"{src}__{p.parent.name}__{p.stem}.png"
            shutil.copyfile(p, rows_dir / name)
            frames.append(name)
    (OUT / "label.html").write_text(HTML.replace("__FRAMES__", json.dumps(frames, indent=1)))
    (OUT / "frames.json").write_text(json.dumps(frames, indent=1))
    print(json.dumps({"frames": len(frames), "html": str(OUT / "label.html")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
