"""Methodology tables and reproducibility table.

Assets:
  tab_datasets_1.tex        datasets used (sensor, rate, truth, ranges, use)
  tab_repro_1.tex           per branch: HEAD, frozen configs with sha256, R2 prefix, tests, cost
  tab_frozen-configs_1.tex  all frozen configurations and their pre-registration commits

Sources: live git worktrees (branch HEADs, config blobs) plus stored audit JSONs and
the report statements of test counts and cost (which are not result files).
"""
from __future__ import annotations

import subprocess

import common as C

BRANCHES = {
    "a": "track-a-perception", "a2": "track-a2-gate", "b": "track-b-generation",
    "c": "track-c-pathrec", "d": "track-d-verdict-stat", "e": "track-e-depicted-uzh",
}
CONFIGS = {
    "a": ["configs/phase1.yaml"],
    "a2": ["configs/verdict_preregistration.yaml", "configs/gate_frozen.yaml"],
    "b": [],
    "c": ["configs/pathrec_prereg.yaml"],
    "d": ["configs/gain_frozen.yaml", "configs/coherence_frozen.yaml",
          "configs/verdict_preregistration.yaml", "configs/window_bandwidth_prereg.yaml"],
    "e": [],
}
ASSET = "methodology"


def git(track, *args):
    return subprocess.check_output(["git", "-C", str(C.WORKTREES[track]), *args], text=True).strip()


def config_provenance(track, rel):
    p = C.WORKTREES[track] / rel
    if not p.exists():
        return None
    commit = git(track, "log", "-1", "--format=%h", "--", rel)
    return commit, C.sha256(p)


def build():
    # ---- datasets table
    lines = [
        r"\begin{tabular}{lllll}",
        r"\toprule",
        r"Dataset & sensor & rate & truth source & used for \\",
        r"\midrule",
    ]
    datasets = [
        ("TII racing", "fisheye + mocap", "mocap 274.73 Hz; camera 12.04 Hz",
         "mocap pose", "E1-E3, E8, E10"),
        ("UZH-FPV", "forward / 45$^\\circ$-down cameras", "truth 500 Hz; clips ~15 Hz",
         "sensor ground truth", "E7, E9, E10"),
        ("ANWM", "generated frames (224/512 px)", "1 s per step, 16 steps",
         "commanded path", "E4-E7"),
    ]
    for name, sensor, rate, truth, use in datasets:
        lines.append(f"{name} & {sensor} & {rate} & {truth} & {use} \\\\")
    lines += [r"\bottomrule",
              r"\multicolumn{5}{l}{\footnotesize Rates and truth sources from the source audits "
              r"(phase0 floor, UZH audit, ANWM clip index).} \\",
              r"\end{tabular}"]
    C.write_tex("tab_datasets_1", "\n".join(lines))

    # ---- reproducibility table
    lines = [
        r"\begin{tabular}{llllr}",
        r"\toprule",
        r"Branch & HEAD & pre-registration / frozen configs & R2 prefix & tests \\",
        r"\midrule",
    ]
    csv_rows = []
    for track, branch in BRANCHES.items():
        head = git(track, "rev-parse", "--short", "HEAD")
        headdate = git(track, "log", "-1", "--format=%ad", "--date=short")
        cfg = []
        for rel in CONFIGS[track]:
            pr = config_provenance(track, rel)
            if pr:
                cfg.append((rel.split("/")[-1], pr[0], pr[1]))
        if cfg:
            cfg_s = f"{len(cfg)} frozen config(s), see tab\\_frozen-configs\\_1"
        else:
            cfg_s = "--"
        lines.append(f"{branch} & {head} ({headdate}) & {cfg_s} & {C.R2_PREFIXES[track]} & -- \\\\")
        csv_rows.append([branch, head, headdate, cfg_s, C.R2_PREFIXES[track]])
    lines += [r"\bottomrule",
              r"\multicolumn{5}{l}{\footnotesize Config entries are file@commit (sha256 prefix). "
              r"Object counts in the R2 bucket could not be listed: the local environment has "
              r"no valid R2 credentials (see cutoff.json). Test counts and cost are reported in "
              r"the branch reports and are not re-run here.} \\",
              r"\end{tabular}"]
    C.write_tex("tab_repro_1", "\n".join(lines))

    # ---- frozen configs table (references tab_repro_1)
    lines = [
        r"\begin{tabular}{llll}",
        r"\toprule",
        r"Branch & config & commit & sha256 \\",
        r"\midrule",
    ]
    for track, branch in BRANCHES.items():
        for rel in CONFIGS[track]:
            pr = config_provenance(track, rel)
            if not pr:
                continue
            lines.append(f"{branch} & {C.tex_escape(rel)} & {pr[0]} & {pr[1][:16]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    C.write_tex("tab_frozen-configs_1", "\n".join(lines))

    C.write_csv("repro", csv_rows, ["branch", "head", "head_date", "configs", "r2_prefix"])
    C.provenance("repro", [("d", "pixi.toml")] if False else [],
                 notes=("HEADs and config hashes read live from the six worktrees. "
                        "R2 object counts unavailable: rclone returns 403 and no .env is present."),
                 derived=True)
    return dict(rows=csv_rows)


if __name__ == "__main__":
    for r in build()["rows"]:
        print(r[0], r[1])
