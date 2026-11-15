# Track B artifacts

Local results (committed unless noted), with the remote copy where one exists.

| artifact | path | where |
| --- | --- | --- |
| corrected excitation audit | `results/anwm_excitation/audit_corrected.json` | local |
| reorganised audit (pooled cells) | `results/anwm_excitation/audit.json` | local |
| k_d split fit (piloted / autonomous) | `results/anwm_excitation/kd_split.json` | local |
| path-only audit, three regimes, dt sweep | `results/anwm_path_audit/audit.json` | local + R2 `track-b/clips/audit.json` |
| native start states (depth/pose + recorded path) | `results/anwm_starts/starts.json` | local + R2 |
| start prior coverage (FFP non-empty test) | `results/anwm_starts/starts_coverage.json` | local + R2 |
| clip specs (82 requested, commanded paths; sha256 d656de52cca15f20…, builder is deterministic) | `results/anwm_clips/specs.json` | local + R2 |
| Gate 2 native rollouts, LPIPS, contact sheets | `results/anwm_native/` | local (24 MB) |
| slim EMA checkpoint | Modal `awmv-b-weights:/slim/0200000_ema.pth.tar` | R2 `track-b/weights/` |
| checkpoint manifest (both sha256) | `awmv-b-weights:/slim/manifest.json` | R2 `track-b/weights/manifest.json` |
| generated clips + index + variance | `data/processed/anwm_clips/` (gitignored) | R2 `track-b/clips/` |

Hashes: source `0200000.pth.tar` 18,146,420,481 bytes
`bdd149cac6ec002ba7dc4ad99ec6f9eb02cd6d4f05320195cf174737b13b0bc2`; slim EMA checkpoint
4,536,552,948 bytes `59def562a12275640e4d04d1af3b38281caa4f656eb104e96678fc49c6ae3f8b`.

Reproduce: `python -m experiments.anwm_path_audit` (CPU, 15 s); gate 2 and generation need Modal
(`remote/b_anwm.py`, volumes `awmv-b-*`).
