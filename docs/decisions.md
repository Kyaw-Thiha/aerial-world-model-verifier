# Decisions log

One line per structural choice, with the reason. Newest at the bottom.

## Setup

- Flat `src/` with `__init__.py` files, no `src/verifier/`: scripts run from the repo root and import `from src.physics import ...` (per project_structure.md).
- Folders are created in the phase that first needs them (Phase 0 creates `src/datasets`, `src/physics`, `experiments`, `scripts`, `tests`, `data`, `results`, `docs/handoffs`); `remote/`, `third_party/`, `src/utils/` etc. are not created until used.
- pixi (`pixi.toml` + committed `pixi.lock`) replaces `requirements.txt`: the handoff says not to create it, although project_structure.md still lists it.
- pixi platform is `linux-64` only (workstation is linux x64); `osx-arm64` can be added if a Mac joins.
- Python 3.12 from conda-forge; Phase 0 default environment is CPU only, no torch/CUDA/perception. A separate `gpu` feature is added in Phase 1.
- `python-dateutil<3` is pinned in the conda deps: the conda solve picked `python-dateutil==4.0.0`, which made the PyPI resolver fall back to ancient boto3 0.0.3 / botocore 0.76 / six 1.7.3 (`import boto3` failed with `six.moves`). With the pin, boto3/botocore 1.43.126 install.
- `boto3>=1.34` and `python-dotenv>=1.0` are lower-bounded for the same reason; `modal` is unpinned (resolved 1.6.3). Modal containers get their own image definition under `remote/` and do not mirror `pixi.toml`; shared pins will be listed here when that exists.
- `.env.example` keeps the user's original entries (R2_*, R2_TOKEN_VALUE, RUNPOD_API_KEY, WANDB_API_KEY, R2_BUCKET default name) and adds the names required by project_structure.md that were missing (MODAL_TOKEN_ID, MODAL_TOKEN_SECRET, HF_TOKEN). Names only; no key values.
- The handoff file was not in the repo, so `docs/handoffs/01_setup_phase0_anwm.md` is a condensed record of the handoff plus the user's follow-up additions, not a verbatim copy.
- Compute: Phase 0 runs locally on CPU. If a download is blocked locally, the fallback is a Modal CPU function writing to a Modal Volume (never to local disk directly), pulled with `modal volume get`; Volume name and path get logged here if used.
