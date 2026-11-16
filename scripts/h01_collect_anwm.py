"""Block until the two ANWM seed-generation calls finish, then print their summaries.

Collecting a spawned call is not polling: this waits on the call handle and returns when the
container has finished. It is run as a background job so the pilot continues meanwhile.
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    for name in ("spawn_b.json", "spawn_c.json"):
        doc = json.loads((ROOT / "results/h01_seed_specs" / name).read_text())
        print("waiting", name, doc["call_id"], flush=True)
        res = modal.FunctionCall.from_id(doc["call_id"]).get(timeout=4 * 3600)
        print(name, json.dumps({k: v for k, v in res.items() if k != "clips"}, indent=1),
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
