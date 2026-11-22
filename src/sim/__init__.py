"""Simulated-flight track (Track I): analytic flights, renderer adapters, labels.

The sim supplies what the real datasets cannot: exact position and attitude at a high
rate, and infeasible flights made infeasible **by construction**, with the violation's
type, magnitude and exact time interval known.

Layout:

* ``src/sim/flight.py``  -- analytic maneuver library, flatness attitude, camera mounts.
  No renderer, no GPU, importable in the CPU env and unit-tested.
* ``src/sim/labels.py``  -- infeasible variants (tilt removed, mirrored roll, tilt scaled,
  thrust-limit) with their intervals. (Step 2.)
* ``src/sim/blender.py`` -- scene + renderer for the Blender candidate. Runs inside Blender.
* ``src/sim/render_*.py``-- one adapter per renderer, all consuming the same ``Flight``.
"""
