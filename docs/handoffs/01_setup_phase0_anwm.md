# Handoff 01 (condensed record): setup, Phase 0 physics floor, ANWM feasibility

Condensed from the handoff message; not a verbatim copy. Scope: Task 1 scaffold, Task 2 Phase 0 (mocap-only flatness floor on TII), Task 3 ANWM feasibility, optional Task 4 compute smoke tests (only after the Phase 0 gate is reported, and not if the gate fails).

## User additions (approved plan, with these changes)

1. Fetch 6 to 8 flights across speeds and both uptilts; leave-one-flight-out CV for scalar `k_d`; report per-flight spread, not only the median; split by flight.
2. Report max and p95 speed per flight; include at least one flight above 10 m/s.
3. Diagnostic only (not the gate): body-frame drag with separate x and y coefficients (Faessler et al. 2018); report whether it lowers the high-speed residual.
4. Save per-frame arrays (implied axis, true axis, residual, speed, |a|) as npz or parquet in the run directory.
5. Modal fetch fallback lands on a Volume; pull with `modal volume get`; log Volume name and path in decisions.md.
6. ANWM excitation audit (if benchmark trajectories or action sequences exist): commanded actions to trajectory, implied-tilt distribution with the Phase 0 code, fraction of steps above 10 degrees, state dt per action.
7. If the Phase 0 gate fails, still do Task 3 but not Task 4. Commit after each step. Never commit data/, .env or .pixi/.
