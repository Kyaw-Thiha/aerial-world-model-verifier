"""Depicted-tilt handling for a second dataset (Track E).

Modules here are UZH-FPV specific: the published mount, the measured frame-to-truth clock
offset, the per-frame gravity metric in the camera frame, and the clip exporter. Nothing in
this package edits `src/perception`, `src/physics`, `src/datasets/tii.py` or Track C's code;
it imports from them read-only where that is the honest reuse.
"""
