### T1 — per-path seed spread

| path | maneuver | designed_deg | seeds | depicted_median_deg | depicted_seed_sd_deg | depicted_seed_range_deg | residual_median_deg | residual_seed_range_deg | feasible_seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C_accel_brake_10_course_0 | accel_brake | 10.00 | 5 | 0.65 | 0.15 | 0.33 | 8.08 | 0.15 | 0 |
| C_accel_brake_20_course_0 | accel_brake | 20.00 | 5 | 0.45 | 0.11 | 0.30 | 8.19 | 0.30 | 0 |
| C_accel_brake_20_course_1 | accel_brake | 20.00 | 5 | 2.64 | 0.62 | 1.53 | 6.21 | 1.52 | 4 |
| C_accel_brake_5_course_0 | accel_brake | 5.00 | 5 | 0.54 | 0.12 | 0.29 | 7.34 | 1.24 | 0 |
| C_climb_turn_10_fixed_0 | climb_turn | 10.00 | 5 | 1.39 | 0.18 | 0.48 | 11.03 | 0.75 | 0 |
| C_climb_turn_20_fixed_0 | climb_turn | 20.00 | 5 | 2.00 | 0.96 | 2.48 | 18.25 | 2.46 | 0 |
| C_climb_turn_20_fixed_1 | climb_turn | 20.00 | 5 | 2.45 | 0.54 | 1.33 | 18.04 | 1.02 | 0 |
| C_climb_turn_5_course_0 | climb_turn | 5.00 | 5 | 0.86 | 0.17 | 0.44 | 8.54 | 0.26 | 0 |
| C_climb_turn_5_fixed_0 | climb_turn | 5.00 | 5 | 0.70 | 0.13 | 0.34 | 8.67 | 0.47 | 0 |
| C_cruise_ctl_course_0 | cruise | — | 5 | 0.57 | 0.07 | 0.16 | 8.00 | 0.20 | 0 |
| C_cruise_ctl_course_1 | cruise | — | 5 | 2.40 | 0.26 | 0.64 | 6.25 | 0.75 | 4 |
| C_slalom_10_fixed_0 | slalom | 10.00 | 5 | 0.55 | 0.14 | 0.30 | 9.43 | 0.72 | 0 |
| C_slalom_10_fixed_1 | slalom | 10.00 | 5 | 2.26 | 0.28 | 0.65 | 7.14 | 1.55 | 1 |
| C_slalom_5_course_0 | slalom | 5.00 | 5 | 0.52 | 0.13 | 0.30 | 8.41 | 0.20 | 0 |
| C_slalom_5_fixed_0 | slalom | 5.00 | 5 | 0.53 | 0.11 | 0.28 | 8.62 | 0.54 | 0 |
| C_turn_10_fixed_0 | turn | 10.00 | 5 | 0.74 | 0.16 | 0.41 | 12.22 | 0.38 | 0 |
| C_turn_20_fixed_0 | turn | 20.00 | 5 | 0.79 | 0.25 | 0.67 | 20.47 | 0.69 | 0 |
| C_turn_20_fixed_1 | turn | 20.00 | 5 | 2.77 | 0.83 | 1.98 | 18.46 | 1.98 | 0 |
| C_turn_5_course_0 | turn | 5.00 | 5 | 0.89 | 0.17 | 0.42 | 8.98 | 0.52 | 0 |
| C_turn_5_fixed_0 | turn | 5.00 | 5 | 0.65 | 0.21 | 0.54 | 9.21 | 0.55 | 0 |

### T2 — oracle best-of-N

| N | n_paths | mean_oracle_residual_deg | median_oracle_residual_deg | feasible_k | feasible_n | feasible_rate | mean_oracle_residual_k0_deg | feasible_k0_k | feasible_k0_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 20 | 10.46 | 8.66 | 2 | 20 | 0.10 | 6.72 | 14 | 0.70 |
| 2 | 20 | 10.39 | 8.66 | 2 | 20 | 0.10 | 6.55 | 14 | 0.70 |
| 3 | 20 | 10.29 | 8.64 | 2 | 20 | 0.10 | 6.41 | 14 | 0.70 |
| 4 | 20 | 10.26 | 8.39 | 2 | 20 | 0.10 | 6.36 | 14 | 0.70 |
| 5 | 20 | 10.16 | 8.39 | 3 | 20 | 0.15 | 6.31 | 14 | 0.70 |

```json
{
 "depicted_spread": {
  "median_seed_sd_deg": 0.16884150080651367,
  "median_seed_range_deg": 0.4299963597511116,
  "max_seed_range_deg": 2.4777975733405775
 },
 "residual_spread": {
  "median_seed_sd_deg": 0.2411924378209521,
  "median_seed_range_deg": 0.6187680451213584,
  "max_seed_range_deg": 2.4578805359995464
 },
 "oracle_reduction_1_to_5": {
  "per_path": [
   0.0,
   0.23267211210848515,
   0.0,
   0.0,
   0.7488527567663912,
   1.2990660632208346,
   0.2500220311966288,
   0.2033095468646824,
   0.3307611734219176,
   0.19895531606040961,
   0.5038356332677525,
   0.48069643986843147,
   0.5001199173362592,
   0.18852027679514727,
   0.27592716349817437,
   0.2608525509546826,
   0.4390457512274075,
   0.10068184183069917,
   0.09309920298016827,
   0.0
  ],
  "median_deg": {
   "point": 0.24134707165255698,
   "lo": 0.14460105931292322,
   "hi": 0.38490346232466255,
   "n_clusters": 20,
   "descriptive": false
  },
  "mean_deg": {
   "point": 0.3053208888699036,
   "lo": 0.18507840094492,
   "hi": 0.4524143870902111,
   "n_clusters": 20,
   "descriptive": false
  },
  "n_paths_flipped_infeasible_to_feasible": 1,
  "n_paths": 20
 },
 "design_contrast": {
  "deg_per_deg": 0.07827217219575174,
  "intercept_deg": 0.2967394116390777,
  "n_paths": 18,
  "aggregation": "mean depicted per path",
  "by_design": {
   "5.0": {
    "mean_depicted_deg": 0.6700944554879227,
    "n_paths": 7
   },
   "10.0": {
    "mean_depicted_deg": 1.1172733495694132,
    "n_paths": 5
   },
   "20.0": {
    "mean_depicted_deg": 1.8516794622283292,
    "n_paths": 6
   }
  },
  "per_clip_seed0": {
   "deg_per_deg": 0.10295342462381535,
   "intercept_deg": 0.1549131745193988,
   "n_clips": 18
  }
 }
}
```

### T3 — lingbot depicted tilt by arm

| path | arm | n | depicted_median_deg | injected_median_deg | lean_cos |
| --- | --- | --- | --- | --- | --- |
| cruise | probe_level | 1 | 3.21 | 0.00 | 0.00 |
| cruise | probe_roll_-20 | 1 | 3.77 | 10.00 | -0.28 |
| cruise | probe_roll_20 | 1 | 3.30 | 10.00 | 0.33 |

### lingbot summary

```json
{
 "by_arm": {
  "probe_level": {
   "median_deg": 3.2103840544121476,
   "n": 1
  },
  "probe_roll_-20": {
   "median_deg": 3.7716802585492837,
   "n": 1
  },
  "probe_roll_20": {
   "median_deg": 3.297035943987149,
   "n": 1
  }
 },
 "flatness_minus_level_deg": null,
 "mirror_flip": null
}
```

### T3 — lingbot_b depicted tilt by arm

| path | arm | n | depicted_median_deg | injected_median_deg | lean_cos |
| --- | --- | --- | --- | --- | --- |
| cruise | probe_roll_60 | 1 | 4.05 | 30.00 | 0.35 |
| cruise | probe_roll_fast20 | 1 | 3.93 | 10.00 | 0.29 |

### lingbot_b summary

```json
{
 "by_arm": {
  "probe_roll_60": {
   "median_deg": 4.045502383565258,
   "n": 1
  },
  "probe_roll_fast20": {
   "median_deg": 3.9284696278825155,
   "n": 1
  }
 },
 "flatness_minus_level_deg": null,
 "mirror_flip": null
}
```

### wan summary

```json
{
 "by_arm": null,
 "flatness_minus_level_deg": null,
 "mirror_flip": null
}
```