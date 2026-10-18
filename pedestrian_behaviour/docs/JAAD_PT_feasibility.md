# JAAD Annotation Feasibility for Prospect Theory

## Prospect Theory Components

Prospect Theory (Kahneman & Tversky, 1979 / Tversky & Kahneman, 1992) models
decision-making under risk via:

1. **Reference point** — outcomes evaluated as gains or losses relative to a neutral baseline
2. **Value function** — concave for gains (risk-averse), convex for losses (risk-seeking); loss aversion λ ≈ 2.25
3. **Probability weighting** — overweighting of small probabilities, underweighting of large ones
4. **CPT parameters**: α, β (curvature), λ (loss aversion), γ (probability distortion)

---

## JAAD Annotation Sources

| Source | Content |
|---|---|
| `annotations/` | Per-frame bounding boxes, occlusion, action, look, cross, reaction, hand gesture |
| `annotations_attributes/` | Per-pedestrian demographics, crossing point, decision point, designated, signalized, intersection, num_lanes |
| `annotations_traffic/` | Per-frame traffic light, ped crossing sign, stop sign |
| `annotations_vehicle/` | Per-frame vehicle action (stopped, decelerating, moving_slow, accelerating) |
| `annotations_appearance/` | Per-frame categorical pose direction and clothing flags — **no keypoint coordinates** |

---

## PT Feature Feasibility

### From JAAD Annotations Directly

| Feature | Field(s) Used | PT Role | Notes |
|---|---|---|---|
| **Decision point** | `attributes.decision_point` | Moment of choice under uncertainty | Frame index when pedestrian commits to crossing or waiting |
| **Crossing outcome** | `attributes.crossing` (+1/-1) | Binary choice variable | Label for the decision |
| **Scene type** (binary) | `attributes.designated` (D=1, ND=0) | Reference point | Designated crosswalk = safe expected state; mid-block = loss-framed departure from reference |
| **Wait time** | `behavior.action` sequence (standing=0) before decision point | Delay cost (foregone gain) | Continuous standing frames at end of observation window ÷ 30 fps; pre-clip wait unobservable |
| **Traffic context** | `traffic_annotations` — `traffic_light`, `ped_crossing`, `ped_sign` | Reference point modifiers | Red light / crosswalk presence anchors the pedestrian's safe-crossing expectation |
| **Vehicle state** | `vehicle_annotations` — action per frame | Objective risk at decision point | Stopped/decelerating = lower collision probability; accelerating = higher |
| **Looking behaviour** | `behavior.look` (looking/not-looking) | Probability weighting | Active gaze toward vehicle = pedestrian is updating collision probability estimate |
| **Reaction** | `behavior.reaction` (clear_path, speed_up, slow_down) | Loss/gain framing response | clear_path = gain confirmed; speed_up/slow_down = reactive to loss threat |
| **Group size** | `attributes.group_size` | Social influence on risk perception | Larger groups may reduce perceived individual risk (diffusion of responsibility) |
| **Demographics** | `attributes.age`, `attributes.gender` | Individual differences in λ, α, β | Older pedestrians tend toward higher loss aversion empirically |
| **Signalized** | `attributes.signalized` (n/a / NS / S) | Reference point | Signalized = stronger safe-crossing expectation |
| **Intersection** | `attributes.intersection` (yes/no) | Reference point context | Intersection crossings have more structured traffic patterns |

### Derived from Bounding Boxes

| Feature | Derivation | PT Role |
|---|---|---|
| **BB expansion rate** | Mean frame-to-frame relative growth of bbox area (w×h) | TTC proxy — expanding bbox = vehicle approaching; feeds probability weighting |

---

## PT Feature Vector (5-dim input to AdaptationNet)

These are the five features passed to `ProspectTheoryModule.adaptation_net`:

| Dim | Feature | Source | Available Without ViTPose? |
|---|---|---|---|
| 0 | `wait_time` | `action` sequence → `labels.csv` | Yes |
| 1 | `risk_proxy` (bb_expansion_rate / TTC) | bbox array → `labels.csv` (JAAD) or OBD speed (PIE) | Yes |
| 2 | `scene_type` | `designated` attribute → `labels.csv` | Yes |
| 3 | `head_orient_var` | Variance of nose–neck x-displacement (joints 0, 5, 6) | **No — requires ViTPose** |
| 4 | `mean_joint_speed` | Mean frame-to-frame displacement across all 17 joints | **No — requires ViTPose** |

Dims 0–2 are pre-computed in `data/preprocess_jaad.py` and stored in `labels.csv`.
Dims 3–4 are computed at load time in `jaad_dataset.py` from `skeletons_vitpose/<tid>.npy`.
Before ViTPose extraction, zero-padded skeletons produce 0.0 for both — training will
down-weight those dimensions until real skeletons are available.

---

## What Is Not Possible from JAAD Annotations

| Feature | Why Not Available |
|---|---|
| **Head orientation variance** (keypoint-based) | `annotations_appearance/` stores only categorical pose direction flags (`pose_front`, `pose_back`, `pose_left`, `pose_right` as 0/1) — no (x, y) coordinates |
| **Mean joint speed** | Same — no keypoint coordinates anywhere in JAAD annotations |
| **Pre-clip wait time** | Annotations begin at clip start; waiting before the clip is unobservable |
| **Ego-vehicle speed** | JAAD has no on-board diagnostics (OBD). BB expansion rate is the best available TTC proxy. PIE provides ground-truth OBD speed. |
| **Subjective collision probability** | Must be back-inferred from observed behaviour; not directly annotated |

---

## Mapping to CPT Parameters

| CPT Parameter | Informed By |
|---|---|
| **α** (gain curvature) | `wait_time`, `scene_type` — how much delay cost shapes willingness to cross |
| **β** (loss curvature) | `risk_proxy`, `head_orient_var` — how vehicle proximity shapes collision loss weighting |
| **λ** (loss aversion) | `scene_type`, demographics — crossing mid-block despite risk reveals λ |
| **γ** (probability distortion) | `risk_proxy`, `mean_joint_speed` — hesitation dynamics reveal probability overweighting |
