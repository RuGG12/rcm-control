# rcm-control

![RCM Animation](rcm_animation.gif)

RCM (Remote Center of Motion) controller for a UR5 robot arm, written in Python.
The controller keeps the surgical tool shaft passing through a fixed point (the
trocar) while the tip moves. If that constraint breaks, the tool tears tissue at
the incision.

---

## Results

| Metric | Value |
|--------|-------|
| RCM error — mean | 0.0777 mm |
| RCM error — max | 0.5475 mm |
| Tolerance from literature | 29.8 mm |
| Margin | 384x |
| Naive controller (no null space) | 195.4 mm mean drift |

![Comparison Plot](rcm_comparison.png)

---

## Why this approach

The naive approach — plain pseudoinverse tracking the tip — drifts 195.4 mm at
the trocar on average. The fix is to treat the trocar constraint as the primary
task and use the remaining joint freedom (null space) for tip tracking.

Three things that matter:

- **Null-space projection** — tip tracking runs in the null space of the RCM
  Jacobian, so it doesn't fight the trocar constraint
- **Damped least squares** — near singular joint configurations the plain
  pseudoinverse blows up; adding λ²I keeps velocities bounded
- **Periodic drift correction** — Euler integration drifts over time; every
  50 steps the trocar error is measured and corrected directly

---

## Setup

```bash
pip install -r requirements.txt
python demo.py
```

---

## Lambda sweep

Tested λ across [0.001, 0.01, 0.05, 0.1, 0.5]. λ=0.01 works best — small
enough to not lose accuracy, large enough to stay stable near singularities.

![Lambda Sweep](lambda_sweep.png)

---

## Note on tip tracking error

Tip error is large (~200 mm) in simulation. This is because the robot starts
at the zero configuration, far from the trajectory workspace. The null space
after enforcing the RCM constraint doesn't have enough freedom to close that
gap. The trocar constraint stays intact — the controller is doing the right
thing, the initial configuration is just poor. In practice the robot would
start near the target.

---

## Math

DH parameters, Jacobian derivation, null-space controller, damped least squares,
drift correction — in [docs/MATH.md](docs/MATH.md).

---

## Background

I worked on hydraulic excavator control on the MOOG-funded EARTH project where
constraint violations have real physical consequences. RCM control is the same
idea — the constraint comes first, everything else is secondary.
