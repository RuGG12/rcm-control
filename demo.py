import numpy as np
from robot.kinematics import forward_kinematics

# Test configuration - all joints at zero
q_zeros = np.zeros(6)

T, transforms = forward_kinematics(q_zeros)

print("=== FK Verification - All Zeros ===")
print(f"End effector position: {T[:3, 3]}")
print(f"End effector rotation:\n{T[:3, :3]}")

q_test = np.array([0, -np.pi/2, 0, 0, 0, 0])
T2, _ = forward_kinematics(q_test)
print("\n=== FK Verification - Shoulder Down ===")
print(f"End effector position: {T2[:3, 3]}")

from robot.kinematics import compute_jacobian

q_test = np.zeros(6)
T, transforms = forward_kinematics(q_test)
J = compute_jacobian(q_test, transforms)

print("\n=== Jacobian at Zero Config ===")
print(f"Shape: {J.shape}")
print(f"J =\n{np.round(J, 4)}")

from robot.kinematics import get_tool_points

q_test = np.zeros(6)
T, transforms = forward_kinematics(q_test)
T_wrist = transforms[5]

p_trocar, p_tip = get_tool_points(T_wrist)

print("\n=== Tool Points at Zero Config ===")
print(f"Wrist position:   {np.round(T_wrist[:3, 3], 4)}")
print(f"Trocar point:     {np.round(p_trocar, 4)}")
print(f"Tool tip:         {np.round(p_tip, 4)}")

q_test = np.zeros(6)
T, transforms = forward_kinematics(q_test)
T_wrist = transforms[5]

p_trocar, p_tip = get_tool_points(T_wrist)

J_wrist  = compute_jacobian(q_test, transforms)
J_trocar = compute_jacobian(q_test, transforms, p_target=p_trocar)
J_tip    = compute_jacobian(q_test, transforms, p_target=p_tip)

print("\n=== Jacobians at Zero Config ===")
print(f"J_wrist  row 0: {np.round(J_wrist[0],  4)}")
print(f"J_trocar row 0: {np.round(J_trocar[0], 4)}")
print(f"J_tip    row 0: {np.round(J_tip[0],    4)}")

# ── Controller smoke test ─────────────────────────────────────────────────────
from robot.controller import RCMController

q0 = np.zeros(6)
T, transforms = forward_kinematics(q0)
T_wrist = transforms[5]
p_trocar, p_tip = get_tool_points(T_wrist)

# Fix the trocar where it currently is — it should not move
p_rcm_target = p_trocar.copy()

# Push the tip target 5 cm in X so the controller has something to track
p_tip_target = p_tip + np.array([0.05, 0.0, 0.0])

ctrl = RCMController(K1=5.0, K2=3.0, lam=0.01)
q_new, e_rcm, e_tip = ctrl.step(q0, p_rcm_target, p_tip_target, dt=0.01)

print("\n=== Controller Step Test ===")
print(f"q_new   : {np.round(q_new,  6)}")
print(f"e_rcm   : {np.round(e_rcm,  6)}  (should be ~0)")
print(f"|e_rcm| : {np.round(np.linalg.norm(e_rcm), 6)}")
print(f"e_tip   : {np.round(e_tip,  6)}")
print(f"|e_tip| : {np.round(np.linalg.norm(e_tip), 6)}")

# ── Simulation loop test ──────────────────────────────────────────────────────
from simulation.sim_loop import run_simulation

q0 = np.zeros(6)
T, transforms = forward_kinematics(q0)
p_trocar, _ = get_tool_points(transforms[5])

history = run_simulation(q0, p_rcm_target=p_trocar, dt=0.01, steps=500)

e_rcm_arr = np.array(history['e_rcm_norm'])
e_tip_arr  = np.array(history['e_tip_norm'])
n_corr     = sum(history['drift_corrections'])

print("\n=== Simulation (500 steps) ===")
print(f"RCM error  — mean: {e_rcm_arr.mean():.6f} m   max: {e_rcm_arr.max():.6f} m")
print(f"Tip error  — mean: {e_tip_arr.mean():.6f} m   max: {e_tip_arr.max():.6f} m")
print(f"Drift corrections applied: {n_corr}")

# ── RCMController vs NaiveController comparison ───────────────────────────────
from robot.controller import NaiveController

q0 = np.zeros(6)
T, transforms = forward_kinematics(q0)
p_trocar0, _ = get_tool_points(transforms[5])
p_rcm_target = p_trocar0.copy()

def run_controller(ctrl, q0, p_rcm_target, dt=0.01, steps=500):
    q = q0.copy()
    e_rcm_log = []
    e_tip_log  = []
    for step in range(steps):
        t = step * dt
        center = p_rcm_target + np.array([0.0, 0.1, 0.0])
        p_tip_target = center + np.array([0.05 * np.cos(t), 0.0, 0.05 * np.sin(t)])
        q, e_rcm, e_tip = ctrl.step(q, p_rcm_target, p_tip_target, dt)
        e_rcm_log.append(np.linalg.norm(e_rcm))
        e_tip_log.append(np.linalg.norm(e_tip))
    return np.array(e_rcm_log), np.array(e_tip_log)

rcm_errors, tip_errors_b = run_controller(
    RCMController(K1=5.0, K2=3.0, lam=0.01), q0, p_rcm_target)

naive_errors, tip_errors_a = run_controller(
    NaiveController(K=5.0), q0, p_rcm_target)

print("\n=== Error Comparison (500 steps) ===")
print(f"{'':20s}  {'RCM mean':>10s}  {'RCM max':>10s}  {'Tip mean':>10s}")
print(f"{'RCMController':20s}  "
      f"{rcm_errors.mean()*1000:10.4f}  "
      f"{rcm_errors.max()*1000:10.4f}  "
      f"{tip_errors_b.mean()*1000:10.4f}")
print(f"{'NaiveController':20s}  "
      f"{naive_errors.mean()*1000:10.4f}  "
      f"{naive_errors.max()*1000:10.4f}  "
      f"{tip_errors_a.mean()*1000:10.4f}")

from visualization.visualize import plot_rcm_comparison

# Temporarily comment out the plot call so the script runs through to the end
# plot_rcm_comparison(rcm_errors, naive_errors)

from simulation.sim_loop import sweep_lambda

lambdas = [0.001, 0.01, 0.05, 0.1, 0.5]

print("\n=== Lambda Sweep ===")
sweep_results = sweep_lambda(q0, p_rcm_target, lambdas)

from visualization.visualize import plot_lambda_sweep

# Temporarily comment out the plot call so the script runs through to the end
# plot_lambda_sweep(sweep_results)

import matplotlib.pyplot as plt
from visualization.visualize import plot_arm_static

q0 = np.zeros(6)
T, transforms = forward_kinematics(q0)
p_trocar, p_tip = get_tool_points(transforms[5])
p_rcm_target = p_trocar.copy()

# Temporarily comment out the static plot call so the script runs through to the end
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection='3d')
# plot_arm_static(transforms, p_trocar, p_tip, p_rcm_target, ax=ax)
# plt.tight_layout()
# plt.savefig('static_arm.png', dpi=150, bbox_inches='tight')
# plt.show()

from visualization.visualize import animate_simulation
from simulation.sim_loop import run_simulation

# Run fresh simulation for animation
history_anim = run_simulation(q0, p_rcm_target=p_rcm_target,
                               dt=0.01, steps=500)

animate_simulation(history_anim, p_rcm_target, dt=0.01, save_gif=True)

# ── Joint limit test ──────────────────────────────────────────────────────────
# Run simulation with joint limit handling
history_limits = run_simulation(q0, p_rcm_target=p_rcm_target,
                                dt=0.01, steps=500)

q_history = np.array(history_limits['q'])

print("\n=== Joint Limit Check ===")
q_min = np.array([-2*np.pi, -2*np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi])
q_max = np.array([ 2*np.pi,  2*np.pi,  np.pi,  2*np.pi,  2*np.pi,  2*np.pi])

for i in range(6):
    min_val = q_history[:, i].min()
    max_val = q_history[:, i].max()
    within = "OK" if min_val >= q_min[i] and max_val <= q_max[i] else "VIOLATED"
    print(f"Joint {i+1}: min={min_val:.3f} max={max_val:.3f} "
          f"limits=[{q_min[i]:.3f}, {q_max[i]:.3f}] {within}")

from simulation.sim_loop import print_metrics

print_metrics(history_limits)