import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def plot_rcm_comparison(rcm_errors, naive_errors, dt=0.01):
    # Side-by-side plot of RCM error for the full controller vs naive baseline.
    steps = len(rcm_errors)
    time  = np.arange(steps) * dt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('RCM Constraint Error: Full Controller vs Naive', fontsize=14)

    axes[0].plot(time, rcm_errors * 1000, color='steelblue', linewidth=1.2)
    axes[0].axhline(0.5, color='red', linestyle='--', linewidth=1, label='0.5mm target')
    axes[0].set_title('Full RCM Controller')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('|e_rcm| (mm)')
    axes[0].legend()
    axes[0].set_ylim(0, 1.0)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time, naive_errors * 1000, color='tomato', linewidth=1.2)
    axes[1].axhline(29.8, color='darkred', linestyle='--', linewidth=1, label='29.8mm literature tolerance')
    axes[1].set_title('Naive Controller (no null space, no damping)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('|e_rcm| (mm)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('rcm_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved to rcm_comparison.png")


def plot_lambda_sweep(sweep_results, dt=0.01):
    # Two plots: RCM error over time for each lambda, and mean error per lambda.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Lambda Sweep — Effect on RCM Error', fontsize=14)

    colors = ['green', 'steelblue', 'orange', 'tomato', 'darkred']
    lambdas = list(sweep_results.keys())
    steps = len(list(sweep_results.values())[0])
    time = np.arange(steps) * dt

    for i, lam in enumerate(lambdas):
        axes[0].plot(time, sweep_results[lam] * 1000,
                     color=colors[i], linewidth=1.2, label=f'λ={lam}')
    axes[0].axhline(0.5,  color='black', linestyle='--', linewidth=1, label='0.5mm target')
    axes[0].axhline(29.8, color='black', linestyle=':',  linewidth=1, label='29.8mm (lit)')
    axes[0].set_title('RCM Error Over Time')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('|e_rcm| (mm)')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    means = [sweep_results[lam].mean() * 1000 for lam in lambdas]
    bar_colors = [c if m < 0.5 else 'tomato' for c, m in zip(colors, means)]
    axes[1].bar([str(l) for l in lambdas], means,
                color=bar_colors, edgecolor='black', linewidth=0.5)
    axes[1].axhline(0.5, color='black', linestyle='--', linewidth=1, label='0.5mm target')
    axes[1].set_title('Mean RCM Error per Lambda')
    axes[1].set_xlabel('Lambda')
    axes[1].set_ylabel('Mean |e_rcm| (mm)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('lambda_sweep.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved to lambda_sweep.png")


def plot_arm_static(transforms, p_trocar, p_tip, p_rcm_target, ax=None):
    # 3D wireframe of the arm at one configuration.
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

    joint_positions = [np.array([0, 0, 0])]
    for T in transforms:
        joint_positions.append(T[:3, 3])
    joint_positions = np.array(joint_positions)

    ax.plot(joint_positions[:, 0],
            joint_positions[:, 1],
            joint_positions[:, 2],
            'o-', color='steelblue', linewidth=2, markersize=5, label='Arm')

    p_wrist = transforms[5][:3, 3]
    shaft = np.array([p_wrist, p_tip])
    ax.plot(shaft[:, 0], shaft[:, 1], shaft[:, 2],
            '-', color='darkorange', linewidth=3, label='Tool shaft')

    ax.scatter(*p_rcm_target, color='red',  s=100, zorder=5, label='RCM target')
    ax.scatter(*p_trocar,     color='pink', s=60,  zorder=5, label='Trocar actual')
    ax.scatter(*p_tip,        color='green', s=60, zorder=5, label='Tip')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend(fontsize=8)
    ax.set_title('UR5 RCM Controller')

    return ax


def animate_simulation(history, p_rcm_target, dt=0.01, save_gif=False):
    # Animate the arm through the simulation.
    # Three panels: 3D arm view, RCM error over time, manipulability over time.
    from robot.kinematics import forward_kinematics, get_tool_points, compute_jacobian

    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    steps = len(history['q'])
    time  = np.arange(steps) * dt
    e_rcm_mm = np.array(history['e_rcm_norm']) * 1000

    def update(frame):
        ax1.cla()
        ax2.cla()
        ax3.cla()

        q = history['q'][frame]
        T, transforms = forward_kinematics(q)
        p_trocar, p_tip = get_tool_points(transforms[5])

        plot_arm_static(transforms, p_trocar, p_tip, p_rcm_target, ax=ax1)
        ax1.set_title(f'Step {frame}/{steps}')

        ax2.plot(time[:frame], e_rcm_mm[:frame], color='steelblue', linewidth=1.2)
        ax2.axhline(0.5, color='red', linestyle='--', linewidth=1, label='0.5mm target')
        for i, corrected in enumerate(history['drift_corrections'][:frame]):
            if corrected:
                ax2.axvline(time[i], color='green', alpha=0.5, linewidth=1, linestyle=':')
        ax2.set_xlim(0, time[-1])
        ax2.set_ylim(0, max(1.0, e_rcm_mm.max() * 1.1))
        ax2.set_title('RCM Error')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('|e_rcm| (mm)')
        ax2.grid(True, alpha=0.3)

        J_current = compute_jacobian(q, transforms)
        manip = np.sqrt(max(0, np.linalg.det(J_current @ J_current.T)))
        history.setdefault('manipulability', []).append(manip)
        manip_history = history.get('manipulability', [])

        ax3.plot(time[:len(manip_history)], manip_history, color='purple', linewidth=1.2)
        ax3.set_title('Manipulability')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('w = sqrt(det(JJt))')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

    anim = FuncAnimation(fig, update, frames=range(0, steps, 5), interval=50)

    if save_gif:
        print("Saving GIF...")
        writer = PillowWriter(fps=20)
        anim.save('rcm_animation.gif', writer=writer)
        print("Saved to rcm_animation.gif")

    plt.show()
    return anim
