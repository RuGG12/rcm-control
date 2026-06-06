import numpy as np
import matplotlib.pyplot as plt

def plot_rcm_comparison(rcm_errors, naive_errors, dt=0.01):
    """
    Plot RCM error over time for both controllers side by side.

    Parameters:
        rcm_errors   : array of |e_rcm| per step for RCMController
        naive_errors : array of |e_rcm| per step for NaiveController
        dt           : timestep in seconds
    """
    steps = len(rcm_errors)
    time  = np.arange(steps) * dt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('RCM Constraint Error: Full Controller vs Naive',
                 fontsize=14, fontweight='bold')

    # --- Controller B (ours) ---
    axes[0].plot(time, rcm_errors * 1000, color='steelblue', linewidth=1.2)
    axes[0].axhline(0.5, color='red', linestyle='--',
                    linewidth=1, label='0.5mm target')
    axes[0].set_title('Controller B — Full RCM Controller')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('|e_rcm| (mm)')
    axes[0].legend()
    axes[0].set_ylim(0, 1.0)
    axes[0].grid(True, alpha=0.3)

    # --- Controller A (naive) ---
    axes[1].plot(time, naive_errors * 1000, color='tomato', linewidth=1.2)
    axes[1].axhline(29.8, color='darkred', linestyle='--',
                    linewidth=1, label='29.8mm clinical limit')
    axes[1].set_title('Controller A — Naive (No Null Space, No Damping)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('|e_rcm| (mm)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('rcm_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved to rcm_comparison.png")


def plot_lambda_sweep(sweep_results, dt=0.01):
    """
    Plot RCM error over time for each lambda value.

    Parameters:
        sweep_results : dict mapping lambda -> array of |e_rcm| per step
        dt            : timestep in seconds
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Lambda Sweep — Effect on RCM Constraint Error',
                 fontsize=14, fontweight='bold')

    colors = ['green', 'steelblue', 'orange', 'tomato', 'darkred']
    lambdas = list(sweep_results.keys())
    steps = len(list(sweep_results.values())[0])
    time = np.arange(steps) * dt

    # Left plot — all lambda curves over time
    for i, lam in enumerate(lambdas):
        axes[0].plot(time, sweep_results[lam] * 1000,
                     color=colors[i], linewidth=1.2,
                     label=f'λ={lam}')
    axes[0].axhline(0.5, color='black', linestyle='--',
                    linewidth=1, label='0.5mm target')
    axes[0].axhline(29.8, color='black', linestyle=':',
                    linewidth=1, label='29.8mm clinical limit')
    axes[0].set_title('RCM Error Over Time')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('|e_rcm| (mm)')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Right plot — mean error per lambda (bar chart)
    means = [sweep_results[lam].mean() * 1000 for lam in lambdas]
    bar_colors = [c if m < 0.5 else 'tomato'
                  for c, m in zip(colors, means)]
    axes[1].bar([str(l) for l in lambdas], means,
                color=bar_colors, edgecolor='black', linewidth=0.5)
    axes[1].axhline(0.5, color='black', linestyle='--',
                    linewidth=1, label='0.5mm target')
    axes[1].set_title('Mean RCM Error per Lambda')
    axes[1].set_xlabel('Lambda (λ)')
    axes[1].set_ylabel('Mean |e_rcm| (mm)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('lambda_sweep.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved to lambda_sweep.png")


def plot_arm_static(transforms, p_trocar, p_tip, p_rcm_target, ax=None):
    """
    Draw the UR5 arm as a 3D wireframe at one configuration.

    Parameters:
        transforms   : list of 6 transforms from forward_kinematics
        p_trocar     : current trocar point position (3,)
        p_tip        : current tool tip position (3,)
        p_rcm_target : fixed RCM target point (3,)
        ax           : matplotlib 3D axis, created if None
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

    # Joint positions — base origin + one position per transform
    joint_positions = [np.array([0, 0, 0])]
    for T in transforms:
        joint_positions.append(T[:3, 3])

    joint_positions = np.array(joint_positions)

    # Draw arm links
    ax.plot(joint_positions[:, 0],
            joint_positions[:, 1],
            joint_positions[:, 2],
            'o-', color='steelblue', linewidth=2,
            markersize=5, label='Arm links')

    # Draw tool shaft — wrist to tip
    p_wrist = transforms[5][:3, 3]
    shaft = np.array([p_wrist, p_tip])
    ax.plot(shaft[:, 0], shaft[:, 1], shaft[:, 2],
            '-', color='darkorange', linewidth=3,
            label='Tool shaft')

    # Draw trocar point — red sphere
    ax.scatter(*p_rcm_target, color='red', s=100, zorder=5,
               label='RCM target')

    # Draw current trocar position
    ax.scatter(*p_trocar, color='pink', s=60, zorder=5,
               label='Trocar actual')

    # Draw tool tip
    ax.scatter(*p_tip, color='green', s=60, zorder=5,
               label='Tool tip')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend(fontsize=8)
    ax.set_title('UR5 RCM Controller — Static View')

    return ax

from matplotlib.animation import FuncAnimation, PillowWriter

def animate_simulation(history, p_rcm_target, dt=0.01, save_gif=False):
    """
    Animate the UR5 arm moving through the simulation.

    Parameters:
        history      : dict from run_simulation
        p_rcm_target : fixed RCM target point (3,)
        dt           : timestep
        save_gif     : if True, save animation as rcm_animation.gif
    """
    from robot.kinematics import forward_kinematics, get_tool_points

    fig = plt.figure(figsize=(14, 5))

    # 3D arm view
    ax1 = fig.add_subplot(131, projection='3d')

    # RCM error plot
    ax2 = fig.add_subplot(132)
    ax2.set_title('RCM Error Over Time')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('|e_rcm| (mm)')
    ax2.axhline(0.5, color='red', linestyle='--',
                linewidth=1, label='0.5mm target')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Manipulability plot
    ax3 = fig.add_subplot(133)
    ax3.set_title('Manipulability Over Time')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('w = sqrt(det(JJ^T))')
    ax3.grid(True, alpha=0.3)

    steps = len(history['q'])
    time  = np.arange(steps) * dt
    e_rcm_mm = np.array(history['e_rcm_norm']) * 1000

    # Precompute tip positions
    tip_positions = np.array(history['p_tip'])

    def update(frame):
        ax1.cla()
        ax2.cla()
        ax3.cla()

        q = history['q'][frame]
        T, transforms = forward_kinematics(q)
        p_trocar, p_tip = get_tool_points(transforms[5])

        # Draw arm
        plot_arm_static(transforms, p_trocar, p_tip, p_rcm_target, ax=ax1)
        ax1.set_title(f'Step {frame}/{steps}')

        # Draw RCM error up to current frame
        ax2.plot(time[:frame], e_rcm_mm[:frame],
                 color='steelblue', linewidth=1.2)
        ax2.axhline(0.5, color='red', linestyle='--', linewidth=1,
                    label='0.5mm target')

        # Mark drift corrections as vertical green lines
        for i, corrected in enumerate(history['drift_corrections'][:frame]):
            if corrected:
                ax2.axvline(time[i], color='green', alpha=0.5,
                            linewidth=1, linestyle=':')

        ax2.set_xlim(0, time[-1])
        ax2.set_ylim(0, max(1.0, e_rcm_mm.max() * 1.1))
        ax2.set_title('RCM Error Over Time')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('|e_rcm| (mm)')
        ax2.grid(True, alpha=0.3)

        # Draw manipulability over time
        J_current = __import__(
            'robot.kinematics', fromlist=['compute_jacobian']
        ).compute_jacobian(q, transforms)

        manip = np.sqrt(max(0, np.linalg.det(J_current @ J_current.T)))
        history.setdefault('manipulability', []).append(manip)

        manip_history = history.get('manipulability', [])
        ax3.plot(time[:len(manip_history)], manip_history,
                 color='purple', linewidth=1.2)
        ax3.set_title('Manipulability Over Time')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('w = sqrt(det(JJ^T))')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

    anim = FuncAnimation(fig, update, frames=range(0, steps, 5),
                         interval=50)

    if save_gif:
        print("Saving GIF... this takes a minute")
        writer = PillowWriter(fps=20)
        anim.save('rcm_animation.gif', writer=writer)
        print("Saved to rcm_animation.gif")

    plt.show()
    return anim

