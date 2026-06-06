import numpy as np 

def dh_transform(a, d, alpha, theta):
    """
    Compute the 4x4 DH transformation matrix for a single joint.

    Parameters:
    a : link length(meters) - distance along x axis
    d : link offset (meters) - distance along z axis
    alpha: link twist (radians) - rotation about x axis
    theta: joint angle (radians) - rotation about z axis (this is our variable)

    Returns:
    T : 4x4 numpy array - tranformation matrix from frame i-1 to frame i
    """

    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    
    T = np.array([
        [ct, -st*ca, st*sa, a*ct],
        [st, ct*ca, -ct*sa, a*st],
        [0, sa, ca, d],
        [0, 0, 0, 1]
    ])

    return T

# UR5 DH Parameters
# Each row is one joint: [a, d, alpha, theta_offset]
# theta_offset is added to the joint variable q at runtime
# Source: Universal Robots UR5 technical documentation

UR5_DH = [
    #    a        d        alpha     theta_offset
    [0.0,      0.08916,  np.pi/2,   0.0],   # Joint 1
    [-0.42500, 0.0,      0.0,       0.0],   # Joint 2
    [-0.39225, 0.0,      0.0,       0.0],   # Joint 3
    [0.0,      0.10915,  np.pi/2,   0.0],   # Joint 4
    [0.0,      0.09465, -np.pi/2,   0.0],   # Joint 5
    [0.0,      0.0823,   0.0,       0.0],   # Joint 6
]

# UR5 joint limits in radians
UR5_JOINT_LIMITS = {
    'min': np.array([-2*np.pi, -2*np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi]),
    'max': np.array([ 2*np.pi,  2*np.pi,  np.pi,  2*np.pi,  2*np.pi,  2*np.pi])
}


def joint_limit_gradient(q):
    """
    Compute gradient of joint limit avoidance potential.
    Pushes joints away from their limits using a quadratic potential field.

    Parameters:
        q : current joint angles (6,)

    Returns:
        grad : gradient vector (6,) pointing away from limits
    """
    q_min = UR5_JOINT_LIMITS['min']
    q_max = UR5_JOINT_LIMITS['max']
    q_mid = 0.5 * (q_max + q_min)
    q_range = q_max - q_min

    # Normalized distance from center
    grad = (q - q_mid) / (q_range ** 2)

    return -grad

# Tool configuration — adjust these to change surgical setup
TOOL_LENGTH   = 0.20   # meters, wrist to tip
TROCAR_DEPTH  = 0.10   # meters, wrist to trocar point

def forward_kinematics(q):
    """
    Compute full FK for UR5 given 6 joint angles.
    
    Parameters:
        q : array of 6 joint angles in radians
    
    Returns:
        T : 4x4 transformation matrix from base to end effector
        transforms : list of 6 intermediate 4x4 matrices (one per joint)
    """
    q = np.asarray(q, dtype=float)
    
    T = np.eye(4)  # start at identity - base frame
    transforms = []
    
    for i in range(6):
        a, d, alpha, theta_offset = UR5_DH[i]
        theta = q[i] + theta_offset
        Ti = dh_transform(a, d, alpha, theta)
        T = T @ Ti
        transforms.append(T.copy())
    
    return T, transforms

def get_position(T):
    """
    Extract XYZ position from a 4x4 transformation matrix.
    
    Parameters:
        T : 4x4 transformation matrix
    
    Returns:
        p : 3D position vector [x, y, z]
    """
    return T[:3, 3]


def get_rotation(T):
    """
    Extract 3x3 rotation matrix from a 4x4 transformation matrix.
    
    Parameters:
        T : 4x4 transformation matrix
    
    Returns:
        R : 3x3 rotation matrix
    """
    return T[:3, :3]


def compute_jacobian(q, transforms, p_target=None):
    """
    Compute the geometric Jacobian for any point on the robot.

    Parameters:
        q          : array of 6 joint angles in radians
        transforms : list of 6 intermediate transforms from forward_kinematics
        p_target   : 3D position to compute Jacobian for.
                     If None, defaults to wrist position.

    Returns:
        J : 3x6 Jacobian matrix
    """
    if p_target is None:
        p_target = transforms[5][:3, 3]

    J = np.zeros((3, 6))

    for i in range(6):
        if i == 0:
            z = np.array([0, 0, 1])
            p = np.array([0, 0, 0])
        else:
            z = transforms[i-1][:3, 2]
            p = transforms[i-1][:3, 3]

        J[:, i] = np.cross(z, p_target - p)

    return J


def get_tool_points(T_wrist, tool_length=TOOL_LENGTH, trocar_depth=TROCAR_DEPTH):
    """
    Compute the trocar (RCM) and tip positions of the surgical instrument
    given the wrist frame.

    The tool shaft lies along the wrist frame z-axis. Two points are
    defined by projecting trocar_depth and tool_length along that axis
    from the wrist origin.

    Parameters:
        T_wrist      : 4x4 transformation matrix of the wrist (last FK frame)
        tool_length  : distance from wrist to tip (default: TOOL_LENGTH)
        trocar_depth : distance from wrist to trocar/RCM point (default: TROCAR_DEPTH)

    Returns:
        p_trocar : (3,) position of the RCM / trocar point
        p_tip    : (3,) position of the instrument tip
    """
    p_wrist = T_wrist[:3, 3]          # wrist origin in base frame
    z_hat   = T_wrist[:3, 2]          # wrist z-axis = tool shaft direction

    p_trocar = p_wrist + trocar_depth * z_hat
    p_tip    = p_wrist + tool_length  * z_hat

    return p_trocar, p_tip