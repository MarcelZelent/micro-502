import numpy as np

# ---------------------------------------------------------------------
# ─── fixed camera-on-body geometry (from the appendix) ───────────────
# camera → body rotation  (columns are camera axes in body frame)
R_cb = np.array([[ -1, 0,  0],      # x_cam points −y_body
                 [ 0,  0, -1],      # y_cam points −z_body
                 [ 0,  -1,  0]])     # z_cam points  x_body

# camera offset in body frame  [m]
t_cb = np.array([0.03, 0.0, 0.01])

# ---------------------------------------------------------------------
# ─── MAIN CONTROL ROUTINE ─────────────────────────────────────────────
def get_command(sensor_data, _camera_data, _dt):
    """
    1. climb to 1.0 m, then hold (x0, y0, 1.0, current yaw)
    2. compute T_wc (camera → world) every call and print it
    """
    # -------- take-off target
    if 'home' not in get_command.__dict__:
        get_command.home = (
            sensor_data['x_global'],
            sensor_data['y_global']
        )

    x0, y0 = get_command.home

    # -------- build body→world rotation from roll-pitch-yaw
    cr, sr = np.cos(sensor_data['roll']),  np.sin(sensor_data['roll'])
    cp, sp = np.cos(sensor_data['pitch']), np.sin(sensor_data['pitch'])
    cy, sy = np.cos(sensor_data['yaw']),   np.sin(sensor_data['yaw'])

    R_wb = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [  -sp,            cp*sr,            cp*cr]
    ])                       # body → world

    # -------- camera→world rotation
    R_wc = R_wb @ R_cb

    # -------- camera origin in world
    t_wb = np.array([sensor_data['x_global'],
                     sensor_data['y_global'],
                     sensor_data['z_global']])
    t_wc = t_wb + R_wb @ t_cb

    # -------- homogeneous 4×4 transform
    T_wc = np.eye(4)
    T_wc[:3, :3] = R_wc
    T_wc[:3,  3] = t_wc

    # -------- print once per frame
    # print("T_wc =\n", T_wc)
    print(np.array2string(T_wc, precision=3, suppress_small=True))

    # -------- simple hover logic
    z_cmd = 1.0
    if sensor_data['z_global'] < 0.95:          # still rising
        z_cmd = 1.0                             # continue to target height

    return [x0, y0, z_cmd, sensor_data['yaw']]
