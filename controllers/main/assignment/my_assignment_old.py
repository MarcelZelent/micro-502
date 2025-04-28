import numpy as np
import time
import cv2

# The available ground truth state measurements can be accessed by calling sensor_data[item]. All values of "item" are provided as defined in main.py within the function read_sensors. 
# The "item" values that you may later retrieve for the hardware project are:
# "x_global": Global X position
# "y_global": Global Y position
# "z_global": Global Z position
# 'v_x": Global X velocity
# "v_y": Global Y velocity
# "v_z": Global Z velocity
# "ax_global": Global X acceleration
# "ay_global": Global Y acceleration
# "az_global": Global Z acceleration (With gravtiational acceleration subtracted)
# "roll": Roll angle (rad)
# "pitch": Pitch angle (rad)
# "yaw": Yaw angle (rad)
# "q_x": X Quaternion value
# "q_y": Y Quaternion value
# "q_z": Z Quaternion value
# "q_w": W Quaternion value

# A link to further information on how to access the sensor data on the Crazyflie hardware for the hardware practical can be found here: https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/api/logs/#stateestimate

# ----------  CONSTANTS  ----------
SEARCH_YAW_RATE = 0.35            # rad / s  (≈ 20 °/s)
SHIFT_SIDE      = 0.30            # metres body-Y
GATE_HSV_LO     = (120,  80,  50) # purple mask
GATE_HSV_HI     = (160, 255, 255)

# ----------  GLOBALS  ----------
fsm_state   = 'TAKEOFF'
first_pose  = None                # (corners, T_wc) from view-1
second_pose = None                # (corners, T_wc) from view-2
gate_target = None                # np.array([x,y,z])
x0 = y0 = None                    # launch coordinates


# def get_command(sensor_data, camera_data, dt):

#     # NOTE: Displaying the camera image with cv2.imshow() will throw an error because GUI operations should be performed in the main thread.
#     # If you want to display the camera image you can call it main.py.

#     # Take off example
#     if sensor_data['z_global'] < 0.49:
#         control_command = [sensor_data['x_global'], sensor_data['y_global'], 1.0, sensor_data['yaw']]
#         return control_command

#     # ---- YOUR CODE HERE ----
#     control_command = [sensor_data['x_global'], sensor_data['y_global'], 1.0, sensor_data['yaw']]
    
#     return control_command # Ordered as array with: [pos_x_cmd, pos_y_cmd, pos_z_cmd, yaw_cmd] in meters and radians


# ----------  MAIN CONTROL  ----------
def get_command(sensor_data, camera_data, dt):
    global fsm_state, first_pose, second_pose, gate_target, x0, y0

    # Save launch reference once
    if x0 is None: x0, y0 = sensor_data['x_global'], sensor_data['y_global']

    # TAKEOFF -------------------------------------------------------------
    if fsm_state == 'TAKEOFF':
        if sensor_data['z_global'] < 0.95:
            return [x0, y0, 1.0, sensor_data['yaw']]
        fsm_state = 'SEARCH'

    # SEARCH – spin CCW until a purple gate appears -----------------------
    if fsm_state == 'SEARCH':
        img   = camera_data['image']
        crnrs = detect_purple(img)
        if crnrs is None:
            # keep height, add left-yaw
            return [x0, y0, 1.0, sensor_data['yaw'] + SEARCH_YAW_RATE*dt]
        first_pose = (crnrs, pose_matrix(sensor_data))
        fsm_state  = 'SHIFT'
        return [x0, y0, 1.0, sensor_data['yaw']]   # stop turning

    # SHIFT – slide left in body Y ----------------------------------------
    if fsm_state == 'SHIFT':
        # body-frame left equals world −sin(yaw) x̂ + cos(yaw) ŷ
        dy = SHIFT_SIDE * np.cos(sensor_data['yaw'])
        dx =-SHIFT_SIDE * np.sin(sensor_data['yaw'])
        tgt_x, tgt_y = first_pose[1][:3,3][:2] + np.array([dx,dy])
        if np.hypot(sensor_data['x_global']-tgt_x,
                    sensor_data['y_global']-tgt_y) < 0.05:
            fsm_state = 'OBS2'
        return [tgt_x, tgt_y, 1.0, sensor_data['yaw']]

    # OBS2 – capture 2nd view --------------------------------------------
    if fsm_state == 'OBS2':
        crnrs = detect_purple(camera_data['image'])
        if crnrs is None:           # keep position, wait until in view
            return [sensor_data['x_global'], sensor_data['y_global'], 1.0, sensor_data['yaw']]
        second_pose = (crnrs, pose_matrix(sensor_data))
        fsm_state   = 'TRIANG'

    # TRIANG – compute gate center ---------------------------------------
    if fsm_state == 'TRIANG':
        K = camera_data['K']        # assume 3×3 intrinsics in dict
        gate_target = triangulate(first_pose[0], first_pose[1],
                                  second_pose[0], second_pose[1], K)
        fsm_state = 'NAVIGATE'

    # NAVIGATE – fly toward gate center ----------------------------------
    if fsm_state == 'NAVIGATE':
        gx, gy, gz = gate_target
        if np.hypot(sensor_data['x_global']-gx,
                    sensor_data['y_global']-gy) < 0.1:
            # arrive: maintain depth through gate (optional)
            return [gx, gy, gz, sensor_data['yaw']]
        return [gx, gy, gz, sensor_data['yaw']]

    # fallback (should not hit)
    return [sensor_data['x_global'], sensor_data['y_global'], 1.0, sensor_data['yaw']]




# ----------  HELPERS  ----------
def detect_purple(img):
    hsv   = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask  = cv2.inRange(hsv, GATE_HSV_LO, GATE_HSV_HI)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    cnt   = max(cnts, key=cv2.contourArea)
    eps   = 0.02 * cv2.arcLength(cnt, True)
    poly  = cv2.approxPolyDP(cnt, eps, True)
    if len(poly) != 4: return None
    return np.squeeze(poly, 1).astype(float)      # shape (4,2)

def pose_matrix(s):
    cr, sr = np.cos(s['roll']),  np.sin(s['roll'])
    cp, sp = np.cos(s['pitch']), np.sin(s['pitch'])
    cy, sy = np.cos(s['yaw']),   np.sin(s['yaw'])
    R = np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                  [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                  [  -sp,              cp*sr,              cp*cr]])
    T = np.eye(4); T[:3,:3] = R; T[:3, 3] = [s['x_global'], s['y_global'], s['z_global']]
    return T                             # world←camera

def triangulate(p1, T1, p2, T2, K):
    """p1,p2: (4,2) pixel corners; T*: 4×4;  K: 3×3 intrinsics"""
    # build 3×4 projection matrices P = K [R|t] in world frame
    P1 = K @ np.linalg.inv(T1)[:3]       # camera←world so invert
    P2 = K @ np.linalg.inv(T2)[:3]
    X_h = cv2.triangulatePoints(P1, P2, p1.T, p2.T)  # 4×4
    X   = (X_h[:3] / X_h[3]).T                       # (4,3)
    return X.mean(axis=0)                            # center of gate
