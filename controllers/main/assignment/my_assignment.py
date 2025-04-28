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


gates = []
# A link to further information on how to access the sensor data on the Crazyflie hardware for the hardware practical can be found here: https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/api/logs/#stateestimate

start_flag = False
buffer_radius = 1
threshold = 0.5
locked_in = False
start_pose = None

start_pose = None
motion_planning_started = False
trajectory_setpoints = None
lap_counter = 0
trajectory_index = 0
returning_to_start = False
pink_prev      = 0.0     # last frame’s ratio
pink_rising    = False   # we are currently inside the rising edge
run_out_target = None      # xyz point 0.5 m past 5th gate


def get_command(sensor_data, camera_data, dt):
    global start_flag
    global buffer_radius
    global threshold
    global locked_in
    global start_pose
    global motion_planning_started
    global trajectory_setpoints
    global lap_counter
    global trajectory_index
    global returning_to_start
    global pink_prev
    global pink_rising
    global run_out_target

    if sensor_data['z_global'] < 0.49:
        control_command = [sensor_data['x_global'], sensor_data['y_global'], 1.0, sensor_data['yaw']]
        return control_command

    if start_flag == False:
        start_pose = np.array([sensor_data['x_global'], sensor_data['y_global'], 1.0, sensor_data['yaw']])
        print("Start pose: ", start_pose)
        start_flag = True

    current_pose = np.array([sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']])

    # run-out phase: fly straight after reaching 5th gate
    if run_out_target is not None and not motion_planning_started:
        dx, dy = run_out_target[0] - sensor_data['x_global'], run_out_target[1] - sensor_data['y_global']
        dz     = run_out_target[2] - sensor_data['z_global']
        if np.linalg.norm([dx, dy, dz]) < 0.05:        
            run_out_target = None                     
        else:                                         
            return [run_out_target[0], run_out_target[1], run_out_target[2], sensor_data['yaw']]

    # Plan trajectory after finding 5 gates
    if len(gates) >= 5 and not motion_planning_started:

        print("Found all gates! Planning trajectory…")

        # list so we can append
        pos_wp = [tuple(start_pose[:3])] + [tuple(g[:3]) for g in gates]

        # push 0.5 m beyond the last gate
        g4 = np.array(pos_wp[-2])          # gate 4 centre
        g5 = np.array(pos_wp[-1])          # gate 5 centre
        dir_unit = (g5 - g4) / (np.linalg.norm(g5 - g4) + 1e-6)
        extra_wp = tuple(g5 + 0.5 * dir_unit)   # 0.5 m further
        pos_wp.append(extra_wp)                 # go through gate
        pos_wp.append(tuple(start_pose[:3]))    # then back to start

        v_target = 2                              # to adjust speed
        cumdist  = [0.0]
        for i in range(1, len(pos_wp)):
            cumdist.append(cumdist[-1] + np.linalg.norm(np.array(pos_wp[i]) -
                                                        np.array(pos_wp[i-1])))
        times = np.array(cumdist) / v_target       

        # Build planner object without calling its __init__
        mp            = MotionPlanner3D.__new__(MotionPlanner3D)
        mp.disc_steps = 100
        mp.vel_lim    = 7.0
        mp.acc_lim    = 50.0
        mp.path       = pos_wp
        mp.times      = times                       # our slow timeline
        mp.run_planner([], pos_wp)                  # generate xyz trajectory

        # xyz set-points only (yaw column = 0)
        trajectory_setpoints = mp.trajectory_setpoints[:, :3]
        trajectory_index     = 0
        lap_counter          = 0
        returning_to_start   = False
        motion_planning_started = True

        print(f"Trajectory planned  |  total time ≈ {times[-1]:.1f}s  |  v ≈ {v_target} m/s")

    # Follow generated trajectory
    if motion_planning_started:

        # Step through the trajectory
        if trajectory_index < len(trajectory_setpoints):
            x, y, z = trajectory_setpoints[trajectory_index]; trajectory_index += 1
            return [x, y, z, sensor_data['yaw']]    # keep current yaw

        # End of path reached → complete lap
        if not returning_to_start:
            lap_counter += 1
            print(f"Completed lap {lap_counter}")
            if lap_counter == 3:                    # discovery lap + 2 extra laps
                print("All laps done, hovering.")
                return [sensor_data['x_global'], sensor_data['y_global'], 1.0, sensor_data['yaw']]
            returning_to_start = True               # fly exactly to start first
            return [start_pose[0], start_pose[1], start_pose[2], sensor_data['yaw']]

        # Now at start → restart lap
        returning_to_start = False
        trajectory_index   = 0
        return [start_pose[0], start_pose[1], start_pose[2], sensor_data['yaw']]

    # Normal pink detection and gate logging
    found_pink = find_pink(camera_data)

    if locked_in:
        pink = detect_gate_and_log_position(camera_data, current_pose)
        local_cmd = np.array([1, 0, 0])
        quaternion = np.array([sensor_data['q_x'], sensor_data['q_y'], sensor_data['q_z'], sensor_data['q_w']])
        cmd = local_to_global_vector(local_cmd, quaternion)
        control_command = [sensor_data['x_global']+cmd[0], sensor_data['y_global']+cmd[1], sensor_data['z_global'], sensor_data['yaw']]

    if found_pink is not None:
        pink_x, pink_y = found_pink

        height_error = 150 - pink_y
        HK_p = 0.004
        change_h = height_error * HK_p

        orientation_error = 150 - pink_x
        YK_p = 0.001
        change_yaw = orientation_error * YK_p

        local_cmd = np.array([0.2, 0, 0])
        quaternion = np.array([sensor_data['q_x'], sensor_data['q_y'], sensor_data['q_z'], sensor_data['q_w']])
        cmd = local_to_global_vector(local_cmd, quaternion)

        pink_rat = detect_gate_and_log_position(camera_data, current_pose)

        control_command = [
            sensor_data['x_global'] + cmd[0],
            sensor_data['y_global'] + cmd[1],
            sensor_data['z_global'] + change_h,
            sensor_data['yaw'] + change_yaw
        ]
        
        errors_combined = abs(height_error) + abs(orientation_error)
        if len(gates) == 4 and errors_combined < 30 and is_far_from_existing_gates(current_pose, gates, buffer_radius) and not locked_in:
            print("Found 4th gate!")
            locked_in = True
    else:
        if not locked_in and not len(gates) >= 5:
            control_command = [sensor_data['x_global']+0.1, sensor_data['y_global']+0.1, sensor_data['z_global'], sensor_data['yaw'] + 0.1]

    return control_command # Ordered as array with: [pos_x_cmd, pos_y_cmd, pos_z_cmd, yaw_cmd] in meters and radians

def is_far_from_existing_gates(position, gates, radius):
    for gate in gates:
        distance = np.linalg.norm(np.array(position) - np.array(gate))
        if distance < radius:
            return False
    return True

def go_to_goal_pose(pose):
    print(f" Reached 5 gates! Navigating to final pose at {pose}")
    control_command = pose
    return control_command
    

def detect_gate_and_log_position(image, position, hi=0.35, lo=0.15):
    """
    Detect a gate when the pink coverage rises above *hi* and
    subsequently falls below *lo* (a peak).  Adds the current
    position to `gates` at the moment of the *falling* edge.
    """

    global pink_prev, pink_rising

    hsv         = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask        = cv2.inRange(hsv, (140, 50, 50), (170, 255, 255))
    pink_ratio  = np.count_nonzero(mask) / mask.size

    # print("pink ratio: ", pink_ratio)

    # rising edge
    if not pink_rising and pink_ratio > hi:
        pink_rising = True

    # falling edge → peak completed
    if pink_rising and pink_ratio < lo:
        pink_rising = False
        if is_far_from_existing_gates(position, gates, buffer_radius):
            gates.append(position.copy())
            print(f"Gate logged at {position}  |  total: {len(gates)}")

            # if this is the 5th gate → compute run-out point
            if len(gates) == 5:
                prev = np.array(gates[-2][:3])
                curr = np.array(gates[-1][:3])
                dir_unit = (curr - prev) / (np.linalg.norm(curr - prev) + 1e-6)
                run_out_target = tuple(curr + 0.5 * dir_unit)   # 0.5 m ahead


    pink_prev = pink_ratio
    return pink_ratio

def find_pink(image_data):
    
    hsv = cv2.cvtColor(image_data, cv2.COLOR_BGR2HSV)
    lower_pink = np.array([140, 50, 50])   
    upper_pink = np.array([170, 255, 255]) 
    
    mask = cv2.inRange(hsv, lower_pink, upper_pink)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        
        largest_contour = max(contours, key=cv2.contourArea)  
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            pink_pixel_middle = (cx, cy)
        else:
            pink_pixel_middle = None
    else:
        pink_pixel_middle = None

    return pink_pixel_middle

import numpy as np

def quat_to_rot_matrix(q):

    x, y, z, w = q
    # Normalize
    norm = np.sqrt(x*x + y*y + z*z + w*w)
    x /= norm
    y /= norm
    z /= norm
    w /= norm

    R = np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

    return R

def local_to_global_vector(local_vector, quaternion):
    R = quat_to_rot_matrix(quaternion)
    global_vector = np.dot(R, local_vector)
    return global_vector


### COPIED FROM EXERCISES

from lib.a_star_3D import AStar3D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class MotionPlanner3D():
    
    #Question: SIMON PID, what is vel_max set for PID? Check should be same here
    def __init__(self, start, obstacles, bounds, grid_size, goal):
        # Inputs:
        # - start: The sequence of input path waypoints provided by the path-planner, including the start and final goal position: Vector of m waypoints, consisting of a tuple with three reference positions each as provided by AStar 
        # - obstacles: 2D array with obstacle locations and obstacle widths [x, y, z, dx, dy, dz]*n_obs
        # - bounds: The bounds of the environment [x_min, x_max, y_min, y_max, z_min, z_max]
        # - grid_size: The grid size of the environment (scalar)
        # - goal: The final goal position of the drone (tuple of 3) 
        
        ## DO NOT MODIFY --------------------------------------------------------------------------------------- ##
        self.ast = AStar3D(start, goal, grid_size, obstacles, bounds)
        self.path = self.ast.find_path()

        self.trajectory_setpoints = None

        self.init_params(self.path)

        self.run_planner(obstacles, self.path)

        # ---------------------------------------------------------------------------------------------------- ##

    def run_planner(self, obs, path_waypoints):    
        # Run the subsequent functions to compute the polynomial coefficients and extract and visualize the trajectory setpoints
         ## DO NOT MODIFY --------------------------------------------------------------------------------------- ##
    
        poly_coeffs = self.compute_poly_coefficients(path_waypoints)
        self.trajectory_setpoints, self.time_setpoints = self.poly_setpoint_extraction(poly_coeffs, obs, path_waypoints)

        ## ---------------------------------------------------------------------------------------------------- ##

    def init_params(self, path_waypoints):

        # Inputs:
        # - path_waypoints: The sequence of input path waypoints provided by the path-planner, including the start and final goal position: Vector of m waypoints, consisting of a tuple with three reference positions each as provided by AStar

        # TUNE THE FOLLOWING PARAMETERS (PART 2) ----------------------------------------------------------------- ##
        self.disc_steps = 20 #Integer number steps to divide every path segment into to provide the reference positions for PID control # IDEAL: Between 10 and 20
        self.vel_lim = 7.0 #Velocity limit of the drone (m/s)
        self.acc_lim = 50.0 #Acceleration limit of the drone (m/s²)
        t_f = 2.8  # Final time at the end of the path (s)

        # Determine the number of segments of the path
        self.times = np.linspace(0, t_f, len(path_waypoints)) # The time vector at each path waypoint to traverse (Vector of size m) (must be 0 at start)

    def compute_poly_matrix(self, t):
        # Inputs:
        # - t: The time of evaluation of the A matrix (t=0 at the start of a path segment, else t >= 0) [Scalar]
        # Outputs: 
        # - The constraint matrix "A_m(t)" [5 x 6]
        # The "A_m" matrix is used to represent the system of equations [x, \dot{x}, \ddot{x}, \dddot{x}, \ddddot{x}]^T  = A_m(t) * poly_coeffs (where poly_coeffs = [c_0, c_1, c_2, c_3, c_4, c_5]^T and represents the unknown polynomial coefficients for one segment)
        A_m = np.zeros((5,6))
        
        # TASK: Fill in the constraint factor matrix values where each row corresponds to the positions, velocities, accelerations, snap and jerk here
        # SOLUTION ---------------------------------------------------------------------------------- ## 
        
        A_m = np.array([
            [t**5, t**4, t**3, t**2, t, 1], #pos
            [5*(t**4), 4*(t**3), 3*(t**2), 2*t, 1, 0], #vel
            [20*(t**3), 12*(t**2), 6*t, 2, 0, 0], #acc  
            [60*(t**2), 24*t, 6, 0, 0, 0], #jerk
            [120*t, 24, 0, 0, 0, 0] #snap
        ])

        return A_m

    def compute_poly_coefficients(self, path_waypoints):
        
        # Computes a minimum jerk trajectory given time and position waypoints.
        # Inputs:
        # - path_waypoints: The sequence of input path waypoints provided by the path-planner, including the start and final goal position: Vector of m waypoints, consisting of a tuple with three reference positions each as provided by AStar
        # Outputs:
        # - poly_coeffs: The polynomial coefficients for each segment of the path [6(m-1) x 3]

        # Use the following variables and the class function self.compute_poly_matrix(t) to solve for the polynomial coefficients
        
        seg_times = np.diff(self.times) #The time taken to complete each path segment
        m = len(path_waypoints) #Number of path waypoints (including start and end)
        poly_coeffs = np.zeros((6*(m-1),3))

        # YOUR SOLUTION HERE ---------------------------------------------------------------------------------- ## 

        # 1. Fill the entries of the constraint matrix A and equality vector b for x,y and z dimensions in the system A * poly_coeffs = b. Consider the constraints according to the lecture: We should have a total of 6*(m-1) constraints for each dimension.
        # 2. Solve for poly_coeffs given the defined system

        for dim in range(3):  # Compute for x, y, and z separately
            A = np.zeros((6*(m-1), 6*(m-1)))
            b = np.zeros(6*(m-1))
            pos = np.array([p[dim] for p in path_waypoints])
            A_0 = self.compute_poly_matrix(0) # A_0 gives the constraint factor matrix A_m for any segment at t=0, this is valid for the starting conditions at every path segment

            # SOLUTION
            row = 0
            for i in range(m-1):
                pos_0 = pos[i] #Starting position of the segment
                pos_f = pos[i+1] #Final position of the segment
                # The prescribed zero velocity (v) and acceleration (a) values at the start and goal position of the entire path
                v_0, a_0 = 0, 0
                v_f, a_f = 0, 0
                A_f = self.compute_poly_matrix(seg_times[i]) # A_f gives the constraint factor matrix A_m for a segment i at its relative end time t=seg_times[i]
                if i == 0: # First path segment
                #     # 1. Implement the initial constraints here for the first segment using A_0
                #     # 2. Implement the final position and the continuity constraints for velocity, acceleration, snap and jerk at the end of the first segment here using A_0 and A_f (check hints in the exercise description)
                    A[row, i*6:(i+1)*6] = A_0[0] #Initial position constraint
                    b[row] = pos_0
                    row += 1
                    A[row, i*6:(i+1)*6] = A_f[0] #Final position constraint
                    b[row] = pos_f
                    row += 1
                    A[row, i*6:(i+1)*6] = A_0[1] #Initial velocity constraint
                    b[row] = v_0
                    row += 1
                    A[row, i*6:(i+1)*6] = A_0[2] #Initial acceleration constraint
                    b[row] = a_0
                    row += 1
                    #Continuity of velocity, acceleration, jerk, snap
                    A[row:row+4, i*6:(i+1)*6] = A_f[1:]
                    A[row:row+4, (i+1)*6:(i+2)*6] = -A_0[1:]
                    b[row:row+4] = np.zeros(4)
                    row += 4
                elif i < m-2: # Intermediate path segments
                #     # 1. Similarly, implement the initial and final position constraints here for each intermediate path segment
                #     # 2. Similarly, implement the end of the continuity constraints for velocity, acceleration, snap and jerk at the end of each intermediate segment here using A_0 and A_f
                    A[row, i*6:(i+1)*6] = A_0[0] #Initial position constraint
                    b[row] = pos_0
                    row += 1
                    A[row, i*6:(i+1)*6] = A_f[0] #Final position constraint
                    b[row] = pos_f
                    row += 1
                    #Continuity of velocity, acceleration, jerk and snap
                    A[row:row+4, i*6:(i+1)*6] = A_f[1:]
                    A[row:row+4, (i+1)*6:(i+2)*6] = -A_0[1:]
                    b[row:row+4] = np.zeros(4)
                    row += 4
                elif i == m-2: #Final path segment
                #     # 1. Implement the initial and final position, velocity and accelerations constraints here for the final path segment using A_0 and A_f
                    A[row, i*6:(i+1)*6] = A_0[0] #Initial position constraint
                    b[row] = pos_0
                    row += 1
                    A[row, i*6:(i+1)*6] = A_f[0] #Final position constraint
                    b[row] = pos_f
                    row += 1
                    A[row, i*6:(i+1)*6] = A_f[1] #Final velocity constraint
                    b[row] = v_f
                    row += 1
                    A[row, i*6:(i+1)*6] = A_f[2] #Final acceleration constraint
                    b[row] = a_f
                    row += 1
            # Solve for the polynomial coefficients for the dimension dim

            poly_coeffs[:,dim] = np.linalg.solve(A, b)   

        return poly_coeffs

    def poly_setpoint_extraction(self, poly_coeffs, obs, path_waypoints):

        # DO NOT MODIFY --------------------------------------------------------------------------------------- ##

        # Uses the class features: self.disc_steps, self.times, self.poly_coeffs, self.vel_lim, self.acc_lim
        x_vals, y_vals, z_vals = np.zeros((self.disc_steps*len(self.times),1)), np.zeros((self.disc_steps*len(self.times),1)), np.zeros((self.disc_steps*len(self.times),1))
        v_x_vals, v_y_vals, v_z_vals = np.zeros((self.disc_steps*len(self.times),1)), np.zeros((self.disc_steps*len(self.times),1)), np.zeros((self.disc_steps*len(self.times),1))
        a_x_vals, a_y_vals, a_z_vals = np.zeros((self.disc_steps*len(self.times),1)), np.zeros((self.disc_steps*len(self.times),1)), np.zeros((self.disc_steps*len(self.times),1))

        # Define the time reference in self.disc_steps number of segements
        time_setpoints = np.linspace(self.times[0], self.times[-1], self.disc_steps*len(self.times))  # Fine time intervals

        # Extract the x,y and z direction polynomial coefficient vectors
        coeff_x = poly_coeffs[:,0]
        coeff_y = poly_coeffs[:,1]
        coeff_z = poly_coeffs[:,2]

        for i,t in enumerate(time_setpoints):
            seg_idx = min(max(np.searchsorted(self.times, t)-1,0), len(coeff_x) - 1)
            # Determine the x,y and z position reference points at every refernce time
            x_vals[i,:] = np.dot(self.compute_poly_matrix(t-self.times[seg_idx])[0],coeff_x[seg_idx*6:(seg_idx+1)*6])
            y_vals[i,:] = np.dot(self.compute_poly_matrix(t-self.times[seg_idx])[0],coeff_y[seg_idx*6:(seg_idx+1)*6])
            z_vals[i,:] = np.dot(self.compute_poly_matrix(t-self.times[seg_idx])[0],coeff_z[seg_idx*6:(seg_idx+1)*6])
            # Determine the x,y and z velocities at every reference time
            v_x_vals[i,:] = np.dot(self.compute_poly_matrix(t-self.times[seg_idx])[1],coeff_x[seg_idx*6:(seg_idx+1)*6])
            v_y_vals[i,:] = np.dot(self.compute_poly_matrix(t-self.times[seg_idx])[1],coeff_y[seg_idx*6:(seg_idx+1)*6])
            v_z_vals[i,:] = np.dot(self.compute_poly_matrix(t-self.times[seg_idx])[1],coeff_z[seg_idx*6:(seg_idx+1)*6])
            # Determine the x,y and z accelerations at every reference time
            a_x_vals[i,:] = np.dot(self.compute_poly_matrix(t-self.times[seg_idx])[2],coeff_x[seg_idx*6:(seg_idx+1)*6])
            a_y_vals[i,:] = np.dot(self.compute_poly_matrix(t-self.times[seg_idx])[2],coeff_y[seg_idx*6:(seg_idx+1)*6])
            a_z_vals[i,:] = np.dot(self.compute_poly_matrix(t-self.times[seg_idx])[2],coeff_z[seg_idx*6:(seg_idx+1)*6])

        yaw_vals = np.zeros((self.disc_steps*len(self.times),1))
        trajectory_setpoints = np.hstack((x_vals, y_vals, z_vals, yaw_vals))

        # self.plot(obs, path_waypoints, trajectory_setpoints)
            
        # Find the maximum absolute velocity during the segment
        vel_max = np.max(np.sqrt(v_x_vals**2 + v_y_vals**2 + v_z_vals**2))
        vel_mean = np.mean(np.sqrt(v_x_vals**2 + v_y_vals**2 + v_z_vals**2))
        acc_max = np.max(np.sqrt(a_x_vals**2 + a_y_vals**2 + a_z_vals**2))
        acc_mean = np.mean(np.sqrt(a_x_vals**2 + a_y_vals**2 + a_z_vals**2))

        print("Maximum flight speed: " + str(vel_max))
        print("Average flight speed: " + str(vel_mean))
        print("Average flight acceleration: " + str(acc_mean))
        print("Maximum flight acceleration: " + str(acc_max))
        
        # Check that it is less than an upper limit velocity v_lim
        assert vel_max <= self.vel_lim, "The drone velocity exceeds the limit velocity : " + str(vel_max) + " m/s"
        assert acc_max <= self.acc_lim, "The drone acceleration exceeds the limit acceleration : " + str(acc_max) + " m/s²"

        # ---------------------------------------------------------------------------------------------------- ##

        return trajectory_setpoints, time_setpoints
    
    def plot_obstacle(self, ax, x, y, z, dx, dy, dz, color='gray', alpha=0.3):
        """Plot a rectangular cuboid (obstacle) in 3D space."""
        vertices = np.array([[x, y, z], [x+dx, y, z], [x+dx, y+dy, z], [x, y+dy, z],
                            [x, y, z+dz], [x+dx, y, z+dz], [x+dx, y+dy, z+dz], [x, y+dy, z+dz]])
        
        faces = [[vertices[j] for j in [0, 1, 2, 3]], [vertices[j] for j in [4, 5, 6, 7]], 
                [vertices[j] for j in [0, 1, 5, 4]], [vertices[j] for j in [2, 3, 7, 6]], 
                [vertices[j] for j in [0, 3, 7, 4]], [vertices[j] for j in [1, 2, 6, 5]]]
        
        ax.add_collection3d(Poly3DCollection(faces, color=color, alpha=alpha))
    
    def plot(self, obs, path_waypoints, trajectory_setpoints):

        # Plot 3D trajectory
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        for ob in obs:
            self.plot_obstacle(ax, ob[0], ob[1], ob[2], ob[3], ob[4], ob[5])

        ax.plot(trajectory_setpoints[:,0], trajectory_setpoints[:,1], trajectory_setpoints[:,2], label="Minimum-Jerk Trajectory", linewidth=2)
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 3)
        ax.set_zlim(0, 1.5)

        # Plot waypoints
        waypoints_x = [p[0] for p in path_waypoints]
        waypoints_y = [p[1] for p in path_waypoints]
        waypoints_z = [p[2] for p in path_waypoints]
        ax.scatter(waypoints_x, waypoints_y, waypoints_z, color='red', marker='o', label="Waypoints")

        # Labels and legend
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_zlabel("Z Position")
        ax.set_title("3D Motion planning trajectories")
        ax.legend()
        plt.show()

