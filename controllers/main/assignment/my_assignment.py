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

# global turning_rate
# turning_rate = 0.1

start_flag = False
buffer_radius = 0.5
threshold = 0.5

def get_command(sensor_data, camera_data, dt):

    # NOTE: Displaying the camera image with cv2.imshow() will throw an error because GUI operations should be performed in the main thread.
    # If you want to display the camera image you can call it main.py.

    # Take off example
    if sensor_data['z_global'] < 0.49:
        control_command = [sensor_data['x_global'], sensor_data['y_global'], 1.0, sensor_data['yaw']]
        return control_command

    # ---- YOUR CODE HERE ----
    global start_flag
    global buffer_radius
    global threshold

    if start_flag == False:
        start_pose = np.array([sensor_data['x_global'], sensor_data['y_global'], 1.0, sensor_data['yaw'] ])   # start pose, but I changed height to 1 so that it doesnt land
        start_flag = True

    current_pose = np.array([sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw'] ])

     # Skip detection if all gates are logged
    if len(gates) >= 5:
        go_to_goal_pose(start_pose)
        return

    found_pink = find_pink(camera_data)

    if found_pink != None:
        # print(f"Result: {found_pink}")
        pink_x = found_pink[0]
        pink_y = found_pink[1]

        # print(f"x is {pink_x}, y is {pink_y}")
        height_error = 150-pink_y
        HK_p = 0.004
        change_h = height_error*HK_p

        orientation_error = 150-pink_x
        YK_p = 0.001
        change_yaw = orientation_error*YK_p

        # Forward in local frame
        local_cmd = np.array([0.2, 0, 0])
        quaternion = np.array([sensor_data['q_x'], sensor_data['q_y'], sensor_data['q_z'], sensor_data['q_w'] ])
        cmd = local_to_global_vector(local_cmd, quaternion)
        
        # print("cmd is: ", cmd)
        detect_gate_and_log_position(camera_data, current_pose, threshold)

        control_command = [sensor_data['x_global']+cmd[0], sensor_data['y_global']+cmd[1], sensor_data['z_global']+change_h, sensor_data['yaw']+change_yaw]
    else:
        control_command = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']+0.1]

    

    return control_command # Ordered as array with: [pos_x_cmd, pos_y_cmd, pos_z_cmd, yaw_cmd] in meters and radians

def is_far_from_existing_gates(position, gates, radius):
    """
    Checks if position is at least `radius` away from all gates.
    """
    for gate in gates:
        distance = np.linalg.norm(np.array(position) - np.array(gate))
        if distance < radius:
            return False
    return True

def go_to_goal_pose(pose):
    """
    Placeholder for your drone's navigation command to go to a specific pose.
    """
    print(f" Reached 5 gates! Navigating to final pose at {pose}")
    # Replace this with your drone's actual move command
    control_command = pose
    pass

def detect_gate_and_log_position(image, position, threshold=0.9):
    """
    Detects pink gate and logs global position if new.
    After 5 gates are logged, navigates to goal pose.
    
    Args:
        image: current camera frame (BGR)
        position: current drone global position [x, y, z]
        threshold: pink coverage ratio for detection
    """

    # Convert image and create pink mask
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_pink = np.array([140, 50, 50])
    upper_pink = np.array([170, 255, 255])
    mask = cv2.inRange(hsv, lower_pink, upper_pink)

    # Calculate pink pixel ratio
    pink_ratio = np.count_nonzero(mask) / mask.size

    # Check if pink dominates and if we're in a new area
    if pink_ratio > threshold and is_far_from_existing_gates(position, gates, buffer_radius):
        print(f"Gate detected and logged at position: {position}")
        gates.append(position.copy())

#### MY FUNCTIONS

# def find_pink_center(frame):
#     # Convert to HSV color space
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
#     # Define lower and upper bounds for pink color in HSV
#     lower_pink = np.array([140, 50, 50])   # Lower bound
#     upper_pink = np.array([170, 255, 255]) # Upper bound
    
#     # Create mask to filter pink color
#     mask = cv2.inRange(hsv, lower_pink, upper_pink)
    
#     # Find contours in the mask
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     if contours:
#         # Find the largest contour
#         largest_contour = max(contours, key=cv2.contourArea)
        
#         # Get the center of the pink area
#         M = cv2.moments(largest_contour)
#         if M["m00"] != 0:
#             cx = int(M["m10"] / M["m00"])
#             cy = int(M["m01"] / M["m00"])
#             return (cx, cy)
    
#     return None

# def camera_feed():
#     cap = cv2.VideoCapture(0)
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # Apply the pink filter in real-time
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         lower_pink = np.array([140, 50, 50])
#         upper_pink = np.array([170, 255, 255])
#         mask = cv2.inRange(hsv, lower_pink, upper_pink)
#         filtered = cv2.bitwise_and(frame, frame, mask=mask)
        
#         cv2.imshow("Filtered Feed", filtered)
        
#         key = cv2.waitKey(1) & 0xFF
        
#         if key == ord(' '):  # Capture image on spacebar press
#             center = find_pink_center(frame)
#             if center:
#                 print(f"Center of pink area: {center}")
#             else:
#                 print("No pink area detected")
        
#         elif key == ord('q'):  # Quit on 'q' press
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()

# def filter_pink(image_data):
#     # Load the image
#     image = image_data
    
#     # Convert to HSV color space
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
#     # Define lower and upper bounds for pink color in HSV
#     lower_pink = np.array([140, 50, 50])   # Lower bound
#     upper_pink = np.array([170, 255, 255]) # Upper bound
    
#     # Create mask to filter pink color
#     mask = cv2.inRange(hsv, lower_pink, upper_pink)
    
#     # Count the number of pink pixels
#     pink_pixel_count = np.count_nonzero(mask)
    
#     return pink_pixel_count

def find_pink(image_data):
    # Convert to HSV color space
    hsv = cv2.cvtColor(image_data, cv2.COLOR_BGR2HSV)
    
    # Define lower and upper bounds for pink color in HSV
    lower_pink = np.array([140, 50, 50])   # Lower bound
    upper_pink = np.array([170, 255, 255]) # Upper bound
    
    # Create mask to filter pink color
    mask = cv2.inRange(hsv, lower_pink, upper_pink)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Calculate moments to find centroid
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
    """
    Converts quaternion [x, y, z, w] to a 3x3 rotation matrix.
    """
    x, y, z, w = q
    # Normalize to be safe
    norm = np.sqrt(x*x + y*y + z*z + w*w)
    x /= norm
    y /= norm
    z /= norm
    w /= norm

    # Compute rotation matrix elements
    R = np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

    return R

def local_to_global_vector(local_vector, quaternion):
    """
    Transforms a local vector to global frame using manual quaternion rotation.
    Args:
        local_vector: [x, y, z] in drone's frame
        quaternion: [x, y, z, w] drone orientation in global frame
    Returns:
        global_vector: [x, y, z] in world frame
    """
    R = quat_to_rot_matrix(quaternion)
    global_vector = np.dot(R, local_vector)
    return global_vector


# def filter_pink_and_show(image_data):
#     # Load the image
#     # image = cv2.imread(image_path)
#     # print("ehllo")
#     image = image_data
#     # cv2.imshow("Original Image", image)
    
#     # Convert to HSV color space
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
#     # Define lower and upper bounds for pink color in HSV
#     lower_pink = np.array([140, 50, 50])   # Lower bound
#     upper_pink = np.array([170, 255, 255]) # Upper bound
    
#     # Create mask to filter pink color
#     mask = cv2.inRange(hsv, lower_pink, upper_pink)
    
#     # Apply mask to original image
#     result = cv2.bitwise_and(image, image, mask=mask)
    
#     # Show the filtered image
#     # cv2.imshow("Filtered Image (Pink)", result)
    
#     # Wait for a key press and close windows
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()