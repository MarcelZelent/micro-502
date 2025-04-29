import numpy as np, cv2

# ───────── camera model
K_DEF = np.array([[161.014, 0., 150.],
                  [0., 161.014, 150.],
                  [0.,   0.   ,  1. ]])
R_cb  = np.array([[0, 0, 1],   # camera→body
                  [-1,0,0],
                  [0,-1,0]])
t_cb  = np.array([0.03, 0.0, 0.01])

HSV_LO, HSV_HI = (120, 80, 50), (160, 255, 255)

# ───────── behaviour
SEARCH_YAW = 5.0              # rad/s
ALIGN_KP, ALIGN_TOL = 0.4, 8
SHIFT_D   = 0.50              # m right for stereo baseline
SETTLE_SEC, V_THR = 1.2, 0.05
MIN_AREA_PX = 1000            # else step forward
STEP_FWD    = 0.5
SEG_HALF    = 0.5             # half-length of “through” segment
MAX_GATES   = 5

# ───────── FSM globals
state='TAKEOFF'; x0=y0=z0=None
gate_cnt=0; t_wait=0.0
p1=p2=None
gate_xyz=gate_yaw=None
entry_tgt=thru_tgt=None
gate_ref_x=gate_ref_area=None
RETURN_ALT = 1.0              # cruise height on way home
TOTAL_LAPS = 3          # 1 (current) + 2 extra
lap_cnt    = 0          # starts at 0
traj = None

# Global planner instance and trajectory
planner = None
current_traj = None
traj_idx = 0
gates = []


# ============================================================= MAIN
def get_command(s, cam, dt):
    global state,x0,y0,z0,t_wait,gate_cnt
    global p1,p2,gate_xyz,gate_yaw,entry_tgt,thru_tgt
    global gate_ref_x,gate_ref_area
    global gates, lap_cnt, traj

    if x0 is None:
        x0,y0,z0 = s['x_global'], s['y_global'], s["z_global"]

    img,K = (cam.get('image'), cam.get('K',K_DEF)) if isinstance(cam,dict) else (cam,K_DEF)

    # ---------- TAKEOFF ----------
    if state=='TAKEOFF':
        if s['z_global']<0.95:
            return [x0,y0,1.0,s['yaw']]
        state='SEARCH'; print("search")

    # ---------- SEARCH ----------
    if state=='SEARCH':
        c,_,area = pick_gate(img)
        if c is None:
            return [x0,y0,1.0, s['yaw']+SEARCH_YAW*dt]
        if area<MIN_AREA_PX:                                   # probe
            fx,fy,_ = forward_pt(s, STEP_FWD)
            return [fx,fy,1.0,s['yaw']]
        state='ALIGN'

    # ---------- ALIGN ----------
    if state=='ALIGN':
        c,_,area = pick_gate(img)
        if c is None:
            state='SEARCH'; return [x0,y0,1.0,s['yaw']]
        err = c.mean(0)[0]-150.
        if abs(err)>ALIGN_TOL:
            return [x0,y0,1.0, s['yaw']-ALIGN_KP*err/150.]
        gate_ref_x,gate_ref_area = c.mean(0)[0], area
        p1 = (c, cam_pose(s)); state,t_wait='SHIFT',0.0

    # ---------- SHIFT (body-right) ----------
    if state=='SHIFT':
        dx,dy = right_vec(s)*SHIFT_D
        tx,ty = p1[1][:3,3][:2] + np.array([dx,dy])
        if np.hypot(s['x_global']-tx, s['y_global']-ty)<0.05:
            state='SETTLE'; t_wait=0.0
        return [tx,ty,1.0,s['yaw']]

    # ---------- SETTLE ----------
    if state=='SETTLE':
        t_wait += dt
        if t_wait<SETTLE_SEC or np.hypot(s['v_x'],s['v_y'])>V_THR:
            return hover(s)
        state='OBS2'

    # ---------- OBS2 ----------
    if state=='OBS2':
        c,_,_ = pick_gate(img, gate_ref_x, gate_ref_area)
        if c is None: return hover(s)
        p2 = (c, cam_pose(s))
        gate_xyz, gate_yaw = triang_and_yaw(p1,p2,K,s)
        gates.append(gate_xyz)
        print("gates: ", gates)
        n = np.array([np.cos(gate_yaw), np.sin(gate_yaw), 0.])
        entry_tgt = gate_xyz - SEG_HALF*n
        thru_tgt  = gate_xyz + SEG_HALF*n
        state='ENTRY'; print("segment planned")

    # ---------- ENTRY ----------
    if state=='ENTRY':
        if dist_xy(s,entry_tgt)<0.1:
            state='THROUGH'
        return [*entry_tgt, entry_tgt[2], gate_yaw]

    # ---------- THROUGH ----------
    if state=='THROUGH':
        if dist_xy(s,thru_tgt)<0.1:
            gate_cnt += 1
            if gate_cnt >= MAX_GATES:       # all frames done  → go home
                state = 'RETURN'
                print("returning to launch")
            else:
                freeze_xy(s)
                state = 'SEARCH'

        return [*thru_tgt, thru_tgt[2], gate_yaw]

    # ---------- RETURN ----------
    if state == 'RETURN':
        if np.hypot(s['x_global']-x0, s['y_global']-y0) < 0.10:
            state = 'STOP'
        if len(gates) == 5:
            state = "HOTLAP"
            lap_cnt = 0
        return [1.0, 4.0, RETURN_ALT, 0.0]    # keep last yaw while cruising

    # ---------- HOTLAP ----------
    if state == 'HOTLAP':
        if lap_cnt == 0:
            traj = init_planner(gates)
            print("traj: ", traj)

    # ---------- STOP ----------
    if state=='STOP':
        return hover(s)

    return hover(s)

def init_planner(waypoints):
    """Initialize motion planner with waypoints"""

    # ------------------------------------------------ mock inputs
    gate_centres = waypoints

    launch = np.array([1.0, 4.0, 1.0])          # where the drone takes off
    bounds = [0, 10, 0, 10, 0, 5]                 # [xmin,xmax, ymin,ymax, zmin,zmax]
    grid   = 0.25                               # coarse grid for A*
    obstacles = np.empty((0,6))                 # none in this example

    # ------------------------------------------------ build waypoint list
    waypoints = [launch] + gate_centres + [launch]   # closed loop

    # ------------------------------------------------ call the planner
    mp = MotionPlanner3D(start      = waypoints[0],
                        obstacles  = obstacles,
                        bounds     = bounds,
                        grid_size  = grid,
                        goal       = waypoints[-1])

    # make it SLOW: overwrite the default timing vector
    total_time = 20.0                      # seconds for the whole loop (~1 m s-¹)
    mp.times   = np.linspace(0, total_time, len(waypoints))

    # re-compute with slow timing
    poly   = mp.compute_poly_coefficients(waypoints)
    traj, traj_t = mp.poly_setpoint_extraction(poly, obstacles, waypoints)

    # ------------------------------------------------  visual check
    print("trajectory shape:", traj.shape)   # (disc_steps*len(times), 4)
    print("first few set-points:\n", traj[:5])

    # feed traj[:,0:3] to your controller in sync with traj_t
    return traj


# ============================================================= helper math
def hover(s): return [s['x_global'],s['y_global'],1.0,s['yaw']]

def forward_pt(s,d): return (s['x_global']+d*np.cos(s['yaw']),
                             s['y_global']+d*np.sin(s['yaw']),
                             1.0)

def right_vec(s):
    """unit body-right expressed in world frame"""
    return np.array([ np.sin(s['yaw']),  -np.cos(s['yaw']) ])   # ← now ndarray


def dist_xy(s,p): return np.hypot(s['x_global']-p[0], s['y_global']-p[1])

def freeze_xy(s):
    global x0,y0,z0
    x0,y0,z0 = s['x_global'], s['y_global'], s["z_global"]

# ============================================================= vision utils
def sort4(pts):
    c=pts.mean(0); return pts[np.argsort(np.arctan2(pts[:,1]-c[1], pts[:,0]-c[0]))]

def pick_gate(img, ref_x=None, ref_area=None):
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,HSV_LO,HSV_HI)
    cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    best,best_cost=None,1e9
    for cnt in cnts:
        area=cv2.contourArea(cnt);   p=None
        if area<400: continue
        poly=cv2.approxPolyDP(cnt,.02*cv2.arcLength(cnt,True),True)
        if len(poly)==4:
            p=sort4(np.squeeze(poly,1).astype(float))
        if p is None: continue
        cx=p[:,0].mean()
        cost = -cx if ref_x is None else 0.4*abs(cx-ref_x)+0.6*abs(area-ref_area)/ref_area
        if cost<best_cost: best,best_cost,best_area=p,cost,area
    return (best,mask,best_area) if best is not None else (None,mask,0)

# ============================================================= pose & triang
def cam_pose(s):
    cr,sr=np.cos(s['roll']),np.sin(s['roll'])
    cp,sp=np.cos(s['pitch']),np.sin(s['pitch'])
    cy,sy=np.cos(s['yaw']),np.sin(s['yaw'])
    R_wb=np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                   [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                   [  -sp,            cp*sr,            cp*cr]])
    R_wc = R_wb @ R_cb
    t_wc = np.array([s['x_global'],s['y_global'],s['z_global']]) + R_wb@t_cb
    T=np.eye(4); T[:3,:3],T[:3,3]=R_wc,t_wc
    return T

def triang_and_yaw(p1,p2,K,s):
    (c1,T1),(c2,T2)=p1,p2
    cx,cy,f = K[0,2],K[1,2],K[0,0]
    R1,t1 = T1[:3,:3],T1[:3,3]
    R2,t2 = T2[:3,:3],T2[:3,3]
    pts=[]
    for (u1,v1),(u2,v2) in zip(c1,c2):
        v1c=np.array([u1-cx,v1-cy,f]); v1c/=np.linalg.norm(v1c)
        v2c=np.array([u2-cx,v2-cy,f]); v2c/=np.linalg.norm(v2c)
        r,sdir = R1@v1c, R2@v2c
        lam,mu=np.linalg.lstsq(np.c_[r,-sdir],t2-t1,rcond=None)[0]
        pts.append( 0.5*((t1+lam*r)+(t2+mu*sdir)) )
    X=np.array(pts)
    centre=X.mean(0)

    # find outward normal (cross bottom-edge with up)
    edge = X[1]-X[0]
    up   = np.array([0,0,1])
    normal = np.cross(edge, up)
    normal/=np.linalg.norm(normal)

    # ensure it points from drone toward gate
    to_gate = centre - np.array([s['x_global'],s['y_global'],s['z_global']])
    if np.dot(normal, to_gate) < 0:          # flip if wrong
        normal = -normal
    yaw = np.arctan2(normal[1], normal[0])   # standard world yaw (x forward)

    return centre, yaw




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

        self.plot(obs, path_waypoints, trajectory_setpoints)
            
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

