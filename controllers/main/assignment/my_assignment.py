import numpy as np, cv2

# ───────── constants
SEARCH_YAW            = 0.35
ALIGN_KP, ALIGN_TOL   = 0.4, 8
SHIFT_D, SHIFT_SIGN   = 0.50, -1.0          # body-right 0.5 m
SETTLE_SEC, V_THR     = 1.2, 0.05           # wait until drone is steady
K_DEF = np.array([[161.014, 0., 150.],
                  [  0.   ,161.014,150.],
                  [  0.   ,  0.  ,  1. ]]) # 300×300, FOV≈1.5 rad
R_cb = np.array([[ 0, -1,  0],   # camera→body rotation (appendix)
                 [ 0,  0, -1],
                 [ 1,  0,  0]])
t_cb = np.array([0.03, 0, 0.01])            # camera offset in body (m)
HSV_LO, HSV_HI       = (120, 80, 50), (160, 255, 255)

# ───────── FSM globals
state = 'TAKEOFF'
p1 = p2 = gate_xyz = None
x0 = y0 = None
t_wait = 0.0
gate_ref_x = None
gate_ref_area = None

# ============================================================= main
def get_command(s, cam, dt):
    global state, p1, p2, gate_xyz, x0, y0, t_wait, gate_ref_x, gate_ref_area

    if x0 is None:                     # remember take-off spot
        x0, y0 = s['x_global'], s['y_global']

    img, K = (cam.get('image'), cam.get('K', K_DEF)) if isinstance(cam, dict) else (cam, K_DEF)

    # ---------- TAKEOFF ----------
    if state == 'TAKEOFF':
        if s['z_global'] < .95:
            return [x0, y0, 1.0, s['yaw']]
        state = 'SEARCH'; print("search")
    # ---------- SEARCH ----------
    if state == 'SEARCH':
        c, mask, area = pick_gate(img)          # no reference yet
        if c is None:
            return [x0, y0, 1.0, s['yaw'] + SEARCH_YAW*dt]
        state = 'ALIGN'; print("gate")
    # ---------- ALIGN ----------
    if state == 'ALIGN':
        c, mask, area = pick_gate(img)
        if c is None:            # lost view
            state = 'SEARCH'; print("lost")
            return [x0, y0, 1.0, s['yaw']]
        err_x = c.mean(0)[0] - 150.
        if abs(err_x) > ALIGN_TOL:
            return [x0, y0, 1.0, s['yaw'] - ALIGN_KP*err_x/150.]
        gate_ref_x, gate_ref_area = c.mean(0)[0], area
        p1 = (c, cam_pose(s))
        cv2.imwrite("view1_mask.png", mask)
        state, t_wait = 'SHIFT', 0.0; print("shift")
    # ---------- SHIFT ----------
    if state == 'SHIFT':
        dy =  SHIFT_SIGN*SHIFT_D*np.cos(s['yaw'])
        dx = -SHIFT_SIGN*SHIFT_D*np.sin(s['yaw'])
        tx, ty = p1[1][:3,3][:2] + np.array([dx, dy])
        if np.hypot(s['x_global']-tx, s['y_global']-ty) < 0.05:
            state, t_wait = 'SETTLE', 0.0; print("settle")
        return [tx, ty, 1.0, s['yaw']]
    # ---------- SETTLE ----------
    if state == 'SETTLE':
        t_wait += dt
        if t_wait < SETTLE_SEC or np.hypot(s['v_x'], s['v_y']) > V_THR:
            return [s['x_global'], s['y_global'], 1.0, s['yaw']]
        state = 'OBS2'
    # ---------- OBS2 ----------
    if state == 'OBS2':
        c, mask, _ = pick_gate(img, gate_ref_x, gate_ref_area)  # same gate
        if c is None:
            return [s['x_global'], s['y_global'], 1.0, s['yaw']]
        p2 = (c, cam_pose(s))
        cv2.imwrite("view2_mask.png", mask)
        debug_overlay(img, c)
        gate_xyz = triang(p1, p2, K)
        print(f"goto {gate_xyz}")
        state = 'NAV'
    # ---------- NAVIGATE ----------
    if state == 'NAV':
        gx, gy, gz = gate_xyz
        if np.hypot(s['x_global']-gx, s['y_global']-gy) < 0.1:
            return [gx, gy, gz, s['yaw']]
        return [gx, gy, gz, s['yaw']]

    # fallback hover
    return [s['x_global'], s['y_global'], 1.0, s['yaw']]

# ============================================================= helpers
def sort4(pts):
    c = pts.mean(0)
    return pts[np.argsort(np.arctan2(pts[:,1]-c[1], pts[:,0]-c[0]))]

def pick_gate(img, ref_x=None, ref_area=None):
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LO, HSV_HI)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best, best_cost = None, 1e9
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 400: continue
        p = cv2.approxPolyDP(cnt, .02*cv2.arcLength(cnt, True), True)
        if len(p) != 4: continue
        p = sort4(np.squeeze(p, 1).astype(float))
        cx = p[:,0].mean()
        if ref_x is None:                       # first view → prefer right-most
            cost = -cx
        else:                                   # second view → match centroid & area
            cost  = 0.4*abs(cx - ref_x)
            cost += 0.6*abs(area - ref_area)/ref_area
        if cost < best_cost:
            best, best_cost, best_area = p, cost, area
    return (best, mask, best_area) if best is not None else (None, mask, None)

def cam_pose(s):
    cr,sr,cp,sp,cy,sy = np.cos(s['roll']), np.sin(s['roll']), \
                        np.cos(s['pitch']), np.sin(s['pitch']), \
                        np.cos(s['yaw']),   np.sin(s['yaw'])
    R_wb = np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                     [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                     [  -sp,            cp*sr,            cp*cr]])
    R_wc = R_wb @ R_cb
    t_wc = np.array([s['x_global'], s['y_global'], s['z_global']]) + R_wb @ t_cb
    T = np.eye(4);  T[:3,:3], T[:3,3] = R_wc, t_wc
    return T

# ----------------------------------------------------------------------
# Appendix-style triangulation: midpoint of shortest segment
# ----------------------------------------------------------------------
def triang(p1, p2, K):
    """
    Re-implements the method in the PDF appendix:

    1. build bearing vectors  v  and  v'
       – pixel → camera ⇒ (u-cx , v-cy , f_pix)
    2. rotate to world frame:   r = R_wc · v
                                s = R_wc'· v'
    3. camera centres in world: P , Q
    4. solve   P + λ r  ≈  Q + μ s   (least-squares)
       λ, μ = pseudoinverse([r  –s]) · (Q−P)
    5. use midpoint H = ½ (F + G)    with F = P+λr , G = Q+μs
    6. return average of the four midpoints  → gate centre
    """
    corners1, T1 = p1            # four (u,v) pixels , 4×4 world←cam
    corners2, T2 = p2
    cx, cy, f = K[0,2], K[1,2], K[0,0]      # principal point, focal [px]

    # camera→world rotations and positions
    R1, t1 = T1[:3,:3], T1[:3,3]
    R2, t2 = T2[:3,:3], T2[:3,3]

    mids_world = []

    for (u1,v1), (u2,v2) in zip(corners1, corners2):
        # --- step 1 : bearing vectors in camera frames
        # v_cam  = np.array([u1-cx, cy-v1, f])
        v_cam  = np.array([u1-cx, v1-cy, f])
        # v_cam /= np.linalg.norm(v_cam)
        v_cam2 = np.array([u2-cx, v2-cy, f])
        # v_cam2 = np.array([u2-cx, cy-v2, f])
        # v_cam2/= np.linalg.norm(v_cam2)
        

        # --- step 2 : rotate to world
        r = R1 @ v_cam
        s = R2 @ v_cam2

        print(r, s)

        # --- step 3 : camera centres
        P = t1
        Q = t2

        # --- step 4 : least-squares λ and μ
        A = np.column_stack([r, -s])            # 3×2
        x, *_ = np.linalg.lstsq(A, Q-P, rcond=None)
        lam, mu = x

        # --- step 5 : points on each ray + midpoint
        F = P + lam * r
        G = Q + mu * s
        H = 0.5*(F+G)
        mids_world.append(H)

    mids_world = np.array(mids_world)           # 4×3

    # --- diagnostics (like before)
    for i,pt in enumerate(mids_world):
        print(f"corner {i}: {pt}")

    # reprojection error for the *centre*
    centre = mids_world.mean(0)
    for (R,t,uvs) in [(R1,t1,corners1), (R2,t2,corners2)]:
        proj = K @ (R.T @ (centre - t))         # world→cam→pixels (homog)
        proj = proj[:2]/proj[2]
        print("err px:", np.linalg.norm(proj - uvs.mean(0)))

    return centre


def debug_overlay(img, pts):
    dbg = img.copy()
    for p in pts: cv2.circle(dbg, tuple(p.astype(int)), 4, (0,255,0), -1)
    cv2.imwrite("debug_view2.png", dbg)
