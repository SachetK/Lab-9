import numpy as np
import matplotlib.pyplot as plt
import math
from queue import PriorityQueue

# === Image dimensions from original image ===
width, height = 1570, 902
image = np.ones((height, width), dtype=np.uint8)

# === Field dimensions in inches ===
x_min, x_max = -7, 7
y_min, y_max = 0, 8

# === Scaling ===
pixels_per_inch_x = width / (x_max - x_min)   # ≈112.14
pixels_per_inch_y = height / (y_max - y_min)  # ≈112.75

def to_pixel(x, y):
    """Convert field inches to image pixel coordinates."""
    px = int((x - x_min) * pixels_per_inch_x)
    py = int((y - y_min) * pixels_per_inch_y)
    return px, height - py

def draw_rectangle(image, x, y, w, h, color=0):
    x0, y0 = to_pixel(x, y + h)
    x1, y1 = to_pixel(x + w, y)
    image[min(y0, y1):max(y0, y1), min(x0, x1):max(x0, x1)] = color

# === Obstacles (in inches) ===
obstacles = [
    (-4, 4, 2, 2),   # Left block
    (1, 5, 2, 2),    # Right block
    (-1, 1, 1, 2),   # Bottom base
    (0, 2, 1, 1),    # Bottom top
]

for x, y, w, h in obstacles:
    draw_rectangle(image, x, y, w, h)

fig, ax = plt.subplots(figsize=(12, 6))
x_field, y_field = -1.0, 5.5  # example coordinate
px, py = to_pixel(x_field, y_field)

ax.imshow(image, cmap="gray")
ax.plot(px, py, marker='o', color='r')

# Set axis limits in inches
ax.set_xlim(0, width)
ax.set_ylim(height, 0)

# Label in inches
inch_ticks_x = np.linspace(x_min, x_max, 15)
inch_ticks_y = np.linspace(y_min, y_max, 9)
pixel_ticks_x = [(x - x_min) * pixels_per_inch_x for x in inch_ticks_x]
pixel_ticks_y = [height - (y - y_min) * pixels_per_inch_y for y in inch_ticks_y]

ax.set_xticks(pixel_ticks_x)
ax.set_xticklabels([f"{x:.0f}\"" for x in inch_ticks_x])
ax.set_yticks(pixel_ticks_y)
ax.set_yticklabels([f"{y:.0f}\"" for y in inch_ticks_y])

ax.set_xlabel("X (inches)")
ax.set_ylabel("Y (inches)")
ax.set_title("Workspace Map (Inch-Aligned, Axis On)")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Arm parameters ===
L1, L2 = 3.75, 2.5  # in inches

# === Field bounds based on image ===
height, width = image.shape[:2]

# === Discretize θ₁, θ₂ ===
theta_res = 2

# === Updated theta ranges ===
theta1_vals = np.arange(0, 181, theta_res)       # θ₁ from 0° to 180°
theta2_vals = np.arange(-180, 181, theta_res)    # θ₂ from -180° to 180°

cspace = np.ones((len(theta1_vals), len(theta2_vals)), dtype=np.uint8)

def forward_kinematics(theta1, theta2):
    t1 = np.radians(theta1)
    t2 = np.radians(theta2)
    x1 = L1 * np.cos(t1)
    y1 = L1 * np.sin(t1)
    x2 = x1 + L2 * np.cos(t1 + t2)
    y2 = y1 + L2 * np.sin(t1 + t2)

    return x1, y1, x2, y2

def inverse_kinematics(x, y, L1=3.75, L2=2.5):
    """
    Given end effector (x, y), return all valid (theta1, theta2) solutions in degrees.
    Returns a list of (theta1, theta2) tuples, or empty list if unreachable.
    """

    r2 = x ** 2 + y ** 2
    r = np.sqrt(r2)

    # Check reachability
    if r > L1 + L2 or r < abs(L1 - L2):
        return []  # unreachable

    # Law of cosines
    cos_theta2 = (r2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_theta2 = np.clip(cos_theta2, -1, 1)  # numerical safety
    theta2_1 = np.arccos(cos_theta2)
    theta2_2 = -theta2_1

    # Solve for theta1 for each theta2
    def solve_theta1(theta2):
        k1 = L1 + L2 * np.cos(theta2)
        k2 = L2 * np.sin(theta2)
        theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)
        return np.degrees(theta1), np.degrees(theta2)

    sol1 = solve_theta1(theta2_1)
    sol2 = solve_theta1(theta2_2)

    return [sol1, sol2]

# === Generate C-space ===
for i, theta1 in enumerate(theta1_vals):
    for j, theta2 in enumerate(theta2_vals):
        _, _, x2, y2 = forward_kinematics(theta1, theta2)

        # Convert end effector to pixel
        tip_px, tip_py = to_pixel(x2, y2)

        # Safe check: in bounds and not inside an obstacle
        if 0 <= tip_py < height and 0 <= tip_px < width:
            if image[tip_py, tip_px] == 0:
                cspace[i, j] = 0  # Collision

def verify_cspace(cspace):
    for i in range(cspace.shape[0]):
        for j in range(cspace.shape[1]):
            _, _, x2, y2 = forward_kinematics(theta1, theta2)

            # Convert end effector to pixel
            tip_px, tip_py = to_pixel(x2, y2)

            if 0 <= tip_py < height and 0 <= tip_px < width:
                if cspace[i, j] != image[tip_py, tip_px]:
                    return False

    return True

cspace_valid = verify_cspace(cspace)
print(cspace_valid)

# === Angle ↔ Index Conversion ===
def theta2_to_index(theta2):
    return int(round((theta2 + 180) / theta_res)) % len(theta2_vals)

def index_to_theta2(idx):
    return -180 + idx * theta_res

# === Dijkstra's Algorithm in C-space ===
def dijkstra_cspace(cspace, start, goal):
    DIRECTIONS = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
                  (-1, -1, math.sqrt(2)), (1, -1, math.sqrt(2)),
                  (-1, 1, math.sqrt(2)), (1, 1, math.sqrt(2))]
    turn_penalty = 0.8

    if start == goal:
        return []

    if cspace[start] == 0 or cspace[goal] == 0:
        raise Exception("Path cannot exist! Endpoints invalid!")

    height, width = cspace.shape

    pq = PriorityQueue()
    pq.put((0, start, None))  # Priority queue of cost, node, and last direction

    distances = {start: 0}  # Store distances with explicit weight tracking
    bp = {start: None}

    while not pq.empty():
        distance, (r, c), last_direction = pq.get()

        if (r, c) == goal:
            break

        for dr, dc, move_cost in DIRECTIONS:
            nr, nc = r + dr, c + dc
            new_direction = (dr, dc)

            if 0 <= nr < height and 0 <= nc < width and cspace[nr, nc] == 1:
                if last_direction is not None and new_direction != last_direction:
                    move_cost += turn_penalty  # Apply turn penalty when changing direction

                new_distance = distance + move_cost

                if (nr, nc) not in distances or new_distance < distances[(nr, nc)]:
                    distances[(nr, nc)] = new_distance
                    bp[(nr, nc)] = (r, c)  # Store path
                    pq.put((new_distance, (nr, nc), new_direction))

    print("Finished path!")
    # Reconstruct path
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = bp[current]

    # Add start to path and reverse to get correct order
    path.reverse()

    for r, c in path:
        assert cspace[r, c] == 1, f"Obstacle in path at {(r, c)}"

    return path if path[0] == start else None

# === Define Start and Goal Positions in Field Space ===
start_xy = (6.25, 0)
goal_xy = (-1.0, 6)

# === Inverse kinematics (choose first solution for simplicity) ===
start_ik = inverse_kinematics(*start_xy)[0]
goal1_ik = inverse_kinematics(*goal_xy)[0]
goal2_ik = inverse_kinematics(*goal_xy)[1]

# === Convert angles to C-space indices ===
theta1_s = int(round(start_ik[0] / theta_res))
theta2_s = theta2_to_index(start_ik[1])

theta1_g1 = int(round(goal1_ik[0] / theta_res))
theta2_g1 = theta2_to_index(goal1_ik[1])

theta1_g2 = int(round(goal2_ik[0] / theta_res))
theta2_g2 = theta2_to_index(goal2_ik[1])

start_idx = (theta1_s, theta2_s)
goal1_idx = (theta1_g1, theta2_g1)
goal2_idx = (theta1_g2, theta2_g2)

# === Print debug info ===
print("Start IK:", start_ik)
print("Goal IK (First Solution):", goal1_ik)
print("Goal IK (Second Solution):", goal2_ik)

print("Start FK:", forward_kinematics(*start_ik))
print("Goal FK:", forward_kinematics(*goal1_ik))

print("Start index:", start_idx)
print("Goal Solution 1 index:", goal1_idx)
print("Goal Solution 2 index:", goal2_idx)

# === C-space validation ===
if cspace[start_idx] == 0:
    print("⚠️ Start configuration is in collision.")
if cspace[goal1_idx] == 0:
    print("⚠️ Goal configuration is in collision.")
if not (0 <= theta1_s < cspace.shape[0] and 0 <= theta2_s < cspace.shape[1]):
    print("⚠️ Start index out of bounds!")
if not ((0 <= theta1_g1 < cspace.shape[0] and 0 <= theta2_g1 < cspace.shape[1]) or
        (0 <= theta1_g2 < cspace.shape[0] and 0 <= theta2_g2 < cspace.shape[1])):
    print("⚠️ Goal index out of bounds!")

def return_minimal_path(cspace, start, goal_first, goal_second):
    path1 = dijkstra_cspace(cspace, start, goal_first) if cspace[goal_first] != 0 else None
    path2 = dijkstra_cspace(cspace, start, goal_second) if cspace[goal_second] != 0 else None

    if path1 is None:
        return path2

    if path2 is None:
        return path1

    return path1

# === Run Dijkstra ===
print(cspace)
path = return_minimal_path(cspace, start_idx, goal1_idx, goal2_idx)
print(path)

# === Plot the C-space and the path ===
plt.figure(figsize=(10, 6))
plt.imshow(cspace, cmap='gray', origin='lower',
           extent=(-180, 180, 0, 180))  # extent sets the axis labels correctly
plt.plot(start_ik[1], start_ik[0], marker='o', color='red')
plt.plot(goal1_ik[1], goal1_ik[0], marker='o', color='blue')
plt.xlabel("Theta2 (degrees)")
plt.ylabel("Theta1 (degrees)")
plt.title("C-space Map")

if path:
    path_t2 = [index_to_theta2(j) for (i, j) in path]
    path_t1 = [i * theta_res for (i, j) in path]
    plt.plot(path_t2, path_t1, color='red', linewidth=2, label='Path')
    plt.show()
else:
    print("⚠️ No path found — double-check start/goal collision or indexing.")