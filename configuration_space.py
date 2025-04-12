import numpy as np
import matplotlib.pyplot as plt

# === Image dimensions from original image ===
width, height = 1570, 902
image = np.ones((height, width, 3), dtype=np.uint8) * 255  # white background

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

def draw_rectangle(image, x, y, w, h, color=(0, 0, 0)):
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

ax.imshow(image)

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

# === Convert RGB image to binary mask ===
# Mark black pixels (0,0,0) as obstacles
obstacle_mask = np.all(image != [0, 0, 0], axis=2).astype(np.uint8)

# === Arm parameters ===
L1, L2 = 3.75, 2.5  # in inches

# === Field bounds based on image ===
height, width = image.shape[:2]

def line_pixels(p0, p1):
    """
    Return list of pixels along a line from p0 to p1 using Bresenham's algorithm.
    Inputs are (x, y) integer pixel coordinates.
    """
    x0, y0 = p0
    x1, y1 = p1

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0

    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1

    pixels = []

    if dx > dy:
        err = dx / 2.0
        while x != x1:
            pixels.append((y, x))  # (row, col) for NumPy indexing
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            pixels.append((y, x))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy

    pixels.append((y1, x1))  # add final point
    return pixels

def is_in_bounds(p):
    return 0 <= p[0] < height and 0 <= p[1] < width

def check_collision(pixels):
    return any(is_in_bounds(p) and obstacle_mask[p[0], p[1]] for p in pixels)

# === Discretize θ₁, θ₂ ===
theta_res = 2

# === Updated theta ranges ===
theta1_vals = np.arange(0, 181, theta_res)       # θ₁ from 0° to 180°
theta2_vals = np.arange(-180, 181, theta_res)    # θ₂ from -180° to 180°

cspace = np.zeros((len(theta1_vals), len(theta2_vals)), dtype=np.uint8)

# === Generate C-space ===
for i, theta1 in enumerate(theta1_vals):
    for j, theta2 in enumerate(theta2_vals):
        t1 = np.radians(theta1)
        t2 = np.radians(theta2)
        x1 = L1 * np.cos(t1)
        y1 = L1 * np.sin(t1)
        x2 = x1 + L2 * np.cos(t1 + t2)
        y2 = y1 + L2 * np.sin(t1 + t2)

        # Convert end effector to pixel
        tip_px, tip_py = to_pixel(x2, y2)

        # Safe check: in bounds and not inside an obstacle
        if 0 <= tip_py < height and 0 <= tip_px < width:
            if obstacle_mask[tip_py, tip_px]:
                cspace[i, j] = 1  # Collision
        else:
            cspace[i, j] = 1  # Out of bounds = collision

# === Show C-space ===
plt.imshow(cspace, cmap='gray', origin='lower',
           extent=(-180, 180, 0, 180))  # for labeled axes
plt.xlabel("Theta2 (degrees)")
plt.ylabel("Theta1 (degrees)")
plt.title("C-space from Workspace Image")
plt.show()

# # Optional: Save
# np.save("workspace_field_exact_units.npy", image)
