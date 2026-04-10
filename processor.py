import cv2
import numpy as np
import os
import math
from skimage.filters import frangi, sato
from tifffile import TiffFile
import cv2
import numpy as np
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from datetime import datetime
import os
import matplotlib.pyplot as plt
from reportlab.lib.utils import ImageReader
from reportlab.lib.pagesizes import A4
# =============================
# A* TRACE (same as yours)
# =============================
def astar_trace(start, goal, mask):
    import heapq

    h, w = mask.shape

    def heuristic(a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if abs(current[0]-goal[0]) < 2 and abs(current[1]-goal[1]) < 2:
            came_from[goal] = current
            break

        for dy in [-1,0,1]:
            for dx in [-1,0,1]:
                ny, nx = current[0]+dy, current[1]+dx

                if not (0 <= ny < h and 0 <= nx < w):
                    continue

                if mask[ny, nx] == 0:
                    continue

                neighbor = (ny, nx)
                tentative_g = g_score[current] + math.hypot(dy, dx)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
                    came_from[neighbor] = current

    if goal not in came_from:
        return []

    path = []
    curr = goal
    while curr in came_from:
        path.append(curr)
        curr = came_from[curr]

    path.append(start)
    path.reverse()

    return path


# =============================
# IMAGE PROCESSING
# =============================
def iso_histogram_threshold(image):
    hist = cv2.calcHist([image], [0], None, [256], [0,256]).flatten()
    hist_smooth = cv2.GaussianBlur(hist, (11,1), 0)

    peaks = []
    for i in range(1, 255):
        if hist_smooth[i] > hist_smooth[i-1] and hist_smooth[i] > hist_smooth[i+1]:
            peaks.append(i)

    peaks = sorted(peaks, key=lambda x: hist_smooth[x], reverse=True)

    if len(peaks) >= 2:
        low, high = sorted(peaks[:2])
        return int((low + high) / 2)
    return int(np.mean(image))


def process_image(image_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    with TiffFile(image_path) as tif:
        image = tif.asarray()

    raw = image.astype(np.uint8)

    # 🔥 crop ONCE
    raw = raw[:-40, :]
    h, w = raw.shape

    cv2.imwrite(os.path.join(output_folder, "01_cropped.png"), raw)

    # 🔥 use separate variable for processing
    orig = raw.copy()

    # =============================
    # 2. CLAHE + BLUR
    # =============================
    orig = cv2.normalize(orig, None, 0, 255, cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    orig = clahe.apply(orig)

    blur = cv2.GaussianBlur(orig, (5,5), 0)

    # =============================
    # 3. ISO THRESHOLD
    # =============================
    gauss = cv2.GaussianBlur(orig, (9,9), 0)
    th = iso_histogram_threshold(gauss)

    _, threshold_mask = cv2.threshold(gauss, th, 255, cv2.THRESH_BINARY)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(threshold_mask, 8)

    clean_mask = np.zeros_like(threshold_mask)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] > 30:
            clean_mask[labels == i] = 255

    threshold_mask = clean_mask
    kernel = np.ones((3,3), np.uint8)
    threshold_mask = cv2.morphologyEx(
        threshold_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )
    # =============================
    # 4. FRANGI + SATO
    # =============================
    blur_float = blur.astype(np.float64) / 255.0

    ridge_f = frangi(
        blur_float,
        sigmas=range(1,4),
        alpha=0.5,
        beta=0.5,
        black_ridges=False
    )

    ridge_s = sato(
        blur_float,
        sigmas=range(1,4),
        black_ridges=False
    )

    ridge = np.maximum(ridge_f, ridge_s)
    ridge = ridge * (threshold_mask / 255.0)

    ridge_norm = cv2.normalize(ridge, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    edges = (ridge_norm > 15).astype(np.uint8) * 255
    cv2.imwrite(os.path.join(output_folder, "02_edges.png"), edges)

    # =============================
    # 5. EDGE FILTERING
    # =============================
    num, lbls, stats, _ = cv2.connectedComponentsWithStats(edges, 8)

    fiber_edges = np.zeros_like(edges)
    particle_edges = np.zeros_like(edges)

    MIN_EDGE_LENGTH = min(h, w) // 8

    for i in range(1, num):
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]

        touches_border = (x <= 2 or y <= 2 or 
                          x + width >= w - 2 or 
                          y + height >= h - 2)

        area = stats[i, cv2.CC_STAT_AREA]

        if (max(width, height) >= MIN_EDGE_LENGTH or touches_border) and area > 10:
            fiber_edges[lbls == i] = 255
        else:
            particle_edges[lbls == i] = 255

    cv2.imwrite(os.path.join(output_folder, "03_fiber_edges.png"), fiber_edges)
    cv2.imwrite(os.path.join(output_folder, "particles_mask.png"), particle_edges)
    # 🔥 CLEAN PARTICLE MASK (CRITICAL FIX)
    num_p, labels_p, stats_p, _ = cv2.connectedComponentsWithStats(particle_edges, 8)

    clean_particles = np.zeros_like(particle_edges)

    for i in range(1, num_p):
        area = stats_p[i, cv2.CC_STAT_AREA]

        if area > 80:
            clean_particles[labels_p == i] = 255

    # 🔥 USE CLEANED MASK
    particle_edges = clean_particles

    # optional smoothing
    kernel = np.ones((3,3), np.uint8)
    particle_edges = cv2.morphologyEx(
        particle_edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )
    # =============================
    # 6. FINAL FIBER MASK
    # =============================
    fiber_pixels = cv2.bitwise_and(raw, raw, mask=fiber_edges)

    kernel = np.ones((3,3), np.uint8)
    fiber_pixels = cv2.morphologyEx(fiber_pixels, cv2.MORPH_CLOSE, kernel, iterations=1)

    _, binary = cv2.threshold(fiber_pixels, 10, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5,5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # remove noise
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)

    clean = np.zeros_like(binary)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] > 80:
            clean[labels == i] = 255

    binary = clean

    cv2.imwrite(os.path.join(output_folder, "04_fiber_mask.png"), binary)

    dist_map = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    print("RAW SHAPE:", raw.shape)
    print("MASK SHAPE:", binary.shape)
    return {
        "mask": binary,
        "dist_map": dist_map,
        "particle_mask": particle_edges,
        "orig": raw   # 🔥 IMPORTANT
    }

# =============================
# TRACE MULTI-POINT FIBER
# =============================
def trace_fiber(points, mask):
    full_path = []

    for i in range(len(points)-1):
        seg = astar_trace(points[i], points[i+1], mask)
        full_path.extend(seg)

    return full_path


# =============================
# COMPUTE FIBER METRICS
# =============================
def compute_fiber_metrics(path, dist_map, pixel_size=1.0):
    if len(path) < 2:
        return None

    import numpy as np
    import math

    # =========================
    # LENGTH
    # =========================
    length_px = 0
    for i in range(1, len(path)):
        y1, x1 = path[i-1]
        y2, x2 = path[i]
        length_px += math.hypot(y2-y1, x2-x1)

    length_um = length_px * pixel_size

    # =========================
    # WIDTH
    # =========================
    ys = np.array([p[0] for p in path])
    xs = np.array([p[1] for p in path])

    widths = 2 * dist_map[ys, xs] * pixel_size

    # 🔥 CRITICAL FIX
    valid_widths = widths[widths > 2]

    if len(valid_widths) == 0:
        valid_widths = widths  # fallback

    min_w = float(np.percentile(valid_widths, 10))
    max_w = float(np.percentile(valid_widths, 95))

    # =========================
    # STRAIGHTNESS (BONUS)
    # =========================
    y0, x0 = path[0]
    y1, x1 = path[-1]

    straight_dist = math.hypot(y1-y0, x1-x0)
    straightness = straight_dist / (length_px + 1e-6)

    return {
        "length": float(length_um),
        "min_width": min_w,
        "max_width": max_w,
        "straightness": float(straightness)
    }


def compute_feret_diameter(contour, pixel_size, angle_step=10):
    pts = contour[:, 0, :]
    max_f, min_f = 0, float('inf')

    for angle in range(0, 180, angle_step):
        theta = np.deg2rad(angle)
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        rotated = pts @ R.T
        xs = rotated[:, 0]
        feret = xs.max() - xs.min()

        max_f = max(max_f, feret)
        min_f = min(min_f, feret)

    return max_f * pixel_size, min_f * pixel_size


def compute_inscribed_circle(component, pixel_size):
    dist = cv2.distanceTransform(component, cv2.DIST_L2, 5)
    _, maxVal, _, maxLoc = cv2.minMaxLoc(dist)

    radius_px = maxVal
    diameter_um = 2 * radius_px * pixel_size

    return maxLoc, radius_px, diameter_um


def analyze_particles(particle_mask, pixel_size=1.0):
    results = []
    num, labels, stats, _ = cv2.connectedComponentsWithStats(particle_mask, 8)

    pid = 1
    if particle_mask is None:
        raise ValueError("particle_mask is None")

    if not isinstance(particle_mask, np.ndarray):
        raise ValueError("particle_mask is not numpy array")

    if particle_mask.dtype != np.uint8:
        particle_mask = particle_mask.astype(np.uint8)

    if len(particle_mask.shape) != 2:
        raise ValueError("particle_mask must be single channel")

    if np.sum(particle_mask) == 0:
        return []  # no particles
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 60:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w_box = stats[i, cv2.CC_STAT_WIDTH]
        h_box = stats[i, cv2.CC_STAT_HEIGHT]

        aspect = max(w_box, h_box) / (min(w_box, h_box) + 1e-6)

        # 🔥 FILTER (same as original)
        if aspect > 8:
            continue

        component = (labels == i).astype(np.uint8) * 255

        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = contours[0]

        # circularity
        area_cnt = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area_cnt / (peri**2 + 1e-6)

        if circularity < 0.3:
            continue
        sphericity = circularity

        # 🔵 inner circle
        center, radius_px, inner_d = compute_inscribed_circle(component, pixel_size)

        # 🟢 outer circle
        (x_o, y_o), r_outer = cv2.minEnclosingCircle(cnt)
        outer_d = 2 * r_outer * pixel_size

        # widths via distance map
        dist_map = cv2.distanceTransform(component, cv2.DIST_L2, 5)
        ys, xs = np.where(component > 0)
        widths = 2 * dist_map[ys, xs] * pixel_size

        valid_widths = widths[widths > 2]   # remove edge pixels

        min_w = float(np.percentile(valid_widths, 10))
        max_w = float(np.percentile(valid_widths, 95))

        length_um = max(w_box, h_box) * pixel_size

        ar_min = length_um / (max_w + 1e-6)
        ar_max = length_um / (min_w + 1e-6)

        feret_max, feret_min = compute_feret_diameter(cnt, pixel_size)

        results.append({
            "id": pid,
            "length": length_um,
            "min_width": min_w,
            "max_width": max_w,
            "ar_min": ar_min,
            "ar_max": ar_max,
            "feret_max": feret_max,
            "feret_min": feret_min,
            "inner_d": inner_d,
            "outer_d": outer_d,
            "sphericity": sphericity,
            "sides": len(cnt)
        })

        pid += 1

    return results
def first_page(canvas, doc):
    HEADER_PATH = "header.png"

    if os.path.exists(HEADER_PATH):
        img = ImageReader(HEADER_PATH)
        img_w, img_h = img.getSize()
        aspect = img_h / img_w

        page_width, page_height = A4
        draw_width = page_width
        draw_height = draw_width * aspect

        canvas.drawImage(
            img,
            0,
            page_height - draw_height,
            width=draw_width,
            height=draw_height,
            preserveAspectRatio=True,
            mask='auto'
        )

def later_pages(canvas, doc):
    page_width, page_height = A4

    # simple line separator
    y = page_height - 20
    canvas.setLineWidth(1)
    canvas.line(20, y, page_width - 20, y)

def generate_histograms(fibers, particles, output_folder):

    # fiber
    fiber_d = [f["metrics"]["max_width"] for f in fibers if f["metrics"]]

    if fiber_d:
        plt.figure()
        plt.hist(fiber_d, bins=20)
        plt.xlabel("Diameter")
        plt.ylabel("Count")
        plt.title("Fiber Diameter Distribution")
        plt.grid()
        plt.savefig(os.path.join(output_folder, "fiber_hist.png"))
        plt.close()

    # particle inner
    inner = [p["inner_d"] for p in particles]
    if inner:
        plt.figure()
        plt.hist(inner, bins=20)
        plt.title("Particle Inner Diameter")
        plt.grid()
        plt.savefig(os.path.join(output_folder, "particle_inner_hist.png"))
        plt.close()

    # particle outer
    outer = [p["outer_d"] for p in particles]
    if outer:
        plt.figure()
        plt.hist(outer, bins=20)
        plt.title("Particle Outer Diameter")
        plt.grid()
        plt.savefig(os.path.join(output_folder, "particle_outer_hist.png"))
        plt.close()

def generate_pdf(output_folder, fibers, particles):
    pdf_path = os.path.join(output_folder, "report.pdf")

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=20
    )

    styles = getSampleStyleSheet()
    elements = []
    elements.append(Spacer(1, 80))

    # ✅ REPORT TITLE
    elements.append(Paragraph(
        "<para align='center'><b>ANALYSIS REPORT</b></para>",
        styles['Title']
    ))
    elements.append(Spacer(1, 20))

    def add_image(title, filename):
        path = os.path.join(output_folder, filename)
        if os.path.exists(path):
            elements.append(Paragraph(f"<b>{title}</b>", styles['Heading3']))
            elements.append(Spacer(1, 5))
            elements.append(Image(path, width=160*mm, height=100*mm))
            elements.append(Spacer(1, 15))
    # =========================
    # 🔹 FIBER SECTION
    # =========================
    elements.append(Paragraph("Fiber Analysis", styles['Heading1']))

    # 1️⃣ Fiber Path Image
    add_image("Fiber Path", "05_fiber_graph_paths.png")

    # 2️⃣ Fiber Histogram
    add_image("Fiber Diameter Histogram", "fiber_hist.png")

    # =============================
    # FIBER TABLE (MOVE HERE)
    # =============================
    fiber_data = [["ID", "Length", "Min Width", "Max Width"]]

    for i, f in enumerate(fibers):
        if f["metrics"]:
            fiber_data.append([
                i+1,
                f"{f['metrics']['length']:.2f}",
                f"{f['metrics']['min_width']:.2f}",
                f"{f['metrics']['max_width']:.2f}"
            ])

    table2 = Table(fiber_data)
    table2.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ]))

    elements.append(Paragraph("<b>Fiber Measurements</b>", styles['Heading2']))
    elements.append(Spacer(1, 10))
    elements.append(table2)

    # =========================
    # 🔹 PARTICLE - POLYGON
    # =========================
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("Polygon Particle Analysis", styles['Heading1']))

    # 4️⃣ Polygon Image
    add_image("Polygon Particles", "particles_polygon.png")

    # 5️⃣ Polygon Histogram
    add_image("Polygon Histogram", "particle_inner_hist.png")


    # =========================
    # 🔹 PARTICLE - CIRCLE
    # =========================
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("Circle Particle Analysis", styles['Heading1']))

    # 6️⃣ Circle Image
    add_image("Circle Particles", "particles_circles.png")

    # 7️⃣ Circle Histogram
    add_image("Circle Histogram", "particle_outer_hist.png")


    # =========================
    # 🔹 PARTICLE TABLE
    # =========================
    elements.append(Spacer(1, 20))


    # =============================
    # PARTICLE TABLE
    # =============================
    particle_data = [[
        "ID", "Length", "MinW", "MaxW",
        "AR Min", "AR Max",
        "FeretMax", "FeretMin",
        "Inner D", "Outer D",
        "Sphericity", "Sides"
    ]]

    for p in particles:
        particle_data.append([
            p["id"],
            f"{p['length']:.2f}",
            f"{p['min_width']:.2f}",
            f"{p['max_width']:.2f}",
            f"{p['ar_min']:.2f}",
            f"{p['ar_max']:.2f}",
            f"{p['feret_max']:.2f}",
            f"{p['feret_min']:.2f}",
            f"{p['inner_d']:.2f}",
            f"{p['outer_d']:.2f}",
            f"{p['sphericity']:.2f}",
            p["sides"]
        ])

    table = Table(particle_data)
    table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ]))

    elements.append(Paragraph("<b>Particle Measurements</b>", styles['Heading2']))
    elements.append(Spacer(1, 10))
    elements.append(table)
    elements.append(Spacer(1, 20))

    # =============================
    # FIBER TABLE
    # =============================
    fiber_data = [["ID", "Length", "Min Width", "Max Width"]]

    for i, f in enumerate(fibers):
        if f["metrics"]:
            fiber_data.append([
                i+1,
                f"{f['metrics']['length']:.2f}",
                f"{f['metrics']['min_width']:.2f}",
                f"{f['metrics']['max_width']:.2f}"
            ])

    table2 = Table(fiber_data)
    table2.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ]))


    doc.build(elements, onFirstPage=first_page, onLaterPages=later_pages)

    return pdf_path

def save_fiber_image(orig, fibers, output_folder):
    # 🔥 keep original image intact
    display = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)

    for i, f in enumerate(fibers):
        path = f["path"]

        # 🔥 draw fiber path (green line)
        for j in range(1, len(path)):
            y1, x1 = path[j-1]
            y2, x2 = path[j]

            cv2.line(display, (x1, y1), (x2, y2), (0,255,0), 2)

        # 🔥 label (like your Image 2)
        if len(path) > 0:
            cy, cx = path[len(path)//2]

            label = f"F{i+1}: {f['metrics']['length']:.1f}um"

            # background box
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(display,
                          (cx, cy - h - 5),
                          (cx + w + 5, cy + 5),
                          (0,0,0),
                          -1)

            # text
            cv2.putText(display,
                        label,
                        (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0,255,255),
                        1,
                        cv2.LINE_AA)

    cv2.imwrite(os.path.join(output_folder, "05_fiber_graph_paths.png"), display)
def generate_particle_images(orig, particle_mask, output_folder):
    annotated_particles = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
    annotated_circles = annotated_particles.copy()

    num, labels, stats, _ = cv2.connectedComponentsWithStats(particle_mask, 8)

    pid = 1

    for i in range(1, num):
        component = (labels == i).astype(np.uint8) * 255

        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = contours[0]

        # 🔴 POLYGON (RED)
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.drawContours(annotated_particles, [approx], -1, (0,0,255), 2)

        # 🔵 INNER CIRCLE (BLUE)
        dist = cv2.distanceTransform(component, cv2.DIST_L2, 5)
        _, maxVal, _, maxLoc = cv2.minMaxLoc(dist)
        cx, cy = maxLoc
        radius = int(maxVal)

        cv2.circle(annotated_circles, (cx, cy), radius, (255,0,0), 2)

        # 🟢 OUTER CIRCLE (GREEN)
        (x, y), r = cv2.minEnclosingCircle(cnt)
        cv2.circle(annotated_circles, (int(x), int(y)), int(r), (0,255,0), 2)

        # 🟡 LABEL
        label = f"P{pid}"
        cv2.putText(annotated_particles, label, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0,255,255), 1, cv2.LINE_AA)

        pid += 1

    cv2.imwrite(os.path.join(output_folder, "particles_polygon.png"), annotated_particles)
    cv2.imwrite(os.path.join(output_folder, "particles_circles.png"), annotated_circles)