from fastapi import FastAPI, UploadFile, File
import uuid
import os
import cv2
from processor import process_image, trace_fiber, compute_fiber_metrics, analyze_particles, generate_histograms, generate_pdf, generate_particle_images, save_fiber_image
from storage import save_data, get_mask, get_dist_map, save_fiber, get_fibers, get_particle_mask, save_pixel_size, get_pixel_size
from tifffile import TiffFile
import re
import time
from fastapi import BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
app = FastAPI()

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",  # Next.js
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/output", StaticFiles(directory="outputs"), name="output")
# ===============================
# 1. Upload & Process
# ===============================

def process_upload(image_path, image_id):
    output_folder = os.path.join(OUTPUT_DIR, image_id)
    os.makedirs(output_folder, exist_ok=True)

    # 🔥 HEAVY WORK HERE
    result = process_image(image_path, output_folder)

    save_data(
        image_id,
        result["mask"],
        result["dist_map"],
        result["particle_mask"]
    )

    generate_particle_images(
        result["orig"],
        result["particle_mask"],
        output_folder
    )
    time.sleep(0.5)
    
@app.post("/upload")
async def upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    image_id = str(uuid.uuid4())

    image_path = os.path.join(OUTPUT_DIR, f"{image_id}.tif")

    with open(image_path, "wb") as f:
        f.write(await file.read())

    background_tasks.add_task(process_upload, image_path, image_id)



    with TiffFile(image_path) as tif:
        page = tif.pages[0]
        resolution = (page.imagelength, page.imagewidth)

        pixel_size = None

        # =========================
        # 1️⃣ ImageDescription (SEM / FEI / Zeiss)
        # =========================
        desc = page.tags.get("ImageDescription")
        if desc:
            desc_str = str(desc.value)

            # try multiple patterns
            patterns = [
                r"Pixel Size\s*=\s*([\d.]+)",     # Pixel Size = 3.2195
                r"([\d.]+)\s*µm",                 # 3.2195 µm
                r"([\d.]+)\s*um",                 # 3.2195 um
            ]

            for p in patterns:
                match = re.search(p, desc_str)
                if match:
                    pixel_size = float(match.group(1))
                    break

        # =========================
        # 2️⃣ Resolution tags
        # =========================
        if pixel_size is None:
            x_res = page.tags.get("XResolution")
            unit = page.tags.get("ResolutionUnit")

            if x_res:
                num, den = x_res.value
                if num != 0:
                    # convert to µm (depends on unit)
                    dpi = num / den

                    if unit and unit.value == 2:  # inch
                        pixel_size = 25400 / dpi   # µm per pixel
                    elif unit and unit.value == 3:  # cm
                        pixel_size = 10000 / dpi


        # =========================
        # 3️⃣ FINAL CORRECT PARSER
        # =========================
        if pixel_size is None:
            full_text = ""

            if page.description:
                full_text += str(page.description)

            for tag in page.tags.values():
                try:
                    full_text += " " + str(tag.value)
                except:
                    pass

            clean_text = full_text.replace("\n", " ").replace("\r", " ")

            print("🔍 Searching correct Pixel Size...")

            # =========================
            # 1️⃣ PRIORITY: AP_PIXEL_SIZE
            # =========================
            match = re.search(r"AP_PIXEL_SIZE.*?Pixel Size\s*=\s*([\d.]+)", clean_text)

            if match:
                pixel_size = float(match.group(1))
                print("✅ AP_PIXEL_SIZE:", pixel_size)

            # =========================
            # 2️⃣ SECOND: IMAGE PIXEL SIZE
            # =========================
            if pixel_size is None:
                match = re.search(r"Image Pixel Size\s*=\s*([\d.]+)", clean_text)

                if match:
                    pixel_size = float(match.group(1))
                    print("✅ IMAGE PIXEL SIZE:", pixel_size)

            # =========================
            # 3️⃣ FALLBACK (scientific)
            # =========================
            if pixel_size is None:
                sci_matches = re.findall(r"\d+\.\d+e[-+]\d+", clean_text)

                for val in sci_matches:
                    num = float(val)

                    if 1e-9 < num < 1e-3:
                        pixel_size = num * 1e6
                        print("✅ SCIENTIFIC:", pixel_size)
                        break

            if pixel_size is None:
                print("❌ Pixel Size NOT FOUND")
        
        
        # =========================
        # 4️⃣ FINAL (no hardcode)
        # =========================
        if pixel_size is None:
            pixel_size = None   # only as safe fallback

        save_pixel_size(image_id, pixel_size)
    return {
        "image_id": image_id,
        "resolution": resolution,
        "pixel_size": pixel_size
    }

# 🔥 ADD THIS FUNCTION (top of file if not present)
def snap_to_fiber(point, mask, radius=40):
    y, x = point
    h, w = mask.shape

    best = None
    min_dist = float('inf')

    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            ny, nx = y + dy, x + dx

            if 0 <= ny < h and 0 <= nx < w:
                if mask[ny, nx] > 0:
                    d = dy*dy + dx*dx
                    if d < min_dist:
                        min_dist = d
                        best = (ny, nx)

    # 🔥 FORCE SNAP fallback
    if best is None:
        return point   # do not fail

    return best
# ===============================
# 2. Trace Fiber
# ===============================
@app.post("/trace")
def trace(data: dict):
    image_id = data["image_id"]
    points = data["points"]

    mask = get_mask(image_id)
    dist_map = get_dist_map(image_id)

    # =========================
    # SAFETY CHECKS
    # =========================
    if mask is None:
        return {"error": "Mask not found"}

    if dist_map is None:
        return {"error": "Distance map not found"}

    if len(points) < 2:
        return {"error": "At least 2 points required"}

    # =========================
    # REMOVE DUPLICATES
    # =========================
    unique_points = []
    for p in points:
        if p not in unique_points:
            unique_points.append(p)

    # =========================
    # CONNECTED COMPONENTS
    # =========================
    num_labels, labels = cv2.connectedComponents(mask)

    # =========================
    # STEP 1: INITIAL SNAP (FULL MASK)
    # =========================
    initial_snapped = [
        snap_to_fiber((p[0], p[1]), mask) for p in unique_points
    ]

    # 🔥 ensure first point is VALID fiber
    first_point = None

    for pt in initial_snapped:
        if mask[pt[0], pt[1]] > 0:
            first_point = pt
            break

    # ❌ if no valid fiber point
    if first_point is None:
        return {
            "error": "No valid fiber selected",
            "path": [],
            "metrics": None
        }

    component_id = labels[first_point[0], first_point[1]]
    print("First valid point:", first_point)
    print("Component ID:", component_id)
    # If clicked outside fiber → reject
    if component_id == 0:
        return {
            "error": "Click on fiber region",
            "path": [],
            "metrics": None
        }

    # Create mask for ONLY that fiber
    fiber_mask = (labels == component_id).astype('uint8') * 255

    # =========================
    # STEP 3: SNAP AGAIN (FIBER-ONLY)
    # =========================
    snapped_points = [
        snap_to_fiber((p[0], p[1]), fiber_mask) for p in initial_snapped
    ]

    # =========================
    # STEP 4: FILTER BAD POINTS
    # =========================
    filtered_points = [snapped_points[0]]

    for p in snapped_points[1:]:
        prev = filtered_points[-1]

        dist = ((p[0]-prev[0])**2 + (p[1]-prev[1])**2)**0.5

        if dist < 300:   # adjust if needed
            filtered_points.append(p)

    # =========================
    # STEP 5: TRACE PATH
    # =========================
    path = trace_fiber(filtered_points, fiber_mask)

    if not path:
        return {
            "error": "Path not found",
            "path": [],
            "metrics": None
        }

    # =========================
    # STEP 6: COMPUTE METRICS
    # =========================
    pixel_size = get_pixel_size(image_id)
    metrics = compute_fiber_metrics(path, dist_map, pixel_size)

    # =========================
    # STEP 7: SAVE
    # =========================
    save_fiber(image_id, {
        "path": path,
        "metrics": metrics
    })

    # =========================
    # RESPONSE
    # =========================
    return {
        "path": path,
        "metrics": metrics
    }

# ===============================
# 3. Get Results
# ===============================
@app.get("/results/{image_id}")
def results(image_id: str):
    return {
        "fibers": get_fibers(image_id)
    }

@app.get("/analyze/{image_id}")
def analyze(image_id: str):
    mask = get_mask(image_id)
    fibers = get_fibers(image_id)
    particle_mask = get_particle_mask(image_id)

    if particle_mask is None:
        return {"error": "Particle mask not found"}

    pixel_size = get_pixel_size(image_id)
    particles = analyze_particles(particle_mask, pixel_size)

    # ✅ SAME FOLDER FOR EVERYTHING
    output_folder = os.path.join(OUTPUT_DIR, image_id)

    generate_histograms(fibers, particles, output_folder)

    # 🔥 ADD THIS BEFORE RETURN
    save_fiber_image(
        cv2.imread(os.path.join(OUTPUT_DIR, image_id, "01_cropped.png"), 0),
        fibers,
        output_folder
    )
    pdf_path = generate_pdf(output_folder, fibers, particles)


    return {
        "particles": particles,
        "fibers": fibers,   # 🔥 ADD THIS
        "fiber_image": f"/output/{image_id}/05_fiber_graph_paths.png",  # 🔥 ADD
        "pdf": f"/output/{image_id}/report.pdf",
        "polygon": f"/output/{image_id}/particles_polygon.png",
        "circles": f"/output/{image_id}/particles_circles.png",
        "fiber_hist": f"/output/{image_id}/fiber_hist.png",
        "inner_hist": f"/output/{image_id}/particle_inner_hist.png",
        "outer_hist": f"/output/{image_id}/particle_outer_hist.png"
    }

@app.post("/clear_fibers")
def clear_fibers(data: dict):
    image_id = data["image_id"]

    # overwrite with empty
    save_data(image_id, get_mask(image_id), get_dist_map(image_id), get_particle_mask(image_id))
    
    return {"status": "cleared"}

@app.get("/status/{image_id}")
def status(image_id: str):
    mask = get_mask(image_id)

    return {
        "ready": mask is not None
    }