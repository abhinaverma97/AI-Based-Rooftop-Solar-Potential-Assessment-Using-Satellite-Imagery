import os
import io
import time
import json
import base64
import torch
import torchvision
import segmentation_models_pytorch as smp
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Initialize FastAPI App
app = FastAPI(title="Solar PV Detection Engine - SegFormer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
STATIC_DIR = os.path.join(BASE_DIR, "static")
MODEL_PATH = os.path.join(BASE_DIR, "best_segformer_large.pth")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading SegFormer Large Model on device: {device}...")

# Load SegFormer (MiT-B2) Large Model
model_name = "SegFormer MiT-B2 Large"
model = smp.Segformer(
    encoder_name="mit_b2",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Required model weights not found: {MODEL_PATH}")

state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
print(f"Loaded SegFormer Large weights: {MODEL_PATH}")

model.to(device)
model.eval()

# Global cache for latest detection geojson
latest_geojson_cache = {}

PRESET_METADATA = {
    "preset_1_residential_dense.jpg": {
        "id": "preset_1",
        "title": "Residential Neighborhood",
        "subtitle": "High-density residential rooftop arrays",
        "resolution": "0.15 m/px",
        "location": "Davis, California"
    },
    "preset_2_suburban_cluster.jpg": {
        "id": "preset_2",
        "title": "Suburban Cluster",
        "subtitle": "Mixed residential with roof angle variations",
        "resolution": "0.15 m/px",
        "location": "Davis, California"
    },
    "preset_3_commercial_flat.jpg": {
        "id": "preset_3",
        "title": "Commercial Facility",
        "subtitle": "Large flat-roof industrial array installation",
        "resolution": "0.15 m/px",
        "location": "Davis, California"
    },
    "preset_4_mixed_neighborhood.jpg": {
        "id": "preset_4",
        "title": "Mixed Housing Zone",
        "subtitle": "Multi-dwelling solar PV installations",
        "resolution": "0.15 m/px",
        "location": "Davis, California"
    },
    "preset_5_unseen_test.jpg": {
        "id": "preset_5",
        "title": "Unseen Test Orthomosaic",
        "subtitle": "Out-of-sample aerial evaluation sector",
        "resolution": "0.15 m/px",
        "location": "North Davis Sector"
    }
}

def image_to_base64(img_bgr: np.ndarray, format: str = "JPEG") -> str:
    _, buffer = cv2.imencode(f".{format.lower()}", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return f"data:image/{format.lower()};base64," + base64.b64encode(buffer).decode("utf-8")

def run_inference_on_image(img_rgb: np.ndarray, conf_thresh: float = 0.40, min_area_px: int = 40):
    t_start = time.time()
    orig_h, orig_w, _ = img_rgb.shape
    
    # Input normalization for SegFormer [0.0, 1.0]
    norm_img = img_rgb.astype(np.float32)
    if norm_img.max() > 1.0:
        norm_img /= 255.0
        
    # Pad image to multiples of 32 required by vision transformer downsampling
    pad_h = (32 - orig_h % 32) % 32
    pad_w = (32 - orig_w % 32) % 32
    if pad_h > 0 or pad_w > 0:
        padded_img = np.pad(norm_img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    else:
        padded_img = norm_img
        
    img_tensor = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(img_tensor)
        prob_map = torch.sigmoid(logits).squeeze().cpu().numpy()
        
    # Crop back to exact original unpadded dimensions
    if pad_h > 0 or pad_w > 0:
        prob_map = prob_map[:orig_h, :orig_w]
        
    if prob_map.shape != (orig_h, orig_w):
        prob_map = cv2.resize(prob_map, (orig_w, orig_h))
        
    t_infer = (time.time() - t_start) * 1000  # ms
    
    # Binary segmentation mask
    bin_mask = (prob_map >= conf_thresh).astype(np.uint8)
    
    # Morphological noise filter (from Google Satellite Large pipeline): removes isolated pixel noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel)
    
    # Extract connected array contours
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area_px:
            valid_contours.append(cnt)
            
    # Sort largest to smallest
    valid_contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)
    
    # Visual layers
    overlay_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    mask_canvas = np.zeros((orig_h, orig_w), dtype=np.uint8)
    
    # Color palette: Vivid Emerald Green + Electric Cyan Stroke
    mask_color_bgr = (0, 225, 110)
    border_color_bgr = (0, 255, 240)
    colored_mask_layer = np.zeros_like(overlay_bgr)
    
    detected_arrays = []
    features_geojson = []
    total_area_px = 0
    
    # 0.15m per pixel -> area per pixel = 0.0225 m^2
    M2_PER_PX = 0.0225
    KW_PER_M2 = 0.175  # ~175W/m2 standard PV efficiency
    
    for i, cnt in enumerate(valid_contours):
        area_px = int(cv2.contourArea(cnt))
        total_area_px += area_px
        
        bx, by, bw, bh = cv2.boundingRect(cnt)
        
        # Calculate mean confidence inside this solar polygon
        cnt_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        cv2.drawContours(cnt_mask, [cnt], -1, 1, -1)
        mean_conf = float(np.mean(prob_map[cnt_mask == 1])) * 100
        
        # Draw on mask canvas & colored overlay
        cv2.drawContours(mask_canvas, [cnt], -1, 255, -1)
        cv2.drawContours(colored_mask_layer, [cnt], -1, mask_color_bgr, -1)
        cv2.drawContours(overlay_bgr, [cnt], -1, border_color_bgr, 2)
        
        approx = cv2.approxPolyDP(cnt, 1.5, True)
        poly_coords = []
        if len(approx) >= 3:
            pts = [[float(p[0][0]), float(p[0][1])] for p in approx]
            pts.append(pts[0])
            poly_coords.append(pts)
            
        area_m2 = round(area_px * M2_PER_PX, 2)
        est_kw = round(area_m2 * KW_PER_M2, 2)
        
        detected_arrays.append({
            "id": i + 1,
            "confidence": round(mean_conf, 1),
            "area_px": area_px,
            "area_m2": area_m2,
            "est_kw": est_kw,
            "bbox": [int(bx), int(by), int(bw), int(bh)]
        })
        
        if poly_coords:
            features_geojson.append({
                "type": "Feature",
                "properties": {
                    "id": i + 1,
                    "confidence": round(mean_conf / 100, 3),
                    "area_m2": area_m2,
                    "est_kw": est_kw
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": poly_coords[0] if len(poly_coords) == 1 else poly_coords
                }
            })
            
    # Blend overlay with 45% transparency
    blended = cv2.addWeighted(overlay_bgr, 0.55, colored_mask_layer, 0.45, 0)
    overlay_bgr[mask_canvas > 0] = blended[mask_canvas > 0]
    
    # Re-draw clean boundary strokes and stylized metadata pills
    for item in detected_arrays:
        bx, by, bw, bh = item["bbox"]
        label = f"#{item['id']} {item['confidence']}% ({item['area_m2']} m2)"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        tag_y = max(th + 6, by - 6)
        
        cv2.rectangle(overlay_bgr, (bx, tag_y - th - 4), (bx + tw + 6, tag_y + 4), (10, 15, 24), -1)
        cv2.rectangle(overlay_bgr, (bx, tag_y - th - 4), (bx + tw + 6, tag_y + 4), (0, 225, 110), 1)
        cv2.putText(overlay_bgr, label, (bx + 3, tag_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 240), 1, cv2.LINE_AA)
        
    # Generate High-Resolution VIRIDIS Heatmap
    heatmap_colored = cv2.applyColorMap((prob_map * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    
    # Aggregated calculations
    total_area_m2 = round(total_area_px * M2_PER_PX, 2)
    total_kw = round(total_area_m2 * KW_PER_M2, 2)
    annual_kwh = round(total_kw * 1450, 0)
    co2_saved_tons = round(annual_kwh * 0.0004, 2)
    
    geojson_data = {
        "type": "FeatureCollection",
        "properties": {
            "model_name": model_name,
            "total_arrays": len(detected_arrays),
            "total_area_m2": total_area_m2,
            "total_capacity_kw": total_kw,
            "annual_yield_kwh": annual_kwh,
            "inference_time_ms": round(t_infer, 1)
        },
        "features": features_geojson
    }
    
    global latest_geojson_cache
    latest_geojson_cache = geojson_data
    
    return {
        "success": True,
        "model_name": model_name,
        "device": str(device).upper(),
        "inference_time_ms": round(t_infer, 1),
        "total_arrays": len(detected_arrays),
        "total_area_m2": total_area_m2,
        "total_capacity_kw": total_kw,
        "annual_yield_kwh": int(annual_kwh),
        "co2_saved_tons": co2_saved_tons,
        "arrays": detected_arrays,
        "images": {
            "original": image_to_base64(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)),
            "overlay": image_to_base64(overlay_bgr),
            "mask": image_to_base64(cv2.cvtColor(mask_canvas, cv2.COLOR_GRAY2BGR)),
            "heatmap": image_to_base64(heatmap_colored)
        }
    }

@app.get("/api/presets")
def get_presets():
    presets = []
    for fname in sorted(os.listdir(SAMPLES_DIR)):
        if fname.endswith((".jpg", ".png", ".tif")):
            fpath = os.path.join(SAMPLES_DIR, fname)
            meta = PRESET_METADATA.get(fname, {
                "id": fname,
                "title": fname.replace("_", " ").title(),
                "subtitle": "Aerial Sample",
                "resolution": "0.15 m/px",
                "location": "Davis, CA"
            })
            img_bgr = cv2.imread(fpath)
            thumb = cv2.resize(img_bgr, (160, 160)) if img_bgr is not None else None
            thumb_b64 = image_to_base64(thumb) if thumb is not None else ""
            presets.append({
                "filename": fname,
                "title": meta["title"],
                "subtitle": meta["subtitle"],
                "resolution": meta["resolution"],
                "location": meta["location"],
                "thumbnail": thumb_b64
            })
    return {"presets": presets}

@app.post("/api/predict")
async def predict(
    file: UploadFile = File(None),
    preset_filename: str = Form(None),
    confidence_threshold: float = Form(0.40),
    min_area: int = Form(40)
):
    img_rgb = None
    if file and file.filename:
        content = await file.read()
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
        img_rgb = np.array(pil_img)
    elif preset_filename:
        preset_path = os.path.join(SAMPLES_DIR, preset_filename)
        if not os.path.exists(preset_path):
            raise HTTPException(status_code=404, detail="Preset file not found")
        img_bgr = cv2.imread(preset_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        raise HTTPException(status_code=400, detail="Either an image file or preset_filename must be provided")

    # Resize if extremely large to prevent OOM
    h, w, _ = img_rgb.shape
    if max(h, w) > 1024:
        scale = 1024 / max(h, w)
        img_rgb = cv2.resize(img_rgb, (int(w * scale), int(h * scale)))

    result = run_inference_on_image(img_rgb, conf_thresh=confidence_threshold, min_area_px=min_area)
    return result

@app.get("/api/export_geojson")
def export_geojson():
    global latest_geojson_cache
    if not latest_geojson_cache:
        return JSONResponse(status_code=404, content={"detail": "No detection performed yet"})
    return JSONResponse(content=latest_geojson_cache)

# Mount static files for HTML/CSS/JS frontend
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    print("Starting Solar Detection Server on http://localhost:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
