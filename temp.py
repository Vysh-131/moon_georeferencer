import os
import time
import json
import threading
import random
import queue
import requests
from bs4 import BeautifulSoup
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO

Image.MAX_IMAGE_PIXELS = None

# ==========================================
# CONFIGURATION
# ==========================================
TARGET_QUOTA = 1400  # Number of images you want to mine before stopping
CONFIDENCE = 0.40
SEARCH_RESULTS = "SearchResults.txt"
MODEL_WEIGHTS = "ejecta.pt"  # Assuming weights are one folder up

# Directories
IMG_DIR = "images"
INPUT_DIR = "input"
STATE_FILE = "miner_state.json"

for d in [IMG_DIR, INPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# ==========================================
# RESUMABILITY STATE
# ==========================================
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {"processed_pids": [], "total_harvested": 0}

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)

# ==========================================
# PARSER & DOWNLOADER
# ==========================================
def parse_search_results(filepath):
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return {}
        
    product_map = {}
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
    header = []
    for line in lines:
        if line.startswith("#INSTRUMENT HOST ID"):
            header = [col.strip().upper() for col in line.replace('#', '').split(',')]
            break
            
    if not header:
        return product_map
        
    try:
        pid_idx = header.index("PRODUCT ID")
        link_idx = header.index("ORBITAL DATA EXPLORER PRODUCT FILES LINK")
    except ValueError:
        return product_map
        
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        cols = [col.strip() for col in line.split(',')]
        if len(cols) > max(pid_idx, link_idx):
            pid = cols[pid_idx].upper()
            link = cols[link_idx]
            product_map[pid] = link
            
    return product_map

def download_image(pid, ode_url):
    headers = {"User-Agent": "Mozilla/5.0"}
    clean_pid = pid.upper().replace('NAC.', '').replace('WAC.', '')
    
    try:
        response = requests.get(ode_url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        download_url = None
        file_ext = None
        
        for a in soup.find_all('a', href=True):
            href = a['href']
            href_upper = href.upper()
            
            if clean_pid in href_upper and (href_upper.endswith('.TIF') or href_upper.endswith('.IMG')):
                download_url = href
                file_ext = ".TIF" if href_upper.endswith('.TIF') else ".IMG"
                break
                
        if not download_url:
            return None
            
        if not download_url.startswith('http'):
            download_url = "https://ode.rsl.wustl.edu/moon/" + download_url.lstrip('/')
            
        filename = f"{clean_pid}{file_ext}"
        filepath = os.path.join(INPUT_DIR, filename)
        
        if os.path.exists(filepath):
            return filepath
            
        with requests.get(download_url, headers=headers, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
        return filepath
    except Exception:
        return None

def convert_to_tif(img_path):
    if img_path.upper().endswith('.TIF'):
        return img_path
    
    out_path = img_path.rsplit('.', 1)[0] + '.tif'
    try:
        import rasterio
        with rasterio.open(img_path) as src:
            meta = src.meta.copy()
            meta.update(driver='GTiff')
            with rasterio.open(out_path, 'w', **meta) as dst:
                dst.write(src.read())
        return out_path
    except Exception:
        return None

# ==========================================
# PARALLEL FETCH WORKER
# ==========================================
def fetch_worker(download_queue, product_map, state):
    for pid, link in product_map.items():
        if pid in state["processed_pids"]:
            continue
            
        raw_path = download_image(pid, link)
        if raw_path:
            tif_path = convert_to_tif(raw_path)
            if tif_path and tif_path != raw_path and os.path.exists(raw_path):
                os.remove(raw_path)
                
            if tif_path:
                print(f"Downloaded {pid}")
                download_queue.put((pid, tif_path))
            else:
                download_queue.put((pid, None))
        else:
            download_queue.put((pid, None))
            
    download_queue.put((None, None))

# ==========================================
# PROCESSING LOGIC
# ==========================================
def calculate_global_histogram(pil_img):
    w, h = pil_img.size
    
    # Small image fallback
    if w < 2048 and h < 2048:
        arr = np.array(pil_img)
        valid = arr[arr > 0]
    else:
        # 5-Block Random Sampling
        samples = []
        for _ in range(5):
            x = random.randint(0, max(0, w - 1024))
            y = random.randint(0, max(0, h - 1024))
            crop = pil_img.crop((x, y, x + 1024, y + 1024))
            samples.append(np.array(crop).flatten())
            
        arr = np.concatenate(samples)
        valid = arr[arr > 0]
        
        # Cap processing at 5,000,000 pixels to save memory
        if len(valid) > 5000000:
            valid = np.random.choice(valid, 5000000, replace=False)

    if len(valid) == 0:
        return 0, 255
        
    return np.percentile(valid, 1), np.percentile(valid, 99.5)
def process_and_harvest(pid, img_path, model, state):
    harvested_in_image = 0
    
    try:
        pil_img = Image.open(img_path)
        g_min, g_max = calculate_global_histogram(pil_img)
        
        raw_w, raw_h = pil_img.size
        m_size = 2048
        m_overlap = 512
        step = m_size - m_overlap
        
        for y in range(0, raw_h - step, step):
            for x in range(0, raw_w - step, step):
                if state["total_harvested"] >= TARGET_QUOTA:
                    return harvested_in_image

                w_crop = min(m_size, raw_w - x)
                h_crop = min(m_size, raw_h - y)
                if w_crop < m_size or h_crop < m_size:
                    continue

                roi = pil_img.crop((x, y, x + w_crop, y + h_crop))
                chunk_arr = np.array(roi).astype(np.float32)
                
                # Apply Section 3.2 Linear Scaling
                chunk_arr = np.clip(chunk_arr, g_min, g_max)
                chunk_arr = (chunk_arr - g_min) / (g_max - g_min + 1e-5) * 255.0
                
                if len(chunk_arr.shape) == 3 and chunk_arr.shape[2] == 3:
                    gray_chunk = cv2.cvtColor(chunk_arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                else:
                    gray_chunk = chunk_arr.astype(np.uint8)

                # --- Section 5.1: Dynamic Illumination ---
                mean_brightness = np.mean(gray_chunk)
                
                if mean_brightness > 180:
                    clip_limit = 1.0  # Mild for bright highlands
                elif mean_brightness > 130:
                    clip_limit = 1.5  # Standard
                elif mean_brightness < 60:
                    clip_limit = 3.0  # Aggressive for dark shadows
                else:
                    clip_limit = 2.0  # Default middle-ground
                    
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(16, 16))
                enhanced_chunk = clahe.apply(gray_chunk)
                # -----------------------------------------

                inference_chunk = cv2.cvtColor(enhanced_chunk, cv2.COLOR_GRAY2RGB)
                
                results = model(inference_chunk, conf=CONFIDENCE, device='cpu', verbose=False)
                
                if results and results[0].boxes:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        pad_x = int((x2 - x1) * 0.3)
                        pad_y = int((y2 - y1) * 0.3)
                        
                        nx1, ny1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                        nx2, ny2 = min(gray_chunk.shape[1], x2 + pad_x), min(gray_chunk.shape[0], y2 + pad_y)
                        
                        if (nx2 - nx1) < 50 or (ny2 - ny1) < 50:
                            continue

                        # Save the dynamically illuminated tile
                        final_tile = enhanced_chunk[ny1:ny2, nx1:nx2]
                        final_tile_rgb = cv2.cvtColor(final_tile, cv2.COLOR_GRAY2RGB)
                        
                        filename = f"{IMG_DIR}/{pid}_{x}_{y}_{harvested_in_image}.png"
                        cv2.imwrite(filename, final_tile_rgb, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                        
                        harvested_in_image += 1
                        state["total_harvested"] += 1
                        
                        if state["total_harvested"] >= TARGET_QUOTA:
                            break
                            
                roi.close()
                
    except Exception as e:
        print(f"Error processing {pid}: {e}")
    finally:
        if 'pil_img' in locals():
            pil_img.close()
            
    return harvested_in_image
                                
# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    state = load_state()
    
    if state["total_harvested"] >= TARGET_QUOTA:
        print(f"Target quota of {TARGET_QUOTA} already reached. Exiting.")
        exit(0)
        
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"Error: Weights not found at {MODEL_WEIGHTS}")
        exit(1)
        
    model = YOLO(MODEL_WEIGHTS)
    product_map = parse_search_results(SEARCH_RESULTS)
    
    if not product_map:
        print("No valid products found in SearchResults.txt.")
        exit(0)
        
    download_queue = queue.Queue(maxsize=1)
    fetch_thread = threading.Thread(target=fetch_worker, args=(download_queue, product_map, state), daemon=True)
    fetch_thread.start()
    
    while True:
        pid, img_path = download_queue.get()
        
        if pid is None:
            break
            
        if img_path:
            obtained = process_and_harvest(pid, img_path, model, state)
            os.remove(img_path)
            state["processed_pids"].append(pid)
            save_state(state)
            print(f"Processed {pid} - Obtained {obtained} images (Total: {state['total_harvested']}/{TARGET_QUOTA})")
            
        if state["total_harvested"] >= TARGET_QUOTA:
            print("Quota reached. Shutting down gracefully.")
            break
