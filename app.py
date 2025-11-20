# =============================================================================
# FILE: app.py
# LOCATION: workpiece_app/app.py
# PURPOSE: Main Flask Web Server. 
#          FEATURES: 
#          - AUTOMATIC BACKGROUND REMOVAL using rembg (New!)
#          - CLAHE Segmentation (Handles shadows)
#          - Pitch: Middle 5 Valleys
#          - Depth: Line-to-Line distance
#          - Diameters: 5th/95th percentile widths in Thread Region
# =============================================================================

import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from scipy import signal, ndimage
from skimage import morphology
import math
from rembg import remove  # Import background removal
from PIL import Image     # Import PIL for image handling
import io

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Calibration Constants ---
SCALE_AXIS = 122 / 3406.31   # mm/pixel (Length)
SCALE_PERP = 20.12 / 589.89  # mm/pixel (Diameter)

def rotate_image(image, angle, center):
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    h, w = image.shape[:2]
    return cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_NEAREST)

def extract_profile_data(bw):
    rows, cols = np.nonzero(bw)
    if len(rows) == 0: return None, None, None

    min_y, max_y = np.min(rows), np.max(rows)
    margin = int((max_y - min_y) * 0.02) 
    start_y = min_y + margin
    end_y = max_y - margin
    
    widths = []
    right_profile = []
    valid_y = []

    for y in range(start_y, end_y):
        row_pixels = np.where(bw[y, :])[0]
        if len(row_pixels) > 0:
            l = row_pixels[0]
            r = row_pixels[-1]
            widths.append(r - l)
            right_profile.append(r)
            valid_y.append(y)
            
    return np.array(widths), np.array(right_profile), np.array(valid_y)

def find_threaded_region(profile_x, profile_y):
    if len(profile_x) < 100: return None, None 

    window_size = 50 
    roughness = [np.std(profile_x[i:i+window_size]) for i in range(len(profile_x) - window_size)]
    
    if len(roughness) > 10:
        smooth_roughness = signal.savgol_filter(roughness, 9, 2)
    else:
        smooth_roughness = np.array(roughness)

    max_roughness_robust = np.percentile(smooth_roughness, 95)
    threshold = max_roughness_robust * 0.3 
    rough_indices = np.where(smooth_roughness > threshold)[0]

    if len(rough_indices) == 0: return None, None

    diffs = np.diff(rough_indices)
    region_starts = [rough_indices[0]] + [rough_indices[i+1] for i, d in enumerate(diffs) if d > 1]
    region_ends = [rough_indices[i] for i, d in enumerate(diffs) if d > 1] + [rough_indices[-1]]

    best_start_idx = None
    best_end_idx = None
    max_length = 0

    for rs, re in zip(region_starts, region_ends):
        current_length = profile_y[re] - profile_y[rs]
        if current_length > max_length:
            max_length = current_length
            best_start_idx = rs
            best_end_idx = re

    if best_start_idx is not None and best_end_idx is not None:
        detected_height = best_end_idx - best_start_idx
        buffer = int(detected_height * 0.15)
        start_idx = max(0, best_start_idx - buffer)
        end_idx = min(len(profile_y) - 1, best_end_idx + buffer)
        return start_idx, end_idx
    else:
        return None, None

def filter_indices_by_pitch(indices, y_coords, min_pitch_mm, max_pitch_mm):
    if len(indices) < 2: return indices
    y_vals = y_coords[indices]
    diffs_mm = np.diff(y_vals) * SCALE_AXIS
    valid_gap_mask = (diffs_mm >= min_pitch_mm) & (diffs_mm <= max_pitch_mm)
    if not np.any(valid_gap_mask): return []

    padded_mask = np.concatenate(([False], valid_gap_mask, [False]))
    max_len = -1
    best_start = -1
    current_len = 0
    current_start = -1
    
    for i, is_valid in enumerate(valid_gap_mask):
        if is_valid:
            if current_len == 0: current_start = i
            current_len += 1
        else:
            if current_len > max_len:
                max_len = current_len
                best_start = current_start
            current_len = 0
    if current_len > max_len:
        max_len = current_len
        best_start = current_start
    if max_len == -1: return []

    return indices[best_start : best_start + max_len + 1]

def analyze_thread_signal(profile_x, profile_y):
    if len(profile_x) < 50: return None
    smooth_x = signal.savgol_filter(profile_x, 9, 2)
    window_size = 101
    if window_size > len(smooth_x): window_size = len(smooth_x) // 2 * 2 + 1
    trend = signal.savgol_filter(smooth_x, window_size, 2)
    normalized_signal = smooth_x - trend
    
    min_dist_pixels = int(1.0 / SCALE_AXIS) 
    search_distance = int(min_dist_pixels * 0.8)

    valleys_idx, _ = signal.find_peaks(-normalized_signal, height=2, distance=search_distance)
    peaks_idx, _ = signal.find_peaks(normalized_signal, height=2, distance=search_distance)

    valleys_idx = filter_indices_by_pitch(valleys_idx, profile_y, 0.8, 5.0)
    peaks_idx = filter_indices_by_pitch(peaks_idx, profile_y, 0.8, 5.0)

    if len(peaks_idx) < 2 or len(valleys_idx) < 2: return None
    return peaks_idx, valleys_idx, smooth_x

def fit_line_and_get_points(x_coords, y_coords):
    points = np.column_stack((x_coords, y_coords))
    line_params = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = line_params.flatten()
    min_y, max_y = np.min(y_coords), np.max(y_coords)
    def get_x(y):
        if abs(vy) < 1e-6: return x0 
        return x0 + (y - y0) * (vx / vy)
    p_start = [float(get_x(min_y)), float(min_y)]
    p_end = [float(get_x(max_y)), float(max_y)]
    return (vx, vy, x0, y0), [p_start, p_end]

def distance_between_lines(line1, line2):
    vx1, vy1, x1, y1 = line1
    vx2, vy2, x2, y2 = line2
    A = -vy1
    B = vx1
    C = vy1*x1 - vx1*y1
    dist = abs(A*x2 + B*y2 + C) / math.sqrt(A**2 + B**2)
    return float(dist)

def process_auto_metrics(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    
    # NOTE: Image is already processed by rembg in upload_image, so background is transparent (alpha=0)
    # We need to handle RGBA or convert transparent pixels to black
    
    if len(img.shape) == 3 and img.shape[2] == 4: # RGBA
        # Extract Alpha channel
        alpha = img[:, :, 3]
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        # Where alpha is 0 (transparent), make gray 0 (black)
        gray[alpha == 0] = 0
    elif len(img.shape) == 3: # RGB/BGR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # CLAHE & Segmentation
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray_enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray_enhanced, (7, 7), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if np.mean(gray[bw > 0]) < np.mean(gray[bw == 0]): 
        bw = cv2.bitwise_not(bw)
        
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    
    bw_bool = bw.astype(bool)
    bw_bool = ndimage.binary_fill_holes(bw_bool)
    bw_bool = morphology.remove_small_objects(bw_bool, 5000)
    bw = bw_bool.astype(np.uint8) * 255
    
    # Contour for visualization
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    viz_contour = []
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        epsilon = 0.002 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        viz_contour = approx.reshape(-1, 2).tolist()
    
    # Orientation
    y_idx, x_idx = np.nonzero(bw)
    if len(y_idx) == 0: return None
    
    coords = np.column_stack((x_idx, y_idx)).astype(float)
    mu = np.mean(coords, axis=0)
    cov_matrix = np.cov(coords, rowvar=False)
    eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
    sort_indices = np.argsort(eig_vals)[::-1]
    pc1 = eig_vecs[:, sort_indices[0]] 
    
    angle = np.arctan2(pc1[1], pc1[0]) * 180 / np.pi
    rotation_angle = 90 - angle
    
    cx, cy = int(mu[0]), int(mu[1])
    rot_bw = rotate_image(bw, rotation_angle, (cx, cy)) > 127

    # Extract Data
    widths, right_profile, valid_y = extract_profile_data(rot_bw)
    if widths is None: return None

    # Bounding Box
    rot_y, rot_x = np.nonzero(rot_bw)
    min_ry, max_ry = np.min(rot_y), np.max(rot_y)
    min_rx, max_rx = np.min(rot_x), np.max(rot_x)
    
    box_rot = np.array([
        [min_rx, min_ry, 1],
        [max_rx, min_ry, 1],
        [max_rx, max_ry, 1],
        [min_rx, max_ry, 1]
    ])
    
    inv_R = cv2.getRotationMatrix2D((cx, cy), -rotation_angle, 1.0)
    box_orig = inv_R.dot(box_rot.T).T
    
    vis_bounding_box = []
    for p in box_orig:
        vis_bounding_box.append([float(p[0]), float(p[1])])

    # Thread Region
    start_idx, end_idx = find_threaded_region(right_profile, valid_y)
    if start_idx is None:
        start_idx = int(len(valid_y) * 0.15)
        end_idx = int(len(valid_y) * 0.75)

    # --- METRICS ---
    length_px = max_ry - min_ry
    
    thread_zone_widths = widths[start_idx:end_idx]
    if len(thread_zone_widths) > 10:
        diameter_px = np.percentile(thread_zone_widths, 95)
    else:
        diameter_px = np.percentile(widths, 95)
        
    thread_metrics = {}
    all_points = {}
    
    region_right_x = right_profile[start_idx:end_idx]
    region_y = valid_y[start_idx:end_idx]
    
    depth_px = 0
    core_dia_px = 0

    thread_data = analyze_thread_signal(region_right_x, region_y)
    
    if thread_data:
        peaks_idx, valleys_idx, smooth_x = thread_data
        
        valleys_for_pitch = valleys_idx
        if len(valleys_idx) > 5:
            mid_idx = len(valleys_idx) // 2
            start_v = max(0, mid_idx - 2)
            valleys_for_pitch = valleys_idx[start_v : start_v + 5]

        valley_y_coords = region_y[valleys_for_pitch]
        if len(valley_y_coords) > 1:
            pitch_px = np.mean(np.diff(valley_y_coords))
        else:
            pitch_px = 0
            
        peak_x = smooth_x[peaks_idx]
        peak_y = region_y[peaks_idx]
        valley_x = smooth_x[valleys_idx]
        valley_y = region_y[valleys_idx]
        
        line_peaks, vis_line_p = fit_line_and_get_points(peak_x, peak_y)
        line_valleys, vis_line_v = fit_line_and_get_points(valley_x, valley_y)
        
        depth_px = distance_between_lines(line_peaks, line_valleys)
        
        thread_metrics = {
            "pitch_mm": round(float(pitch_px * SCALE_AXIS), 4),
            "depth_mm": round(float(depth_px * SCALE_PERP), 4),
            "count_peaks": int(len(peaks_idx)),
            "count_valleys": int(len(valleys_idx)),
            "count_valleys_used": int(len(valleys_for_pitch))
        }

        def map_back_points(indices, x_arr, y_arr):
            mapped = []
            for i in indices:
                px, py = x_arr[i], y_arr[i]
                pt = np.array([px, py, 1])
                orig = inv_R.dot(pt)
                mapped.append([float(orig[0]), float(orig[1])])
            return mapped

        def map_back_lines(pts_list):
            mapped = []
            for p in pts_list:
                pt = np.array([p[0], p[1], 1])
                orig = inv_R.dot(pt)
                mapped.append([float(orig[0]), float(orig[1])])
            return mapped
        
        all_points = {
            "peaks": map_back_points(peaks_idx, smooth_x, region_y),
            "valleys": map_back_points(valleys_idx, smooth_x, region_y),
            "valleys_used": map_back_points(valleys_for_pitch, smooth_x, region_y),
            "line_peaks": map_back_lines(vis_line_p),
            "line_valleys": map_back_lines(vis_line_v)
        }
    
    core_dia_px = diameter_px - (2 * depth_px)
    
    def get_line_coords_for_width(target_width, search_widths, start_offset):
        idx = (np.abs(search_widths - target_width)).argmin()
        real_idx = start_offset + idx
        y_val = valid_y[real_idx]
        x_right = right_profile[real_idx]
        x_left = x_right - target_width
        return [[x_left, y_val, 1], [x_right, y_val, 1]]

    def map_back_raw_lines(pts_list_homogenous):
        mapped = []
        for pt in pts_list_homogenous:
            orig = inv_R.dot(np.array(pt))
            mapped.append([float(orig[0]), float(orig[1])])
        return mapped

    vis_outer_dia_line_rot = get_line_coords_for_width(diameter_px, thread_zone_widths, start_idx)
    vis_core_dia_line_rot = get_line_coords_for_width(core_dia_px, thread_zone_widths, start_idx)
    
    all_points["vis_outer_dia"] = map_back_raw_lines(vis_outer_dia_line_rot)
    all_points["vis_core_dia"] = map_back_raw_lines(vis_core_dia_line_rot)

    return {
        "length_mm": round(float(length_px * SCALE_AXIS), 2),
        "od_mm": round(float(diameter_px * SCALE_PERP), 2),
        "core_mm": round(float(core_dia_px * SCALE_PERP), 2),
        "img_width": int(img.shape[1]),
        "img_height": int(img.shape[0]),
        "thread_metrics": thread_metrics,
        "all_points": all_points,
        "vis_bounding_box": vis_bounding_box,
        "contour": viz_contour 
    }

# --- ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files: return jsonify({'error': 'No file'})
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'No selected file'})
    
    # 1. Save Raw Image
    filename = file.filename
    raw_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(raw_path)
    
    # 2. Remove Background using REMBG
    try:
        with open(raw_path, 'rb') as i:
            input_data = i.read()
            output_data = remove(input_data)
            
        # Overwrite raw image with clean image
        # Or save as new file. Let's overwrite to keep it simple for process_auto_metrics
        with open(raw_path, 'wb') as o:
            o.write(output_data)
            
    except Exception as e:
        print(f"Background removal failed: {e}")
        # Proceed with raw image if rembg fails (better than crash)
    
    # 3. Process Clean Image
    metrics = process_auto_metrics(raw_path)
    if not metrics: return jsonify({'error': 'Processing failed'})
    
    return jsonify({'image_url': f'/uploads/{filename}', 'auto_data': metrics})

@app.route('/calculate_manual', methods=['POST'])
def calculate_manual():
    try:
        data = request.json
        points = data['points']
        if len(points) != 4: return jsonify({'error': 'Error: Need exactly 4 points'})
        
        p1, p2, v1, v2 = [np.array(p) for p in points]
        peak_pitch = np.linalg.norm(p2 - p1) * SCALE_AXIS
        valley_pitch = np.linalg.norm(v2 - v1) * SCALE_AXIS
        final_pitch = (peak_pitch + valley_pitch) / 2
        
        def get_line(pa, pb): return pb[1]-pa[1], pa[0]-pb[0], pb[0]*pa[1]-pa[0]*pb[1]
        A, B, C = get_line(p1, p2)
        if A == 0 and B == 0: return jsonify({'error': 'Points cannot be identical'})
        def dist(pt, A, B, C): return abs(A*pt[0] + B*pt[1] + C) / math.sqrt(A**2 + B**2)
        d1 = dist(v1, A, B, C)
        d2 = dist(v2, A, B, C)
        final_depth = ((d1 + d2) / 2) * SCALE_PERP
            
        return jsonify({
            'pitch_mm': round(float(final_pitch), 4),
            'depth_mm': round(float(final_depth), 4)
        })
    except Exception as e:
        return jsonify({'error': 'Calculation Error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
