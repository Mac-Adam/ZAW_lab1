import cv2
import numpy as np
import os

IN_DEPTH_MODE = False
CONFIG = {
    'mask_ranges': {
        'glare': [(0, 255), (0, 55), (191, 255)],
        'brown_cable': [(0, 110), (0, 48), (20, 128)],
        'blue_cable': [(97, 120), (114, 255), (0, 255)],
        'yellow_cable': [(29, 75), (49, 155), (74, 206)],
        'insulation': [(85, 103), (39, 113), (71, 255)],
        'black': [(0, 255), (21, 85), (0, 25)],
        'copper': [(0, 24), (63, 105), (30, 255)]
    },
    'locations': {
        'brown_cable': [(450, 1024), (450, 1024)],
        'blue_cable': [(0, 550), (450, 1024)],
        'yellow_cable': [(0, 1024), (0, 512)],
    },
    'mask_filters': {
        'glare': [0, 0, 0],
        'brown_cable': [2000, 0, 0],
        'blue_cable': [2000, 0, 0],
        'yellow_cable': [2000, 0, 0],
        'insulation': [24000, 0, 0],
        'black': [10000, 0, 0.2],
        'copper': [0, 0, 0]
    },
    'matching_thr': {
        'brown_cable': [170, 20],
        'blue_cable': [170, 20],
        'yellow_cable': [170, 20],
        'insulation': [200, 10],
        'wire': [235, 85],
        'black': [170, 20]
    },
    'geometry': {
        'roi_radius': 425,
        'donut_min_pts': 50,
        'donut_min_hull': 5,
        'donut_max_aspect': 1.50,
        'donut_area_ratio': 0.114
    },
    'wire_strands': {
        'min_area': 200,
        'max_area': 42000,
        'circularity_thresh': 0.5275, 
        'dilate_size': 13
    },
    'insulation_defects': {
        'min_large_area': 15000,
        'avg_area_high': 280,
        'avg_area_low': 140,
        'defect_count_thr': 8,
        'morph_open_k': 7
    },
    'combine': {
        'wire_open_k': 3,
        'wire_close1_k': 11, 
        'wire_dilate_k': 15,
        'wire_dilate_iter': 1,
        'wire_close2_k': 5,
        'wire_cleanup_params': [8000, 0.6, 0.5],
        'wire_cleanup_override': 25000,
        'brown_close_k': 7,
        'brown_open_k': 13,
        'brown_cleanup_params': [5000, 0, 0]
    },
    'predict': {
        'donut_erode_k': 5,
        'donut_erode_iter': 6,
        'donut_err_cleanup': [900, 0.76, 0.39],
        'binary_err_cleanup': [1200, 0, 0],
        'final_open_k': 7
    },
    'evaluation': {
        'min_defect_pixels_to_alarm': 50
    }
}

CACHED_GOLDEN_MASKS = {}

def fit_ideal_donut(mask: np.ndarray, config: dict):
    ideal_donut = np.zeros_like(mask)
    points = cv2.findNonZero(mask)

    if points is None or len(points) < config['geometry']['donut_min_pts']:
        return None, None

    hull = cv2.convexHull(points)
    
    if len(hull) < config['geometry']['donut_min_hull']:
        return None, None
        
    outer_ellipse = cv2.fitEllipse(hull)
    (center, (w, h), angle) = outer_ellipse

    if min(w, h) == 0:
        return None, None
        
    aspect_ratio = max(w, h) / min(w, h)
    ellipse_area = (np.pi * w * h) / 4
    if aspect_ratio > config['geometry']['donut_max_aspect']: 
        return None, None

    if ellipse_area > (mask.shape[0] * mask.shape[1] * config['geometry']['donut_area_ratio']):
        return None, None

    solid_outer = np.zeros_like(mask)
    cv2.ellipse(solid_outer, outer_ellipse, 255, -1)
    
    internal_void = cv2.bitwise_and(solid_outer, cv2.bitwise_not(mask))
    dist_transform = cv2.distanceTransform(internal_void, cv2.DIST_L2, 5)
    _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)
    
    ideal_donut = solid_outer.copy()
    inner_radius = int(max_val)
    if inner_radius > 0:
        cv2.circle(ideal_donut, max_loc, inner_radius, 0, -1)
        
    return ideal_donut, max_loc

def check_wire_strands(wire_mask: np.ndarray, config: dict):
    p = config['wire_strands']
    circular_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p['dilate_size'], p['dilate_size']))
    smooth_mask = cv2.dilate(wire_mask, circular_kernel, iterations=1)
    error_mask = np.zeros_like(smooth_mask)
    contours, _ = cv2.findContours(smooth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < p['min_area']:
            continue 

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity < p['circularity_thresh'] or area > p['max_area']:
            cv2.drawContours(error_mask, [contour], -1, 255, -1)

    return error_mask

def check_insulation_defects(c_mask: np.ndarray, g_mask: np.ndarray, matching_thr_tuple: tuple, config: dict):
    p = config['insulation_defects']
    error_mask = np.zeros_like(c_mask)
    H, W = c_mask.shape
    total_image_area = H * W
    inv_mask = cv2.bitwise_not(c_mask)
    
    contours, _ = cv2.findContours(inv_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    avg_area = 0
    large_contours = []
    defect_contours = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > (total_image_area * 0.95):
            continue

        if area > p['min_large_area']:
            large_contours.append(cnt)
        else:
            if area > 10: 
                avg_area += area
                defect_contours.append(cnt)
                
    if avg_area != 0:
        avg_area /= len(defect_contours)

    if len(large_contours) == 2:
        if avg_area > p['avg_area_high'] or (avg_area > p['avg_area_low'] and len(defect_contours) >= p['defect_count_thr']):
            cv2.drawContours(error_mask, defect_contours, -1, 255, -1)
        
        k_size = p['morph_open_k']
        error_mask = cv2.morphologyEx(error_mask, cv2.MORPH_OPEN, np.ones((k_size, k_size), np.uint8))
        return error_mask
        
    else:
        g_mask_must = np.zeros_like(c_mask)
        g_mask_must[g_mask > matching_thr_tuple[0]] = 255

        g_mask_can = np.zeros_like(c_mask)
        g_mask_can[g_mask > matching_thr_tuple[1]] = 255
        
        err = (g_mask_must == 255) & (c_mask == 0)
        err |= (c_mask == 255) & (g_mask_can == 0)
        
        error_mask = (err.astype(np.uint8) * 255)
        return error_mask

def predict(image: np.ndarray, config: dict = CONFIG, debug: bool = False) -> np.ndarray:
    global CACHED_GOLDEN_MASKS
    H, W = image.shape[:2]
    defect_mask = np.zeros((H, W), dtype=np.uint8)

    mask_names = ['insulation', 'blue_cable', 'yellow_cable', 'brown_cable', 'wire', 'black']
    
    if not CACHED_GOLDEN_MASKS:
        base_dir = os.path.dirname(os.path.abspath(__file__)) 
        for name in mask_names:
            path = os.path.join(base_dir, "masks", f"mask_{name}.png")
            if os.path.exists(path):
                CACHED_GOLDEN_MASKS[name] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            else:
                print(f"Warning: {path} not found!")
                return defect_mask

    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    raw_masks = make_raw_masks(bgr_image, config)
    proc_masks = process_masks(raw_masks, config)
    current_masks = combine_masks(proc_masks, config)

    errors = {}
    p = config['predict']

    for key in current_masks:
        c_mask = current_masks[key]
        g_mask = CACHED_GOLDEN_MASKS[key]

        if 'cable' in key:
            if key == "brown_cable":
                continue
            ideal_donut, location = fit_ideal_donut(c_mask, config)
            loc_cfg = config['locations'][key]
            
            if ideal_donut is not None and (loc_cfg[0][0] < location[0] < loc_cfg[0][1]) and (loc_cfg[1][0] < location[1] < loc_cfg[1][1]):
                k_size = p['donut_erode_k']
                ideal_donut = cv2.morphologyEx(ideal_donut, cv2.MORPH_ERODE, np.ones((k_size, k_size)), iterations=p['donut_erode_iter'])
                err = cv2.bitwise_and(ideal_donut, cv2.bitwise_not(c_mask))
                err = clean_up_mask(err, p['donut_err_cleanup'])
            else:
                g_binary = np.zeros_like(g_mask)
                g_binary[g_mask > config['matching_thr'][key][1]] = 255
                err = cv2.bitwise_and(c_mask, cv2.bitwise_not(g_binary))
                err = clean_up_mask(err, p['binary_err_cleanup'])

        elif key == 'wire':
            err = check_wire_strands(c_mask, config)
        elif key == 'insulation':
            err = check_insulation_defects(c_mask, g_mask, config['matching_thr'][key], config)
        else:
            g_mask_must = np.zeros(c_mask.shape, dtype=np.uint8)
            g_mask_must[g_mask > config['matching_thr'][key][0]] = 255

            g_mask_can = np.zeros(c_mask.shape, dtype=np.uint8)
            g_mask_can[g_mask > config['matching_thr'][key][1]] = 255
            
            err = (g_mask_must == 255) & (c_mask == 0)
            err |= (c_mask == 255) & (g_mask_can == 0)
            err = (err.astype(np.uint8) * 255)
        
        errors[key] = err

        # Restore the debug views
        if debug:
            c_disp = cv2.resize(c_mask, (256, 256))
            g_disp = cv2.resize(g_mask, (256, 256))
            e_disp = cv2.resize(err, (256, 256))
            comparison_img = np.hstack((g_disp, c_disp, e_disp))
            win_name = f"Compare [Golden | Current | Err]: {key}"
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_name, 768, 256)
            cv2.imshow(win_name, comparison_img)

    final_k = p['final_open_k']
    ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (final_k, final_k))
    for key, err in errors.items():
        err = cv2.morphologyEx(err, cv2.MORPH_OPEN, ellipse_kernel)
        defect_mask |= err

    return defect_mask

def validate_prediction(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)

    TP = np.sum((pred == 1) & (gt == 1))
    TN = np.sum((pred == 0) & (gt == 0))
    FP = np.sum((pred == 1) & (gt == 0))
    FN = np.sum((pred == 0) & (gt == 1))

    def safe_divide(numerator, denominator):
        return float(numerator) / float(denominator) if denominator > 0 else 0.0

    if TP + FP + FN == 0:
        iou = 1.0
        f1 = 1.0
    else:
        iou = safe_divide(TP, TP + FP + FN)
        f1 = safe_divide(2 * TP, 2 * TP + FP + FN)

    return {
        "Accuracy": safe_divide(TP + TN, TP + TN + FP + FN),
        "Precision": safe_divide(TP, TP + FP),
        "Recall": safe_divide(TP, TP + FN),
        "F1-Score": f1,
        "IoU": iou
    }

def clean_up_mask(mask, params, size_override=0):
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    good_idx = []
    MIN_AREA = params[0]
    MIN_ASPECT_RATIO = params[1]
    MIN_EXTENT = params[2]
    
    for i in range(1, retval):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= MIN_AREA:
            if size_override != 0 and area > size_override:
                good_idx.append(i)
                continue
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            if w == 0 or h == 0:
                continue
                
            aspect_ratio = min(w, h) / max(w, h)
            bounding_box_area = w * h
            extent = area / bounding_box_area

            if aspect_ratio >= MIN_ASPECT_RATIO and MIN_EXTENT <= extent:
                good_idx.append(i)

    return (np.isin(labels, good_idx) * 255).astype(np.uint8)

def make_raw_masks(image, config: dict):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    W, H = image.shape[:2]
    area_of_intrest = np.zeros((H, W), dtype=np.uint8)

    center = (W // 2, H // 2)
    radius = config['geometry']['roi_radius']

    cv2.circle(area_of_intrest, center, radius, 255, -1)

    masks = {}
    for key, values in config['mask_ranges'].items():
        lower_bound = np.array([r[0] for r in values])
        upper_bound = np.array([r[1] for r in values])
        masks[key] = cv2.inRange(hsv, lower_bound, upper_bound) & area_of_intrest
    
    return masks

def process_masks(masks, config: dict):
    new_masks = {}
    for key, mask in masks.items():
        new_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3)), iterations=3)
        new_masks[key] = clean_up_mask(new_mask, config['mask_filters'][key])

    return new_masks

def combine_masks(masks, config: dict):
    p = config['combine']
    new_masks = {}
    already_verified = masks['insulation'].copy()
    new_masks['insulation'] = masks['insulation']

    new_masks['blue_cable'] = masks['blue_cable'] & ~already_verified
    already_verified |= new_masks['blue_cable']

    new_masks['yellow_cable'] = masks['yellow_cable'] & ~already_verified
    already_verified |= new_masks['yellow_cable']

    new_masks['black'] = masks['black'] & ~already_verified
    already_verified |= new_masks['black']

    new_masks['wire'] = (masks['copper'] | masks['glare']) & ~already_verified
    
    new_masks['wire'] = cv2.morphologyEx(new_masks['wire'], cv2.MORPH_CLOSE, np.ones((p['wire_close1_k'], p['wire_close1_k'])))
    new_masks['wire'] = cv2.morphologyEx(new_masks['wire'], cv2.MORPH_OPEN, np.ones((p['wire_open_k'], p['wire_open_k'])))
    new_masks['wire'] = cv2.morphologyEx(new_masks['wire'], cv2.MORPH_DILATE, np.ones((p['wire_dilate_k'], p['wire_dilate_k'])), iterations=p['wire_dilate_iter'])
    new_masks['wire'] = cv2.morphologyEx(new_masks['wire'], cv2.MORPH_CLOSE, np.ones((p['wire_close2_k'], p['wire_close2_k'])))
    new_masks['wire'] = clean_up_mask(new_masks['wire'], p['wire_cleanup_params'], size_override=p['wire_cleanup_override'])

    already_verified |= new_masks['wire']

    new_masks['brown_cable'] = masks['brown_cable'] & ~already_verified
    new_masks['brown_cable'] = cv2.morphologyEx(new_masks['brown_cable'], cv2.MORPH_CLOSE, np.ones((p['brown_close_k'], p['brown_close_k'])))
    new_masks['brown_cable'] = cv2.morphologyEx(new_masks['brown_cable'], cv2.MORPH_OPEN, np.ones((p['brown_open_k'], p['brown_open_k'])))
    new_masks['brown_cable'] = clean_up_mask(new_masks['brown_cable'], p['brown_cleanup_params'])

    return new_masks

def prepare_golden_masks(train_dir="./data/train/good", config: dict = CONFIG):
    print("Preparing Golden Masks")
    
    accumulators = {}
    image_count = 0

    for root, dirs, files in os.walk(train_dir):
        for file in files:
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            img_path = os.path.join(root, file)
            in_img_bgr = cv2.imread(img_path)
            
            if in_img_bgr is None:
                continue

            masks = make_raw_masks(in_img_bgr, config)
            masks = process_masks(masks, config)
            masks = combine_masks(masks, config)

            if image_count == 0:
                for key in masks.keys():
                    accumulators[key] = np.zeros(masks[key].shape, dtype=np.float32)

            for key, mask in masks.items():
                accumulators[key] += (mask.astype(np.float32) / 255.0)

            image_count += 1
            print(f"Processed {image_count} good images...", end="\r")

    print(f"\nFinished processing {image_count} training images.")
    
    if image_count == 0:
        print("No images found to process!")
        return {}

    golden_masks = {}
    for key, acc in accumulators.items():
        probability_map = acc / image_count
        golden_mask = (probability_map*255).astype(np.uint8)
        golden_masks[key] = golden_mask
        cv2.imwrite(f"masks/mask_{key}.png", golden_masks[key])

    print("Golden masks successfully generated and saved to disk.")
    return golden_masks

if __name__ == "__main__":
    IN_DEPTH_MODE = True 

    test_base_dir = "./data/test"
    print(f"\nStarting Evaluation... (In-Depth Mode: {IN_DEPTH_MODE})")
    
    total_pixel_metrics = {"Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0, "F1-Score": 0.0, "IoU": 0.0}
    total_img_TP, total_img_TN, total_img_FP, total_img_FN = 0, 0, 0, 0
    total_images_processed = 0

    for folder_name in os.listdir(test_base_dir):
        folder_path = os.path.join(test_base_dir, folder_name)
        
        if not os.path.isdir(folder_path):
            continue
            
        print(f"\n--- Testing Category: {folder_name} ---")
        
        cat_pixel_metrics = {"Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0, "F1-Score": 0.0, "IoU": 0.0}
        cat_img_TP, cat_img_TN, cat_img_FP, cat_img_FN = 0, 0, 0, 0
        cat_images_processed = 0
        
        for file in os.listdir(folder_path):
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            file_path = os.path.join(folder_path, file)
            if IN_DEPTH_MODE:
                print(f"Processing: {file_path}")
            
            in_img_bgr = cv2.imread(file_path)
            if in_img_bgr is None: continue
            
            in_img_rgb = cv2.cvtColor(in_img_bgr, cv2.COLOR_BGR2RGB)
            
            if folder_name.lower() == 'good':
                gt_img = np.zeros(shape=(in_img_bgr.shape[0], in_img_bgr.shape[1]), dtype=np.uint8)
            else:
                gt_path = file_path.replace("test", "ground_truth").replace(".png", "_mask.png").replace(".jpg", "_mask.png")
                gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                if gt_img is None:
                    gt_img = np.zeros(shape=(in_img_bgr.shape[0], in_img_bgr.shape[1]), dtype=np.uint8)

            pred_mask = predict(in_img_rgb, CONFIG, IN_DEPTH_MODE)
            
            metrics = validate_prediction(pred_mask, gt_img)
            for k in total_pixel_metrics.keys():
                cat_pixel_metrics[k] += metrics[k]
                total_pixel_metrics[k] += metrics[k]

            pred_is_defective = cv2.countNonZero(pred_mask) > CONFIG['evaluation']['min_defect_pixels_to_alarm']
            gt_is_defective = cv2.countNonZero(gt_img) > 0
            
            if pred_is_defective and gt_is_defective:
                cat_img_TP += 1; total_img_TP += 1
            elif not pred_is_defective and not gt_is_defective:
                cat_img_TN += 1; total_img_TN += 1
            elif pred_is_defective and not gt_is_defective:
                cat_img_FP += 1; total_img_FP += 1
            elif not pred_is_defective and gt_is_defective:
                cat_img_FN += 1; total_img_FN += 1

            cat_images_processed += 1
            total_images_processed += 1

            if IN_DEPTH_MODE:
                print(f"  Pixel Metrics: {{k: f'{{v:.2f}}' for k, v in metrics.items()}}")
                print(f"  Image Alarm: {'DEFECT' if pred_is_defective else 'GOOD'} | Ground Truth: {'DEFECT' if gt_is_defective else 'GOOD'}")
                
                win_input, win_gt, win_pred = "1. Input", "2. Ground Truth", "3. Prediction Mask"
                cv2.namedWindow(win_input, cv2.WINDOW_NORMAL)
                cv2.namedWindow(win_gt, cv2.WINDOW_NORMAL)
                cv2.namedWindow(win_pred, cv2.WINDOW_NORMAL)
                
                cv2.resizeWindow(win_input, 512, 512)
                cv2.resizeWindow(win_gt, 512, 512)
                cv2.resizeWindow(win_pred, 512, 512)

                cv2.imshow(win_input, in_img_bgr)
                cv2.imshow(win_gt, gt_img)
                cv2.imshow(win_pred, pred_mask)

                print("Press any key to load the next image. Press 'n' to skip folder. Press 'ESC' to stop.")
                key = cv2.waitKey(0)
                if key == ord('n'): break 
                if key == 27: 
                    cv2.destroyAllWindows()
                    exit()

        if cat_images_processed > 0:
            print(f"\n[{folder_name}] - Category Results ({cat_images_processed} images):")
            print(f"  Image-Level Detections -> TP: {cat_img_TP} | TN: {cat_img_TN} | FP: {cat_img_FP} | FN: {cat_img_FN}")
            cat_img_acc = (cat_img_TP + cat_img_TN) / cat_images_processed
            print(f"  Image-Level Accuracy:     {cat_img_acc * 100:.1f}%")
            
            print(f"  Pixel-Level Averages   -> ", end="")
            for k, v in cat_pixel_metrics.items():
                print(f"{k}: {(v / cat_images_processed):.3f} | ", end="")
            print("")

    cv2.destroyAllWindows()

    if not IN_DEPTH_MODE and total_images_processed > 0:
        print("\n========================================================")
        print(f"               FINAL OVERALL DATASET RESULTS            ")
        print("========================================================")
        print(f"Total Images Analyzed: {total_images_processed}")
        print(f"Overall Image-Level:   TP:{total_img_TP}  TN:{total_img_TN}  FP:{total_img_FP}  FN:{total_img_FN}")
        
        overall_img_acc = (total_img_TP + total_img_TN) / total_images_processed
        overall_img_prec = total_img_TP / (total_img_TP + total_img_FP) if (total_img_TP + total_img_FP) > 0 else 0.0
        overall_img_rec = total_img_TP / (total_img_TP + total_img_FN) if (total_img_TP + total_img_FN) > 0 else 0.0
        
        print(f"Image-Level Accuracy:  {overall_img_acc * 100:.2f}%")
        print(f"Image-Level Precision: {overall_img_prec * 100:.2f}%")
        print(f"Image-Level Recall:    {overall_img_rec * 100:.2f}%")
        print("-" * 56)
        print("Average Pixel Metrics:")
        for k, v in total_pixel_metrics.items():
            print(f"  {k}: {(v / total_images_processed):.4f}")
        print("========================================================")