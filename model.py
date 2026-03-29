import cv2
import numpy as np
import os
import numpy as np
# Global list to hold pre-processed (256x256, grayscale, blurred) exemplars
EXEMPLARS = []

mask_ranges = {
    'glare': [(0,255),(0,55),(191,255)],
    'brown_cable': [(0,110),(0,48),(20,128)],
    'blue_cable':[(97,120),(114,255),(0,255)],
    'yellow_cable':[(29,75),(49,155),(48,206)],
    'insulation':[(75,103),(36,113),(71,255)],
    'black':[(0,255),(21,85),(0,25)],
    'copper':[(0,24),(63,105),(30,255)]
}

# min area, min aspect_ratio min extent
mask_filters = {
    'glare': [0,0,0],
    'brown_cable': [2000,0,0],
    'blue_cable':[2000,0,0],
    'yellow_cable':[2000,0,0],
    'insulation':[20000,0,0],
    'black':[5000,0.7,0.6],
    'copper':[0,0,0]
}
# must be, can be
matching_thr = {
    'brown_cable': [250,30],
    'blue_cable':[250,30],
    'yellow_cable':[250,30],
    'insulation':[220,30],
    'wire':[250,30],
    'black':[150,30]
}

CACHED_GOLDEN_MASKS = {}

def predict(image: np.ndarray) -> np.ndarray:
    """
    Args:
        image: tablica NumPy, kształt (H, W, 3), dtype uint8, RGB
        confidence_threshold: Float (0.0 to 1.0). How strict the golden mask should be.

    Returns:
        Maska binarna (H, W), dtype uint8. 255 = wada, 0 = brak wady.
    """
    global CACHED_GOLDEN_MASKS
    H, W = image.shape[:2]
    defect_mask = np.zeros((H, W), dtype=np.uint8)

    mask_names = ['insulation', 'blue_cable', 'yellow_cable', 'brown_cable', 'wire', 'black']
    
    if not CACHED_GOLDEN_MASKS:
        for name in mask_names:
            path = f"masks/mask_{name}.png"
            if os.path.exists(path):
                CACHED_GOLDEN_MASKS[name] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            else:
                print(f"Warning: {path} not found! Run prepare_golden_masks() first.")
                return defect_mask

    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    raw_masks = make_raw_masks(bgr_image)
    proc_masks = process_masks(raw_masks)
    current_masks = combine_masks(proc_masks)

    errors = {}

    for key in current_masks:
        # Grab the current mask
        c_mask = current_masks[key]
        g_mask = CACHED_GOLDEN_MASKS[key]

        # Define explicitly as uint8 so OpenCV doesn't complain later
        g_mask_must = np.zeros(c_mask.shape, dtype=np.uint8)
        g_mask_must[g_mask > matching_thr[key][0]] = 255

        g_mask_can = np.zeros(c_mask.shape, dtype=np.uint8)
        g_mask_can[g_mask > matching_thr[key][1]] = 255
        
        err = (g_mask_must == 255) & (c_mask == 0)
        err |= (c_mask == 255) & (g_mask_can == 0)
        
        err_img = (err.astype(np.uint8) * 255)
        
        errors[key] = err_img
            
        # Resize them to 256x256 for display
        c_disp = cv2.resize(c_mask, (256, 256))
        g_disp = cv2.resize(g_mask, (256, 256))
        e_disp = cv2.resize(err_img, (256, 256)) # Use the uint8 version
        
        # Stack them horizontally: Golden | Current | Err
        comparison_img = np.hstack((g_disp, c_disp, e_disp))
        
        # Display the combined image
        win_name = f"Compare [Golden | Current | Err]: {key}"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 256*3, 256) # 768 wide, 256 tall
        cv2.imshow(win_name, comparison_img)

    return defect_mask

def validate_prediction(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    """Calculates segmentation metrics between a prediction and a ground truth mask."""
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)

    TP = np.sum((pred == 1) & (gt == 1))
    TN = np.sum((pred == 0) & (gt == 0))
    FP = np.sum((pred == 1) & (gt == 0))
    FN = np.sum((pred == 0) & (gt == 1))

    def safe_divide(numerator, denominator):
        return float(numerator) / float(denominator) if denominator > 0 else 0.0

    return {
        "Accuracy": safe_divide(TP + TN, TP + TN + FP + FN),
        "Precision": safe_divide(TP, TP + FP),
        "Recall": safe_divide(TP, TP + FN),
        "F1-Score": safe_divide(2 * TP, 2 * TP + FP + FN),
        "IoU": safe_divide(TP, TP + FP + FN)
    }

def clean_up_mask(mask,params,size_override = 0):
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

def make_raw_masks(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    W,H = image.shape[:2]
    area_of_intrest = np.zeros((H, W), dtype=np.uint8)

    center = (W // 2, H // 2)
    radius = 400

    cv2.circle(area_of_intrest, center, radius, 255, -1)

    masks = {}
    for key,values in mask_ranges.items():
        lower_bound = np.array([r[0] for r in values])
        upper_bound = np.array([r[1] for r in values])
        masks[key] = cv2.inRange(hsv, lower_bound, upper_bound) & area_of_intrest
    
    return masks

def process_masks(masks):
    new_masks = {}
    for key,mask in masks.items():
        new_mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3)),iterations=3)
        new_masks[key] = clean_up_mask(new_mask,mask_filters[key])

    return new_masks

def combine_masks(masks):
    new_masks = {}
    already_verified = masks['insulation'].copy()
    new_masks['insulation'] = masks['insulation']

    new_masks['blue_cable'] = masks['blue_cable'] &~ already_verified
    already_verified |= new_masks['blue_cable']

    new_masks['yellow_cable'] = masks['yellow_cable'] &~ already_verified
    already_verified |= new_masks['yellow_cable']

    new_masks['black'] = masks['black'] &~ already_verified
    already_verified |= new_masks['black']

    new_masks['wire'] = (masks['copper'] | masks['glare']) &~ already_verified
    
    
    new_masks['wire'] = cv2.morphologyEx(new_masks['wire'],cv2.MORPH_CLOSE, np.ones((3,3)))
    new_masks['wire'] = cv2.morphologyEx(new_masks['wire'],cv2.MORPH_OPEN, np.ones((7,7)))
    new_masks['wire'] = cv2.morphologyEx(new_masks['wire'],cv2.MORPH_DILATE, np.ones((11,11)),iterations=2)
    new_masks['wire'] = cv2.morphologyEx(new_masks['wire'],cv2.MORPH_CLOSE, np.ones((3,3)))
    new_masks['wire'] = clean_up_mask(new_masks['wire'],[10000,0.6,0.5],size_override = 25000)

    already_verified |= new_masks['wire']

    new_masks['brown_cable'] = masks['brown_cable'] &~ already_verified
    new_masks['brown_cable'] = cv2.morphologyEx(new_masks['brown_cable'],cv2.MORPH_CLOSE, np.ones((3,3)))
    new_masks['brown_cable'] = cv2.morphologyEx(new_masks['brown_cable'],cv2.MORPH_OPEN, np.ones((13,13)))
    new_masks['brown_cable'] = clean_up_mask(new_masks['brown_cable'],[5000,0,0])

    return new_masks


def prepare_golden_masks(train_dir="./data/train/good", confidence_threshold=0.5):
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

            masks = make_raw_masks(in_img_bgr)
            masks = process_masks(masks)
            masks = combine_masks(masks)

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

    # --- 2. Iterate through test folders ---
    test_base_dir = "./data/test"
    print("\nStarting Evaluation...")
    
    for folder_name in os.listdir(test_base_dir):
        folder_path = os.path.join(test_base_dir, folder_name)
        
        # Only process if it's actually a directory
        if not os.path.isdir(folder_path):
            continue
            
        print(f"\n--- Testing Category: {folder_name} ---")
        
        for file in os.listdir(folder_path):
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            file_path = os.path.join(folder_path, file)
            print(f"Processing: {file_path}")
            
            # Load images
            in_img_bgr = cv2.imread(file_path)
            
            # Predict function expects RGB
            in_img_rgb = cv2.cvtColor(in_img_bgr, cv2.COLOR_BGR2RGB)
            
            # Find Ground Truth
            gt_path = file_path.replace("test", "ground_truth").replace(".png", "_mask.png").replace(".jpg", "_mask.png")
            gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            
            if in_img_bgr is None:
                print(f"Warning: Could not read {file_path}. Skipping.")
                continue
                
            if gt_img is None:
                print(f"Warning: Missing ground truth mask at {gt_path}. Using blank mask for evaluation.")
                gt_img = np.zeros(shape=(in_img_bgr.shape[0], in_img_bgr.shape[1]), dtype=np.uint8)

            # Run prediction
            pred_mask = predict(in_img_rgb)
            
            # Validate metrics
            metrics = validate_prediction(pred_mask, gt_img)
            print("Metrics:", {k: f"{v:.2f}" for k, v in metrics.items()})
            
            # Define window names
            win_input = "1. Input Image"
            win_gt = "3. Ground Truth"
            win_pred = "4. Prediction Mask"
            
            # Create windows with the 'NORMAL' flag so they can be resized
            cv2.namedWindow(win_input, cv2.WINDOW_NORMAL)
            cv2.namedWindow(win_gt, cv2.WINDOW_NORMAL)
            cv2.namedWindow(win_pred, cv2.WINDOW_NORMAL)
            
            # Force the windows to open at a specific smaller size (e.g., 512x512)
            display_size = (512, 512)
            cv2.resizeWindow(win_input, display_size[0], display_size[1])
            cv2.resizeWindow(win_gt, display_size[0], display_size[1])
            cv2.resizeWindow(win_pred, display_size[0], display_size[1])

            # Now display the images in those pre-sized windows
            cv2.imshow(win_input, in_img_bgr)
            cv2.imshow(win_gt, gt_img)
            cv2.imshow(win_pred, pred_mask)

            print("Press any key to load the next image. Press 'ESC' to stop.")
            key = cv2.waitKey(0)
            if key == 27: # ESC key
                cv2.destroyAllWindows()
                exit()

    cv2.destroyAllWindows()