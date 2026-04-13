import copy
import json
import optuna
import cv2
import numpy as np
import os
from model import predict, validate_prediction

import copy
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

def evaluate_dataset(config: dict, test_base_dir: str = "./data/test") -> dict:
    """Runs the full test dataset using the provided config and returns overall metrics."""
    total_pixel_metrics = {"Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0, "F1-Score": 0.0, "IoU": 0.0}
    total_images_processed = 0

    for folder_name in os.listdir(test_base_dir):
        folder_path = os.path.join(test_base_dir, folder_name)
        if not os.path.isdir(folder_path): continue
            
        for file in os.listdir(folder_path):
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')): continue
                
            file_path = os.path.join(folder_path, file)
            in_img_bgr = cv2.imread(file_path)
            if in_img_bgr is None: continue
            
            in_img_rgb = cv2.cvtColor(in_img_bgr, cv2.COLOR_BGR2RGB)
            
            # Ground Truth Handling
            if folder_name.lower() == 'good':
                gt_img = np.zeros(shape=(in_img_bgr.shape[0], in_img_bgr.shape[1]), dtype=np.uint8)
            else:
                gt_path = file_path.replace("test", "ground_truth").replace(".png", "_mask.png").replace(".jpg", "_mask.png")
                gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                if gt_img is None:
                    gt_img = np.zeros(shape=(in_img_bgr.shape[0], in_img_bgr.shape[1]), dtype=np.uint8)

            # Predict (Disable debug mode for speed)
            pred_mask = predict(in_img_rgb, config, debug=False)
            
            metrics = validate_prediction(pred_mask, gt_img)
            for k in total_pixel_metrics.keys():
                total_pixel_metrics[k] += metrics[k]

            total_images_processed += 1

    if total_images_processed == 0:
        return total_pixel_metrics

    # Average the metrics
    for k in total_pixel_metrics.keys():
        total_pixel_metrics[k] /= total_images_processed

    return total_pixel_metrics

def objective(trial):
    """Optuna objective function for the final refinement round."""
    trial_config = copy.deepcopy(CONFIG)
    
    # ---------------------------------------------------------
    # 1. Donut Geometry & Fit Strictness
    # ---------------------------------------------------------
    # Tolerances for what constitutes an acceptable "donut" shape
    trial_config['geometry']['donut_max_aspect'] = trial.suggest_float('donut_max_aspect', 1.1, 1.8)
    trial_config['geometry']['donut_area_ratio'] = trial.suggest_float('donut_area_ratio', 0.05, 0.25)
    
    # ---------------------------------------------------------
    # 2. Defect Blob Cleanup Parameters (Area, Aspect Ratio, Extent)
    # ---------------------------------------------------------
    # Tuning the minimum size and shape a defect must be to trigger an alarm
    trial_config['predict']['donut_err_cleanup'][0] = trial.suggest_int('donut_err_min_area', 300, 2500, step=100)
    trial_config['predict']['donut_err_cleanup'][1] = trial.suggest_float('donut_err_min_aspect', 0.1, 0.8)
    trial_config['predict']['donut_err_cleanup'][2] = trial.suggest_float('donut_err_min_extent', 0.1, 0.8)
    
    trial_config['predict']['binary_err_cleanup'][0] = trial.suggest_int('binary_err_min_area', 300, 2500, step=100)

    # ---------------------------------------------------------
    # 3. Golden Mask Matching Thresholds
    # ---------------------------------------------------------
    # How closely the cables and insulation must match the cached golden models
    # We group the color cables together to keep the search space manageable
    cable_must_thr = trial.suggest_int('cable_must_thr', 120, 200, step=10)
    cable_can_thr = trial.suggest_int('cable_can_thr', 10, 80, step=10)
    
    for key in ['blue_cable', 'yellow_cable', 'brown_cable', 'black']:
        trial_config['matching_thr'][key][0] = cable_must_thr
        trial_config['matching_thr'][key][1] = cable_can_thr
        
    trial_config['matching_thr']['insulation'][0] = trial.suggest_int('insul_must_thr', 180, 240, step=10)
    trial_config['matching_thr']['insulation'][1] = trial.suggest_int('insul_can_thr', 10, 60, step=10)

    # Run Evaluation
    metrics = evaluate_dataset(trial_config, test_base_dir="./data/test")
    
    # Display Current Run Metrics
    print(f"Trial {trial.number:03d} | Acc: {metrics['Accuracy']:.4f} | "
          f"Prec: {metrics['Precision']:.4f} | Rec: {metrics['Recall']:.4f} | "
          f"F1: {metrics['F1-Score']:.4f} | IoU: {metrics['IoU']:.4f}")
    
    return metrics['IoU']

def save_best_config_callback(study, trial):
    """Callback triggered after every trial. Saves to disk if it's the best so far."""
    if study.best_trial.number == trial.number:
        print(f"--> New Best IoU Achieved: {study.best_value:.4f}! Saving to best_config.json")
        with open("best_config.json", "w") as f:
            json.dump(study.best_trial.params, f, indent=4)
            
            
if __name__ == "__main__":
    import logging
    # Disable optuna's default verbose logging to keep your terminal clean
    optuna.logging.set_verbosity(optuna.logging.WARNING) 
    
    print("========================================================")
    print("          STARTING HYPERPARAMETER OPTIMIZATION          ")
    print("========================================================")
    
    # Create the study object. direction='maximize' because we want the highest IoU
    study = optuna.create_study(direction="maximize")
    
    try:
        # n_trials is the total number of evaluations to run. 
        # You can interrupt it safely with Ctrl+C at any time.
        study.optimize(objective, n_trials=200, callbacks=[save_best_config_callback])
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
        
    print("\n========================================================")
    print("                    OPTIMIZATION FINISHED               ")
    print("========================================================")
    print(f"Best IoU: {study.best_value:.4f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")