# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 15:13:36 2024

@author: olivi
"""
import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from skimage.feature import local_binary_pattern
from scipy.stats import entropy


def var_green_transformation(image):
    if image.shape[2] == 3:  # Ensure the image has three channels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    numerator = G - R
    denominator = G + R + np.finfo(float).eps  # Avoid division by zero with epsilon
    VARgreen = numerator / denominator
    VARgreen = (VARgreen + 1) / 2
    return VARgreen


def calculate_brightness(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Compute average brightness
    brightness = np.mean(gray)
    return brightness

def calculate_canny_edges(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)
    # Calculate average edge value
    canny_value = np.mean(edges)
    return canny_value / 255  # Normalize to range [0,1]

def compute_tgdi(image):
    # Calculate brightness and Canny values
    brightness = calculate_brightness(image)
    canny = calculate_canny_edges(image)
    
    # Compute TGDI
    if canny == 0:
        return float('inf')  # Handle zero Canny value to avoid division by zero in log
    tgdi = -np.log10(canny) * brightness
    return tgdi

def apply_gaussian_perpendicular(image, center, direction, kernel_size, sigma):
    kernel = np.linspace(-(kernel_size - 1) // 2, (kernel_size - 1) // 2, kernel_size)
    kernel = np.exp(-0.5 * (kernel / sigma) ** 2)
    kernel /= kernel.sum()
    metrics = []
    perp_direction = np.array([-direction[1], direction[0]])  # Perpendicular direction
    
    min_x = center[0] + perp_direction[0] * (-(kernel_size - 1) // 2)
    max_x = center[0] + perp_direction[0] * ((kernel_size - 1) // 2)
    min_y = center[1] + perp_direction[1] * (-(kernel_size - 1) // 2)
    max_y = center[1] + perp_direction[1] * ((kernel_size - 1) // 2)
    
    if (0 <= min_x < image.shape[1] and 0 <= max_x < image.shape[1] and
        0 <= min_y < image.shape[0] and 0 <= max_y < image.shape[0]):
        for offset in range(-(kernel_size // 2), kernel_size // 2 + 1):
            sample_x = int(center[0] + perp_direction[0] * offset)
            sample_y = int(center[1] + perp_direction[1] * offset)
            metrics.append(image[sample_y, sample_x] * kernel[offset + kernel_size // 2])
        return sum(metrics)
    return None  # Return None if the area is not fully within the image

def process_power_line(transformed_image, original_image, xywhr):
    x, y, width, height, rotation = xywhr
    dx = np.cos(rotation)
    dy = np.sin(rotation)
    line_length = int(width)
    line_points = []
    step_size = 4  # Step every four pixels along the line
    
    num_steps = line_length // step_size
    for step in range(-num_steps // 2, num_steps // 2 + 1):
        point_x = int(x + dx * step * step_size)
        point_y = int(y + dy * step * step_size)
        line_points.append((point_x, point_y))

    vegetation_metrics = []
    brg_metrics = []
    edge_metrics = []
    tgdi_metrics = []
    for point in line_points:
        metric = apply_gaussian_perpendicular(transformed_image, point, (dx, dy), 101, 50)  # 101 pixels for 50 on each side + center
        if metric is not None:
            vegetation_metrics.append(metric)
            
        # Extract 100x100 patches for LBP and edge density from the original image
        half_patch_size = 50
        patch_x_start = max(point[0] - half_patch_size, 0)
        patch_y_start = max(point[1] - half_patch_size, 0)
        patch_x_end = min(point[0] + half_patch_size, original_image.shape[1])
        patch_y_end = min(point[1] + half_patch_size, original_image.shape[0])
        
        patch = original_image[patch_y_start:patch_y_end, patch_x_start:patch_x_end]
        
        if patch.shape[0] == 100 and patch.shape[1] == 100:
            brg_val = calculate_brightness(patch)
            edge_val = calculate_canny_edges(patch)
            tgdi_val = compute_tgdi(patch)
            brg_metrics.append(brg_val)
            edge_metrics.append(edge_val)
            tgdi_metrics.append(tgdi_val)
    
    return line_points, vegetation_metrics, brg_metrics, edge_metrics, tgdi_metrics

def remove_outliers(metrics):
    if len(metrics) == 0:
        return metrics
    Q1 = np.percentile(metrics, 25)
    Q3 = np.percentile(metrics, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return [x for x in metrics if lower_bound <= x <= upper_bound]

# Load the model
params = r"path/to/your/destination/folder/best.pt"
model = YOLO(params)

image_folder = r"path/to/your/destination/folder"
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]

# Data storage for CSV files
data_frames = {
    'VarGreen': []
}

for image_path in image_files:
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transformations = {
        'VarGreen': var_green_transformation(image)
    }
    
    results = model([image])
    
    # Initialize metrics storage for the current image
    combined_veg_metrics = []
    combined_brg_metrics = []
    combined_edge_metrics = []
    combined_tgdi_metrics = []

    for transformation_name, transformed_image in transformations.items():
        for result in results:
            obb = result.obb
            if obb.xywhr.size(0) > 0:
                for i in range(obb.xywhr.size(0)):
                    xywhr = obb.xywhr[i].cpu().numpy()
                    line_points, veg_metrics, brg_metrics, edge_metrics, tgdi_metrics = process_power_line(transformed_image, image, xywhr)
                    
                    # Accumulate metrics for all detected lines in the image
                    combined_veg_metrics.extend(veg_metrics)
                    combined_brg_metrics.extend(brg_metrics)
                    combined_edge_metrics.extend(edge_metrics)
                    combined_tgdi_metrics.extend(tgdi_metrics)

        # Remove outliers and compute statistics once all lines are processed
        filtered_veg_metrics = remove_outliers(combined_veg_metrics)
        filtered_brg_metrics = remove_outliers(combined_brg_metrics)
        filtered_edge_metrics = remove_outliers(combined_edge_metrics)
        filtered_tgdi_metrics = remove_outliers(combined_tgdi_metrics)

        if filtered_veg_metrics:
            # Compute aggregated statistics here
            # You can add more statistics as required
            mean_val = np.mean(filtered_veg_metrics)
            std_val = np.std(filtered_veg_metrics)
            median_val = np.median(filtered_veg_metrics)
            max_val = np.max(filtered_veg_metrics)
            percentile_75_val = np.percentile(filtered_veg_metrics, 75)
            percentile_90_val = np.percentile(filtered_veg_metrics, 90)
            comp_metric = 0.5 * max_val + 0.5 * median_val
            avg_tgdi = np.mean(filtered_tgdi_metrics) if filtered_tgdi_metrics else 0
            avg_brg = np.mean(filtered_brg_metrics) if filtered_brg_metrics else 0
            avg_edge = np.mean(filtered_edge_metrics) if filtered_edge_metrics else 0
            tgdi_imageLevel = compute_tgdi(image)

            # Add to data frame
            data_frames[transformation_name].append({
                'Image Path': image_path,
                'Mean': mean_val,
                'STD': std_val,
                'Percentile_75': percentile_75_val,
                'Percentile_90': percentile_90_val,
                'Median': median_val,
                'Max': max_val,
                'Composite Metric': comp_metric,
                'BRG': avg_brg,
                'Edge Density': avg_edge,
                'TGDI_image': tgdi_imageLevel,
                'VegMetric': 0.3526 * comp_metric - 0.8465 * tgdi_imageLevel #after the comp_metric and tgdi_imagelevel are normalized
            })

# Write data to CSV and generate statistics
for name, data in data_frames.items():
    df = pd.DataFrame(data)
    
    # Save CSV
    csv_path = f'{name}_metrics_brg_aggre.csv'
    df.to_csv(csv_path, index=False)
    
    # Calculate statistics
    stats = df.describe()
    stats_path = f'{name}_statistics_filtered.csv'
    stats.to_csv(stats_path)

print("Data statistics have been generated and saved.")

