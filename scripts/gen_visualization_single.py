import sys
sys.path.append("..")
import torch
import cv2
import argparse
import einops
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as TF

import math
import PIL.Image as Image
from PIL import ImageDraw
import os
import matplotlib.pyplot as plt
from matplotlib import cm

from src.data.transforms import transforms_imagenet_train, transforms_imagenet_eval, ImageFolderWithEntropy
from src.models.entropy_utils import (
    compute_patch_entropy_vectorized, 
    compute_patch_laplacian_vectorized,
    compute_patch_mse_batched,
    select_patches_by_threshold, 
    visualize_selected_patches_cv2_non_overlapping
)

# Default parameters
IMAGE_SIZE = 336
BASE_PATCH_SIZE = 14
NUM_SCALES = 3
THRESHOLDS = [6.0, 6.0]
LINE_COLOR = (255, 255, 255)  # White color
LINE_THICKNESS = 1
OUTPUT_PATH = "../assets/vis_single.jpg"
OUTPUT_DIR = "../assets"

def parse_args():
    parser = argparse.ArgumentParser(description='Generate image visualizations with different options')
    parser.add_argument('--input', type=str, required=True, help='Input image path')
    parser.add_argument('--vis_type', type=str, default='grid', choices=['entropy', 'grid', 'none'], 
                        help='Visualization type: entropy-based, standard grid, or no lines')
    parser.add_argument('--method', type=str, default='entropy', choices=['entropy', 'laplacian', 'upsample_mse'],
                        help='Method for computing importance maps: entropy, laplacian, or upsample_mse')
    parser.add_argument('--aggregate', type=str, default='mean', choices=['mean', 'max', 'std'],
                        help='Aggregation method for laplacian and upsample_mse (default: mean)')
    parser.add_argument('--image_size', type=int, default=IMAGE_SIZE, help='Target image size for the shorter side')
    parser.add_argument('--patch_size', type=int, default=BASE_PATCH_SIZE, help='Base patch size')
    parser.add_argument('--num_scales', type=int, default=NUM_SCALES, help='Number of scales for entropy calculation')
    parser.add_argument('--thresholds', type=float, nargs='+', default=THRESHOLDS, 
                        help='Thresholds for entropy-based patch selection')
    parser.add_argument('--line_color', type=int, nargs=3, default=list(LINE_COLOR), 
                        help='Line color in RGB format (3 values from 0-255)')
    parser.add_argument('--line_thickness', type=int, default=LINE_THICKNESS, 
                        help='Line thickness (integer value)')
    parser.add_argument('--output_prefix', type=str, default='entropy_map',
                        help='Prefix for importance map output files (will be saved as prefix_l1.jpg, prefix_l2.jpg, etc.)')
    parser.add_argument('--output', type=str, default=OUTPUT_PATH,
                        help='Output path for the final visualization image (default: ../assets/vis_single.jpg)')
    parser.add_argument('--no_resize', action='store_true',
                        help='Skip image resizing and use original image size')
    
    return parser.parse_args()

def save_entropy_map_as_heatmap(entropy_map, output_path, original_size):
    """Save entropy map as a matplotlib heatmap resized to the original image size."""
    # Convert entropy map to numpy array
    entropy_np = entropy_map.cpu().numpy()
    
    # Create figure without axes
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    
    # Create heatmap with inferno colormap
    plt.imshow(entropy_np, cmap='inferno')
    plt.tight_layout(pad=0)
    
    # Save the figure
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Resize the saved image to match the original image size
    saved_img = Image.open(output_path)
    saved_img = saved_img.resize(original_size, Image.LANCZOS)
    saved_img.save(output_path)


def process_image(image_path, vis_type, method, aggregate, image_size, patch_size, num_scales, thresholds, line_color, line_thickness, output_prefix, no_resize=False):
    """Process a single image and return the visualization based on the specified type and method."""
    print(f"Processing image: {image_path}")
    
    # Open the image
    image = Image.open(image_path)
    print(f"Original image size: {image.size}")
    
    if no_resize:
        # Use original image without resizing
        img = image
        print("Using original image size (no resize)")
    else:
        # Find the image size
        width, height = image.size

        # Calculate the target size for the shorter side
        # Make the shorter side always image_size
        if width < height:
            new_width = image_size
            new_height = int(height * (image_size / width))
        else:
            new_height = image_size
            new_width = int(width * (image_size / height))

        # Adjust the longer side to be a multiple of patch_size * 4
        if width > height:
            new_width = (new_width // (patch_size * 4)) * (patch_size * 4)
        else:
            new_height = (new_height // (patch_size * 4)) * (patch_size * 4)

        # Resize the image
        img = image.resize((new_width, new_height))
        print(f"Resized image size: {img.size}")

    # Load the image as numpy array
    np_img = np.array(img)
    
    # If visualization type is 'none', return the resized image without any lines
    if vis_type == 'none':
        return img
    
    # Get image dimensions
    height, width = np_img.shape[:2]
    
    if vis_type == 'grid':
        # Draw a simple grid on the image
        image_with_grid = np_img.copy()
        
        # Draw vertical lines
        for x in range(0, width, patch_size):
            cv2.line(image_with_grid, (x, 0), (x, height), tuple(line_color), line_thickness)

        # Draw horizontal lines
        for y in range(0, height, patch_size):
            cv2.line(image_with_grid, (0, y), (width, y), tuple(line_color), line_thickness)
        
        # Convert back to PIL image
        return Image.fromarray(image_with_grid)
    
    elif vis_type == 'entropy':
        # Convert image to tensor for importance map calculation
        img_tensor = TF.to_tensor(img) * 255.0
        img_int_tensor = img_tensor.to(torch.uint8)
        
        # Compute importance maps based on selected method
        if method == 'entropy':
            importance_maps = compute_patch_entropy_vectorized(img_tensor, patch_size, num_scales)
            map_name = 'entropy'
        elif method == 'laplacian':
            importance_maps = compute_patch_laplacian_vectorized(img_tensor, patch_size, num_scales, aggregate=aggregate)
            map_name = f'laplacian_{aggregate}'
        elif method == 'upsample_mse':
            # upsample_mse only has batched version, so add batch dimension
            img_batched = img_tensor.unsqueeze(0)  # Add batch dimension
            importance_maps_batched = compute_patch_mse_batched(img_batched, patch_size, num_scales, aggregate=aggregate)
            # Remove batch dimension from results
            importance_maps = {k: v.squeeze(0) for k, v in importance_maps_batched.items()}
            map_name = f'upsample_mse_{aggregate}'
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Save importance maps as heatmaps
        for scale, importance_map in importance_maps.items():
            output_path = os.path.join(OUTPUT_DIR, f"{output_prefix}_{scale}.jpg")
            save_entropy_map_as_heatmap(importance_map, output_path, (width, height))
            print(f"Saved {map_name} map for scale {scale} to {output_path}")
        
        # Prepare importance maps for visualization
        for k, v in importance_maps.items():
            importance_maps[k] = v.unsqueeze(0)

        # Select patches based on threshold
        all_masks = select_patches_by_threshold(importance_maps, thresholds)

        # Generate patch sizes list
        patch_sizes = [patch_size * (2**i) for i in range(num_scales)]
        
        # Prepare masks for visualization (remove batch dimension)
        visualization_masks = {}
        for scale, mask in all_masks.items():
            visualization_masks[scale] = mask.squeeze(0)
            
        # Draw patches on the image using cv2
        vis_img = visualize_selected_patches_cv2_non_overlapping(
            image_tensor=img_tensor,
            masks=visualization_masks,
            patch_sizes=patch_sizes,
            color=tuple(line_color),
            thickness=line_thickness
        )
        
        return vis_img

def main():
    args = parse_args()
    
    # Validate thresholds length
    if len(args.thresholds) != args.num_scales - 1:
        print(f"Warning: Number of thresholds ({len(args.thresholds)}) does not match required number ({args.num_scales - 1})")
        print(f"Using default thresholds: {THRESHOLDS}")
        args.thresholds = THRESHOLDS
    
    # Process the image
    result_image = process_image(
        args.input, 
        args.vis_type,
        args.method,
        args.aggregate,
        args.image_size, 
        args.patch_size, 
        args.num_scales, 
        args.thresholds, 
        args.line_color, 
        args.line_thickness,
        args.output_prefix,
        args.no_resize
    )
    
    # Ensure the output directory exists
    output_path = args.output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the result
    result_image.save(output_path)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    main()
