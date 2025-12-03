import numpy as np
from scipy import signal
import tifffile
import os
import argparse
from tqdm import tqdm

def max_min_scale(img, MAX=65535, MIN=0):
    """
    Scale image to target range using percentiles to avoid outliers.
    """
    _max = np.percentile(img, 99.8)
    _min = np.percentile(img, 0.2)
    img_tmp = np.clip(img, _min, _max)
    _img_scale = MIN + (img_tmp - _min) * (MAX - MIN) / (_max - _min)
    return np.array(_img_scale, dtype='uint16')

def local_normalize_2d(im, half_size=30, bias=30.0):
    """
    Perform local contrast normalization on a 2D image.
    
    Formula:
        g*(x,y) = (g(x,y) - mean(x,y)) / (std(x,y) + bias)
        
    Args:
        im (ndarray): Input 2D image.
        half_size (int): Radius of the sliding window. Window size = 2 * half_size + 1.
        bias (float): Regularization parameter to avoid division by zero.
        
    Returns:
        ndarray: Normalized image (float).
    """
    # 1. Initial global scaling to [0, 1]
    _max = np.max(im)
    _min = np.min(im)
    if _max - _min == 0:
        return np.zeros_like(im, dtype=np.float32)
        
    im_norm = (im - _min) / (_max - _min)
    
    # 2. Local statistics using convolution
    full_size = 2 * half_size + 1
    kernel = np.ones([full_size, full_size]) / (full_size * full_size)
    
    # Calculate Local Mean
    im_mean = signal.convolve2d(im_norm, kernel, boundary='symm', mode='same')
    
    # Calculate Local Standard Deviation
    # std = sqrt(E[x^2] - (E[x])^2)
    im_square = np.square(im_norm)
    im_square_sum = signal.convolve2d(im_square, kernel, boundary='symm', mode='same')
    im_std = np.sqrt(np.maximum(0, im_square_sum - np.square(im_mean)) + bias)
    
    # 3. Normalization
    im_final = np.divide((im_norm - im_mean), im_std)
    
    return im_final

def process_image(img, nor_size=30, bias=0.0005, do_rescale=False):
    """
    Wrapper to process a single image with padding and rescaling (optionally).
    Args:
        img (ndarray): Input image.
        nor_size (int): Size for local normalization window (used for padding).
        bias (float): Bias parameter for normalization.
        do_rescale (bool): Whether to rescale normalized image to uint16 using max_min_scale.
    Returns:
        ndarray: Processed image (uint16 if do_rescale, else float32).
    """
    half = nor_size // 2

    # Pad image to handle boundaries
    img_padding = np.pad(img, ((half, half), (half, half)), 'symmetric')

    # Apply Local Normalization
    im_nor = local_normalize_2d(img_padding, half_size=10, bias=bias)

    # Crop back to original size
    im_nor_cropped = im_nor[half:-half, half:-half]

    if do_rescale:
        # Rescale to uint16 range for saving
        im_nor_cropped = max_min_scale(im_nor_cropped)
        return im_nor_cropped.astype(np.uint16)
    else:
        return im_nor_cropped.astype(np.float32)

def main():
    parser = argparse.ArgumentParser(description="Local Contrast Normalization for Microscopy Images")
    parser.add_argument('--input', type=str, required=True, help="Path to input TIFF file (single image or stack)")
    parser.add_argument('--output', type=str, required=True, help="Path to save output TIFF file")
    parser.add_argument('--radius', type=int, default=30, help="Radius for local normalization window (default: 30)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        return

    print(f"Loading {args.input}...")
    images = tifffile.imread(args.input)
    
    # Handle single image vs stack
    if images.ndim == 2:
        images = images[np.newaxis, :]
        is_single = True
    else:
        is_single = False
        
    num_images = images.shape[0]
    print(f"Processing {num_images} frames...")
    
    images_nor = np.zeros(images.shape, dtype='uint16')
    
    for i in tqdm(range(num_images)):
        images_nor[i] = process_image(images[i], nor_size=args.radius)
        
    if is_single:
        images_nor = images_nor[0]
        
    print(f"Saving to {args.output}...")
    tifffile.imwrite(args.output, images_nor)
    print("Done.")

if __name__ == '__main__':
    main()

