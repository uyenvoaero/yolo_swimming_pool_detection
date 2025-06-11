import os
import numpy as np
import streamlit as st
import xml.etree.ElementTree as ET
from glob import glob
import re
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

IMG_DIR = "./original_dataset/CANNES_TILES_512x512_PNG"
LABEL_DIR = "./original_dataset/CANNES_TILES_512x512_labels"

def pad_image(img, target_size=(512, 512)):
    if img.shape[:2] == target_size:
        return img
    padded_img = np.zeros((target_size[0], target_size[1], img.shape[2] if img.ndim == 3 else 1), dtype=img.dtype)
    h, w = img.shape[:2]
    padded_img[:h, :w] = img
    return padded_img

def pad_mask(mask, target_size=(512, 512)):
    if mask.shape == target_size:
        return mask
    padded_mask = np.zeros(target_size, dtype=np.uint8)
    h, w = mask.shape
    padded_mask[:h, :w] = mask
    return padded_mask

def parse_xml_label(xml_path):
    if xml_path is None or not os.path.exists(xml_path):
        return np.zeros((512, 512), dtype=np.uint8)
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        mask = np.zeros((512, 512), dtype=np.uint8)
        
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name == 'pool':
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                xmin, xmax = max(0, min(xmin, 511)), max(0, min(xmax, 511))
                ymin, ymax = max(0, min(ymin, 511)), max(0, min(ymax, 511))
                if xmax > xmin and ymax > ymin:
                    mask[ymin:ymax, xmin:xmax] = 1
        mask = pad_mask(mask)
        return mask
    except Exception:
        return np.zeros((512, 512), dtype=np.uint8)

def get_tile_position(filename):
    match = re.search(r'\.(\d+)\.', filename)
    if match:
        return int(match.group(1))
    return -1

def stitch_patches(patches, patch_coords, output_shape, patch_size=512):
    if patches[0].ndim == 3:
        stitched_map = np.zeros(output_shape, dtype=np.uint8)
    else:
        stitched_map = np.zeros(output_shape, dtype=np.uint8)
    for patch, (x, y) in zip(patches, patch_coords):
        if patch is None:
            continue
        h = min(patch_size, output_shape[0] - y)
        w = min(patch_size, output_shape[1] - x)
        stitched_map[y:y+h, x:x+w] = patch[:h, :w]
    return stitched_map

def load_and_process_dataset():
    image_files = sorted(glob(os.path.join(IMG_DIR, "*.png")))
    label_files = sorted(glob(os.path.join(LABEL_DIR, "*.xml")))
    
    if not image_files:
        st.error("No images found in directory.")
        return None, None, None
    
    patch_size = 512
    patches_per_row = 49
    max_rows = 49
    output_shape = (max_rows * patch_size, patches_per_row * patch_size)
    
    label_map = {}
    for label_path in label_files:
        tile_num = get_tile_position(os.path.basename(label_path))
        if tile_num != -1:
            label_map[tile_num] = label_path
    
    image_patches = []
    mask_patches = []
    patch_coords = []
    valid_images = 0
    valid_labels = 0
    
    for img_path in image_files:
        try:
            img = np.array(Image.open(img_path))
            if img.shape[-1] == 4:
                img = img[:, :, :3]
            img = pad_image(img)
            if img.shape[:2] != (512, 512):
                continue
        except Exception:
            continue
        
        tile_num = get_tile_position(os.path.basename(img_path))
        if tile_num == -1 or tile_num > 2400:
            continue
        row = (tile_num // patches_per_row) * patch_size
        col = (tile_num % patches_per_row) * patch_size
        
        label_path = label_map.get(tile_num, None)
        seg_map = parse_xml_label(label_path)
        if seg_map is None:
            continue
        if np.sum(seg_map) > 0:
            valid_labels += 1
        
        patch_coords.append((col, row))
        image_patches.append(img)
        mask_patches.append(seg_map)
        valid_images += 1
    
    if valid_images == 0:
        st.error("No valid images found.")
        return None, None, None
    
    try:
        stitched_image = stitch_patches(image_patches, patch_coords, (output_shape[0], output_shape[1], 3), patch_size)
        stitched_mask = stitch_patches(mask_patches, patch_coords, output_shape, patch_size)
    except Exception as e:
        st.error(f"Error stitching images: {str(e)}")
        return None, None, None
    
    return stitched_image, stitched_mask, output_shape

def run_streamlit():
    st.title("Patch Stitching")
    st.write("Stitching original images and masks from CANNES_TILES_512x512 dataset (Pool)")
    
    stitched_image, stitched_mask, output_shape = load_and_process_dataset()

    if stitched_image is None or stitched_mask is None:
        return
    
    # Display original image and mask side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(stitched_image, caption=f"Original Stitched Image")
    with col2:
        st.image(stitched_mask * 255, caption="Mask (Pool = white)")

if __name__ == "__main__":
    run_streamlit()
