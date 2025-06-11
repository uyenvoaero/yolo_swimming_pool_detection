# Pool Detection and Visualization

This project stitches 512x512 image tiles from the CANNES_TILES_512x512 dataset to create a large map, visualizes pools using a Streamlit app, and trains a YOLOv8 model for pool detection.

## Features
- Stitches 512x512 image tiles and segmentation masks into a large map (up to 49x49 tiles).
- Displays stitched images and masks via Streamlit.
- Converts XML annotations to YOLO format and splits data into train/val/test sets.
- Trains and performs inference with YOLOv8 for pool detection.

## Dataset
The project uses the [Swimming Pool 512x512 dataset](https://www.kaggle.com/datasets/alexj21/swimming-pool-512x512) from Kaggle, containing 512x512 PNG images and XML annotations for pools.

## Setup
1. Clone the repo:
   ```bash
   git clone https://github.com/uyenvoaero/yolo_swimming_pool_detection.git
   ```
2. Install dependencies:
   ```bash
   pip install numpy streamlit pillow ultralytics pyyaml
   ```
3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/alexj21/swimming-pool-512x512) and place it in `./original_dataset`.

## Usage
- **Visualize**: Run `streamlit run main.py` to stitch images/masks and view results.
  ![Streamlit Interface](https://raw.githubusercontent.com/uyenvoaero/yolo_swimming_pool_detection/main/runs/streamlit.png)
- **Train/Infer**: Run `python main.py` to train YOLOv8 and perform inference.
  ![Sample Inference](https://raw.githubusercontent.com/uyenvoaero/yolo_swimming_pool_detection/main/runs/inference/CANNES_TILES_512x512.118.jpg)
