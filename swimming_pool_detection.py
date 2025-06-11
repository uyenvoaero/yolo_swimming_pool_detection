import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from ultralytics import YOLO
import yaml


class DataProcessor:
    def __init__(self, original_dataset_dir, img_dir, label_dir, dataset_dir):
        self.original_dataset_dir = original_dataset_dir
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.dataset_dir = dataset_dir
        self.setup_directories()

    def setup_directories(self):
        os.makedirs(self.dataset_dir, exist_ok=True)
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(self.dataset_dir, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.dataset_dir, split, 'labels'), exist_ok=True)

    def xml_to_yolo(self, xml_file, img_width=512, img_height=512):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        yolo_labels = []
        
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name != 'pool':
                continue  # Only process class 'pool'
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            
            # Convert to YOLO coordinates (normalized)
            x_center = (xmin + xmax) / 2 / img_width
            y_center = (ymin + ymax) / 2 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            # Class ID (0 for pool since there is only one class)
            yolo_labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        return yolo_labels

    def process_files(self, file_list, split):
        for img_file in file_list:
            # Copy image
            src_img = os.path.join(self.img_dir, img_file)
            dst_img = os.path.join(self.dataset_dir, split, 'images', img_file)
            shutil.copy(src_img, dst_img)
            
            # Convert and save labels
            xml_file = os.path.join(self.label_dir, img_file.replace('.png', '.xml'))
            if os.path.exists(xml_file):
                yolo_labels = self.xml_to_yolo(xml_file)
                label_file = os.path.join(self.dataset_dir, split, 'labels', img_file.replace('.png', '.txt'))
                with open(label_file, 'w') as f:
                    f.write('\n'.join(yolo_labels))

    def create_yaml(self):
        yaml_content = {
            'path': self.dataset_dir,
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': 1,
            'names': ['pool']
        }

        yaml_file = os.path.join(self.dataset_dir, 'data.yaml')
        with open(yaml_file, 'w') as f:
            yaml.dump(yaml_content, f)
        return yaml_file

    def split_data(self):
        img_files = [f for f in os.listdir(self.img_dir) if f.endswith('.png')]
        xml_files = [f for f in os.listdir(self.label_dir) if f.endswith('.xml')]

        # Only take images with labels
        valid_files = []
        for img in img_files:
            xml_name = img.replace('.png', '.xml')
            if xml_name in xml_files:
                valid_files.append(img)

        # Split data
        random.shuffle(valid_files)
        n_total = len(valid_files)
        n_train = int(0.8 * n_total)  # 80% train
        n_val = int(0.1 * n_total)    # 10% val
        n_test = n_total - n_train - n_val  # 10% test

        train_files = valid_files[:n_train]
        val_files = valid_files[n_train:n_train + n_val]
        test_files = valid_files[n_train + n_val:]

        return train_files, val_files, test_files


class ModelTrainer:
    def __init__(self, yaml_file):
        self.yaml_file = yaml_file

    def train(self):
        model = YOLO('yolov8n.pt')  # Use YOLOv8 nano pre-trained
        model.train(
            data=self.yaml_file,
            epochs=50,
            imgsz=512,
            batch=16,
            name='train',
            project="./runs",
            device='cpu'  # Use CPU
        )

    def predict(self, test_dir):
        model = YOLO('yolov8n.pt')
        results = model.predict(
            source=test_dir,
            save=True,
            project="./runs",
            name='inference'
        )

if __name__ == "__main__":
    # Prepare data for training
    original_dataset_dir = "./original_dataset"
    img_dir = os.path.join(original_dataset_dir, "CANNES_TILES_512x512_PNG", "CANNES_TILES_512x512_PNG")
    label_dir = os.path.join(original_dataset_dir, "CANNES_TILES_512x512_labels", "CANNES_TILES_512x512_labels")
    dataset_dir = "./dataset"

    processor = DataProcessor(original_dataset_dir, img_dir, label_dir, dataset_dir)
    train_files, val_files, test_files = processor.split_data()

    processor.process_files(train_files, 'train')
    processor.process_files(val_files, 'val')
    processor.process_files(test_files, 'test')

    # Training
    yaml_file = processor.create_yaml()

    trainer = ModelTrainer(yaml_file)
    trainer.train()

    # Inference
    trainer.predict(os.path.join(dataset_dir, 'test/images'))