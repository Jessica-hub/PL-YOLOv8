
# Data & Weights Download Instructions

This folder includes instructions for downloading the necessary datasets and installations for the project. The directional block is located in `block.py` within the `ultralytics/nn/modules/` directory, and the `yolov8-ehobb` configuration is under `ultralytics/cfg/models/v8`.You are advised to download the base YOLOv8 model and integrate these modifications. Additionally, the algorithm for assessing vegetation metrics is accessible in Final.py within the veg_encroch_index directory.

**For further reading on the application of this technology, refer to the paper titled "Advanced YOLO-based Real-time Power Line Detection for Vegetation Management," which is currently under review.**
---

## 📂 Datasets

### 1️⃣ `image640.zip`
- **Description:** This dataset includes images, labels, and YAML files for power line images with a resolution of **640×640**.
- **Download Link:** [image640.zip](https://drive.google.com/file/d/1W1UfZsbCIQWDTiSzhQjwEQkO_hrOxl8S/view?usp=drive_link)

### 2️⃣ `images_vegmetric.zip`
- **Description:** This dataset contains power line image data used for generating vegetation encroachment metrics via `veg_encroch_index/Final.py`.
- **Download Link:** [images_vegmetric.zip](https://drive.google.com/file/d/19xxtcgXUuCRm7JSNGakvloeHml36Do-E/view?usp=sharing)

---

## 🎯 Model Weights

### 📌 `weights.zip`
- **Description:** This file includes trained **PL-YOLOv8 models** in different sizes (**small, medium, and large**) with both **directional block and original versions**.
- **Download Link:** [weights.zip](https://drive.google.com/file/d/1tY4HjSvwp98aTlxIBInNscqrdPxt9W3E/view?usp=sharing)

---

## 🔧 Setup Instructions

Follow these steps to set up the project on your local machine:

### 1️⃣ Install Git
- Download and install Git from [git-scm.com](https://git-scm.com/downloads).

### 2️⃣ Verify Git Installation
- Open **PowerShell** (or your preferred terminal) and check if Git is installed by running:
  ```bash
  git --version
  ```

### 3️⃣ Clone the Repository
- Navigate to your preferred project directory:
  ```bash
  cd path/to/your/destination/folder
  ```
- Clone the repository:
  ```bash
  git clone https://github.com/Jessica-hub/PL-YOLOv8.git
  ```
- Change directory to the cloned repository:
  ```bash
  cd ultralytics
  ```

### 4️⃣ Install the Project in Editable Mode
- Run the following command to install all dependencies:
  ```bash
  pip install -e .
  ```

---

## 🚀 Training Your Models

### 1️⃣ Navigate to the YOLO Directory
```bash
cd ultralytics
```

### 2️⃣ Train the **Original YOLOv8 Model**
```bash
yolo task=obb mode=train model=yolov8m-obb.pt \
     data=./datasets/image640/pLdata640.yaml \
     epochs=150 patience=50 batch=16 device=0 \
     single_cls=True project=C:/Users/olivi/ultralytics/logs/ \
     name=yolov8m
```

### 3️⃣ Train the **PL-YOLOv8 Model**
```bash
yolo task=obb mode=train model=yolov8m-ehobb.yaml \
     data=./datasets/image640/pLdata640.yaml \
     epochs=150 patience=50 batch=16 device=0 \
     single_cls=True project=C:/Users/olivi/ultralytics/logs/ \
     name=yolov8m pretrained=yolov8m-obb.pt
```

---

## 📝 Additional Notes

- **Dataset Placement:** Ensure that the downloaded datasets are extracted into the correct folders as expected by the training scripts.
- **Model Verification:** After downloading the model weights, verify their integrity before use.
- **Basic YOLOv8 Usage:** For general YOLOv8 usage and examples, refer to the official Ultralytics repository:  
  🔗 [Ultralytics YOLOv8 Documentation](https://github.com/ultralytics/ultralytics/tree/main)

Happy Training! 🚀🔥

