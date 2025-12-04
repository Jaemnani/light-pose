# LightPose

#### Thanks to inzapp.

## My Updates

Through this major update, we have established a pipeline ranging from **Multi-mode support** to **Data Preprocessing and Deployment (Export)**.

1.  **Multi-Mode Support**: The architecture has been refactored to support various objects such as **Human, Pod, and Pallet**, moving away from a single-model structure.
2.  **Automated Pipeline**:
    * `tools/parse_coco_keypoints.py`: Automatically converts COCO format data for model training (Auto-Crop & Normalize).
    * `tools/train_valid_split.py`: Automatically splits Train/Validation datasets.
3.  **Deployment Ready**: Added `tools/export.py` to convert trained models into **ONNX** and **OpenVINO IR** formats.
4.  **Smart Inference**: Automatically detects the mode by analyzing the model's output shape when running `test.py`.

---

## 🛠 Requirements

* python 3.8+
```bash
pip install -r requirements.txt
```

## 📂 Dataset Preparation

### 1. Place Dataset
Place the compressed file of COCO Keypoint format (including images) downloaded from CVAT or other sources into the `datasets/` folder.

```text
datasets/
├── keypoints_dataset_no119.zip
└── keypoints_dataset_no120.zip
```

### 2. Unzip
Unzip the files to match the folder structure.

```text
datasets/
├── no119
│   ├── annotations
│   └── images
│       └── default
└── no120
    ├── annotations
    └── images
        └── default
```

### 3. COCO Parsing (Preprocessing)
Parses the JSON in COCO format to crop objects and generate normalized labels (.txt).
```Bash
python tools/parse_coco_keypoints.py
```

### 4. Train / Validation Split
Splits the preprocessed data into training and validation sets.
```Bash
python tools/train_valid_split.py
```

### 🚀 Training
Start training by specifying the desired mode (`--mode`).
```Bash
# Train Pod
python train.py --mode Pod 

# Train Pallet
python train.py --mode Pallet
```

### 🧪 Testing
Specify the model file and the test image folder. (The system automatically distinguishes Pod/Pallet/Human modes via the model's Output Size.)
```Bash
# Basic Usage
python test.py --model_path {model_name}.h5 --test_path {test_img_dir}

# Example
python test.py --model_path model.h5 --test_path preprocessed/pod/test/
```

### 📦 Export (ONNX, OpenVINO)
Convert the trained model into a deployment format. The results are saved in the `result/export/` path.
```Bash
python tools/export.py --model_path {model_name}.h5
```

--- 

LightPose is a top-down key point detection model that inspired by the yolo object detection model

This is a one-stage model in which both feature extraction and key point detection are performed through a simple convolution network

Unlike other keypoint detection models, this model is characterized by high coordinate accuracy because it additionally learns offset for grid

It also learns low confidence for human-invisible-keypoints, resulting in significantly less FP

For these reasons, models tend to learn more about the shape of a person

For example, if the input image is an upper body image,

The model predicts only key points for the upper body and not unnecessary key points for the lower body

The more diverse the training data, the more robust the model is

<img src="/md/sample.jpg" width="500"><br>




## Augmentation

In addition to basic augmentation, rotation augmentation can be used to improve the performance of the model

The model is the same image, but by further learning the rotated image, overfitting is avoided and more generalized

<img src="/md/augmentation.jpg" width="800"><br>

We also provide scripts that can perform this augmentation simply

## Loss function

This model uses ALE loss, an improved version of Binary Crossentropy loss

See [**absolute-logarithmic-error**](https://github.com/inzapp/absolute-logarithmic-error)

## Labeling

What labeling tools should I use to make training data?

This model provides a dedicated labeling tool, label_pose

<img src="/md/label_pose.gif" width="500"><br>

Here's how to use it

d : next image<br>
a : previous image<br>
e : next limb point<br>
q : previous limb point<br>
w : toggle to show skeleton line<br>
f : auto find and go to not labeled image<br>
x : remove current label content<br>
left click : set limb point<br>
right click : remove limb point<br>
ESC : exit program
