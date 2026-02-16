<img width="2560" height="1440" alt="Añadir un título(1)" src="https://github.com/user-attachments/assets/16e98af3-04f2-4de9-9400-928b93b4aa6e" />

# Karmi1.1M: Eye State Detection System

**Karmi** is a hybrid computer vision system designed to classify eye states (Open/Closed) in real-time. It combines the geometric reliability of **Haar Cascades** with the precision of a custom **Convolutional Neural Network (CNN)**.

> **Project Status:** Functional (Training & Live Inference).  
> **Base Dataset:** MRL Eyes Dataset (2018).

## Key Features

### 1. Hybrid Detection Architecture
Unlike standard detectors, Karmi implements an **Anatomical Anchor Logic** in `try.py`:
* **Precise Mode (Blue Line):** When a **nose** is detected via `haarcascade_mcs_nose.xml`, the system calculates the exact anatomical position of the eyes based on facial structure, significantly reducing false positives.
* **Estimation Mode (Yellow Line):** If the nose is obscured, it falls back to standard facial proportions.

### 2. Optimized CNN (64x64)
The "brain" of the system (`train.py`) is a lightweight network built for high-speed inference:
* **Input:** RGB images normalized to $64 \times 64$ pixels.
* **Structure:** 3 Convolutional blocks with *BatchNormalization*, *MaxPooling*, and *Dropout* to prevent overfitting.
* **Output:** Binary classification (Sigmoid) determining the probability of the eye being closed ($0 \rightarrow$ Open, $1 \rightarrow$ Closed).



### 3. Advanced Preprocessing
To ensure reliability across various lighting conditions, the system applies:
* **CLAHE:** *Contrast Limited Adaptive Histogram Equalization* to highlight the pupil and eyelids in dark environments.
* **Normalization:** Tensor adjustment ($1/255$) to match training distribution.

---

## Model Architecture

The following table details the internal layers of the **Karmi** sequential model as defined in `train.py`:

| Layer (type) | Output Shape | Param # |
| :--- | :--- | :--- |
| **RandomFlip** (Augmentation) | (None, 64, 64, 3) | 0 |
| **RandomRotation** | (None, 64, 64, 3) | 0 |
| **RandomZoom** | (None, 64, 64, 3) | 0 |
| **Conv2D** (32 filters, 3x3) | (None, 64, 64, 32) | 896 |
| **BatchNormalization** | (None, 64, 64, 32) | 128 |
| **MaxPooling2D** | (None, 32, 32, 32) | 0 |
| **Dropout** (0.25) | (None, 32, 32, 32) | 0 |
| **Conv2D** (64 filters, 3x3) | (None, 32, 32, 64) | 18,496 |
| **BatchNormalization** | (None, 32, 32, 64) | 256 |
| **MaxPooling2D** | (None, 16, 16, 64) | 0 |
| **Dropout** (0.25) | (None, 16, 16, 64) | 0 |
| **Conv2D** (128 filters, 3x3) | (None, 16, 16, 128) | 73,856 |
| **BatchNormalization** | (None, 16, 16, 128) | 512 |
| **MaxPooling2D** | (None, 8, 8, 128) | 0 |
| **Dropout** (0.25) | (None, 8, 8, 128) | 0 |
| **Flatten** | (None, 8192) | 0 |
| **Dense** (128 units, ReLU) | (None, 128) | 1,048,704 |
| **Dropout** (0.5) | (None, 128) | 0 |
| **Dense** (64 units, ReLU) | (None, 64) | 8,256 |
| **Dropout** (0.3) | (None, 64) | 0 |
| **Dense** (1 unit, Sigmoid) | (None, 1) | 65 |

**Total Trainable Params:** 1,150,721 (~4.39 MB)

---

## 🛠️ Project Structure

The workflow is divided into three main scripts:

| File | Function | Technical Description |
| :--- | :--- | :--- |
| `feed.py` | **Data ETL** | Processes the *MRL Eyes* dataset, sorting images into `open/closed` folders based on filename parsing. |
| `train.py` | **Training** | Trains the CNN using *Data Augmentation* and *Early Stopping*. Saves the final model as `modelo_ojos_64.h5`. |
| `try.py` | **Inference** | Runs real-time detection via webcam. Includes GUI sliders for "Sensitivity Threshold" and "Eye Padding". |

## 💻 Installation and Usage

1. **Prepare Dataset:** Download *MRL Eyes 2018* and run:
   ```bash
   python feed.py
   ```
2. **Train the model:**
   ```bash
   python train.py
   ```
3. **Try the model:**
   ```bash
   python try.py
   ```
