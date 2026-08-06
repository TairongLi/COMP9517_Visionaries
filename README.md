
## Project Structure
```
+---notebooks
|       01_cnn_baseline_lr1e-3.ipynb
|       02_cnn_baseline_lr1e-2.ipynb
|       03_cnn_baseline_lr1e-4.ipynb
|       04_resnet18_pretrained_lr1e-4.ipynb
|       05_resnet18_pretrained_lr1e-3.ipynb
|       06_resnet18_pretrained_lr1e-2.ipynb
|       07_resnet18_scratch_lr1e-4.ipynb
|       08_resnet18_scratch_lr1e-3.ipynb
|       09_resnet18_scratch_lr1e-2.ipynb
|       10__confusion_matrixcnnlr1e-4.ipynb
|       11__confusion_matrixcnnlr1e-3.ipynb
|       12__confusion_matrixcnnlr1e-2.ipynb
|       13__confusion_matrixResnet_pretrain_lr1e-4.ipynb
|       14__confusion_matrixResnet_pretrain_lr1e-3.ipynb
|       15__confusion_matrixResnet_pretrain_lr1e-2.ipynb
|       16__confusion_matrixResnet_scratch_lr1e-4.ipynb
|       17__confusion_matrixResnet_scratch_lr1e-3.ipynb
|       18__confusion_matrixResnet_scratch_lr1e-2.ipynb
|       traditional_bovw.ipynb
|       traditional_handcrafted_features.ipynb
|
\---src
    |   cnn_baseline.py
    |   data_utils.py
    |   eval_utils.py
    |   metrics.py
    |   plot_utils.py
    |   resnet18_pretrained.py
    |   resnet18_scratch.py
    |   train_utils.py
    |   utils.py
    |   __init__.py
    |
    +---traditional
    |   |   bovw.py
    |   |   dataset.py
    |   |   features.py
    |   |   io.py
    |   |   run_features.py
    |   |   __init__.py
    |   |
    |   \---__pycache__
    |           bovw.cpython-313.pyc
    |           dataset.cpython-313.pyc
    |           features.cpython-313.pyc
    |           io.cpython-313.pyc
    |           __init__.cpython-313.pyc
    |
    \---__pycache__
            cnn_baseline.cpython-311.pyc
            cnn_baseline.cpython-312.pyc
            data_utils.cpython-311.pyc
            data_utils.cpython-312.pyc
            data_utils.cpython-313.pyc
            eval_utils.cpython-311.pyc
            eval_utils.cpython-312.pyc
            plot_utils.cpython-311.pyc
            plot_utils.cpython-312.pyc
            resnet18_pretrained.cpython-311.pyc
            resnet18_pretrained.cpython-312.pyc
            resnet18_scratch.cpython-312.pyc
            train_utils.cpython-311.pyc
            train_utils.cpython-312.pyc
            utils.cpython-311.pyc
            utils.cpython-312.pyc
            __init__.cpython-311.pyc
            __init__.cpython-312.pyc
            __init__.cpython-313.pyc# Image Classification Project
```
This project contains implementations of both deep learning and traditional machine learning methods for image classification. It includes training scripts, evaluation notebooks, visualization tools, and feature extraction pipelines.

---

---

# notebooks/

The notebooks are used for experiments, model training, evaluation, and visualization.

## CNN Baseline Experiments

| Notebook | Description |
|----------|-------------|
| 01_cnn_baseline_lr1e-3.ipynb | Train the baseline CNN using learning rate = 1e-3 |
| 02_cnn_baseline_lr1e-2.ipynb | Train the baseline CNN using learning rate = 1e-2 |
| 03_cnn_baseline_lr1e-4.ipynb | Train the baseline CNN using learning rate = 1e-4 |

---

## ResNet18 (Pretrained)

| Notebook | Description |
|----------|-------------|
| 04_resnet18_pretrained_lr1e-4.ipynb | Fine-tune pretrained ResNet18 with learning rate = 1e-4 |
| 05_resnet18_pretrained_lr1e-3.ipynb | Fine-tune pretrained ResNet18 with learning rate = 1e-3 |
| 06_resnet18_pretrained_lr1e-2.ipynb | Fine-tune pretrained ResNet18 with learning rate = 1e-2 |

---

## ResNet18 (Training from Scratch)

| Notebook | Description |
|----------|-------------|
| 07_resnet18_scratch_lr1e-4.ipynb | Train ResNet18 from scratch using learning rate = 1e-4 |
| 08_resnet18_scratch_lr1e-3.ipynb | Train ResNet18 from scratch using learning rate = 1e-3 |
| 09_resnet18_scratch_lr1e-2.ipynb | Train ResNet18 from scratch using learning rate = 1e-2 |

---

## Confusion Matrix Evaluation

These notebooks load trained models and visualize their confusion matrices.

### CNN

- 10__confusion_matrixcnnlr1e-4.ipynb
- 11__confusion_matrixcnnlr1e-3.ipynb
- 12__confusion_matrixcnnlr1e-2.ipynb

### Pretrained ResNet18

- 13__confusion_matrixResnet_pretrain_lr1e-4.ipynb
- 14__confusion_matrixResnet_pretrain_lr1e-3.ipynb
- 15__confusion_matrixResnet_pretrain_lr1e-2.ipynb

### Scratch ResNet18

- 16__confusion_matrixResnet_scratch_lr1e-4.ipynb
- 17__confusion_matrixResnet_scratch_lr1e-3.ipynb
- 18__confusion_matrixResnet_scratch_lr1e-2.ipynb

---

## Traditional Machine Learning

| Notebook | Description |
|----------|-------------|
| traditional_handcrafted_features.ipynb | Classification using handcrafted image features (e.g., HOG, LBP, Color Histogram). |
| traditional_bovw.ipynb | Classification using the Bag of Visual Words (BoVW) pipeline. |

---

# src/

The `src` directory contains reusable Python modules used by the notebooks.

| File | Description |
|------|-------------|
| cnn_baseline.py | Defines the baseline CNN architecture. |
| resnet18_pretrained.py | Loads and configures a pretrained ResNet18 model for transfer learning. |
| resnet18_scratch.py | Defines a ResNet18 model trained entirely from scratch. |
| data_utils.py | Dataset loading, preprocessing, augmentation, and DataLoader creation. |
| train_utils.py | Training loop, validation loop, checkpoint saving, and learning utilities. |
| eval_utils.py | Model evaluation, prediction generation, and testing functions. |
| metrics.py | Computes evaluation metrics such as accuracy, precision, recall, and F1-score. |
| plot_utils.py | Visualization utilities for training curves, confusion matrices, and other plots. |
| utils.py | General helper functions shared across the project. |
| __init__.py | Marks the directory as a Python package. |

---

# src/traditional/

Modules implementing traditional computer vision pipelines.

| File | Description |
|------|-------------|
| dataset.py | Dataset loading utilities for traditional methods. |
| features.py | Handcrafted feature extraction (e.g., HOG, LBP, color features). |
| bovw.py | Bag of Visual Words implementation, including vocabulary construction and feature encoding. |
| io.py | Input/output helper functions for saving and loading extracted features. |
| run_features.py | Executes the complete handcrafted feature extraction pipeline. |
| __init__.py | Package initialization file. |

---

# Workflow

The typical workflow of this project is:

1. Load and preprocess the dataset (`data_utils.py`).
2. Train a CNN or ResNet model using the corresponding notebook.
3. Evaluate the trained model with evaluation utilities.
4. Visualize performance using confusion matrices and training curves.
5. Compare deep learning methods with traditional handcrafted-feature and BoVW approaches.

---

# Models Included

- Baseline CNN
- ResNet18 (Pretrained)
- ResNet18 (Scratch)
- Handcrafted Feature Classifier
- Bag of Visual Words (BoVW)

These models are compared under different learning rates and evaluated using common classification metrics and confusion matrices.

