![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)

_______________________________________________________________________________________________________________________________________________________

🧠 COVID-19 Multi-Class Classification from Chest X-ray Images
Transfer Learning with VGG16

_______________________________________________________________________________________________________________________________________________________
📌 Project Overview

This project implements a structured deep learning pipeline to classify Chest X-ray images into:

⁕ 🦠 COVID-19

⁕ 🫁 Viral Pneumonia

⁕ ✅ Normal

The main goal was not only to train a model, but to design a robust training & evaluation pipeline suitable for small medical datasets while addressing:

⁕ Overfitting

⁕ Class imbalance

⁕ Proper medical evaluation metrics
_______________________________________________________________________________________________________________________________________________________

📊 Dataset Summary

Context
Helping Deep Learning and AI Enthusiasts like me to contribute to improving COVID-19 detection using just Chest X-rays.

Content
It is a simple directory structure branched into test and train and further branched into the respective 3 classes which contains the images.


Training Set (251 images)

⁕ COVID: 111

⁕ Normal: 70

⁕ Viral Pneumonia: 70

Test Set (66 images)

⁕ COVID: 26

⁕ Normal: 20

⁕ Viral Pneumonia: 20

Due to limited data size, careful regularization and transfer learning were applied.
_______________________________________________________________________________________________________________________________________________________


🧠 Model Architecture

Backbone: Pretrained VGG16 (ImageNet)
Custom Head:

⁕ GlobalAveragePooling2D

⁕ Dense(128, ReLU)

⁕ Dropout(0.5)

⁕ Softmax (3 classes)

Why VGG16?

⁕ Strong feature extractor

⁕ Stable convergence

⁕ Performs well on small medical datasets

_______________________________________________________________________________________________________________________________________________________


⚙️ Training Strategy

⁕ Frozen convolutional base (initial training phase)

⁕ Selective fine-tuning of last layers

⁕ Data augmentation

⁕ Class weighting using compute_class_weight

⁕ Early stopping to prevent overfitting

⁕ Adam optimizer (low learning rate)
_______________________________________________________________________________________________________________________________________________________


📈 Evaluation

Medical AI models require more than accuracy.

The model was evaluated using:

⁕ Confusion Matrix

⁕ Precision

⁕ Recall

⁕ F1-Score

📊 Results

Test Accuracy: ~89%

Key observation:
Minor confusion between COVID and Viral Pneumonia due to radiographic similarity — expected in small datasets.
_______________________________________________________________________________________________________________________________________________________


📊 Training Curves

Accuracy:

<img width="640" height="480" alt="accuracy" src="https://github.com/user-attachments/assets/8781b275-748b-4754-b0ba-315e9fffe729" />

Loss:

<img width="640" height="480" alt="loss" src="https://github.com/user-attachments/assets/ab24ad21-f6f1-4a46-89b2-11327abe0f84" />


Confusion Matrix:

<img width="600" height="600" alt="confusion_matrix" src="https://github.com/user-attachments/assets/0cb76ea7-f141-49a8-8b27-c983ba3bbc69" />


_______________________________________________________________________________________________________________________________________________________

🚀 How to Run
git clone https://github.com/AbdUllahMohammedIsmail/covid-xray-vgg16.git
cd covid-xray-vgg16
pip install -r requirements.txt

Run training:

python src/train.py

Run evaluation:

python src/evaluate.py
_______________________________________________________________________________________________________________________________________________________

🛠 Tech Stack

⁕ Python

⁕ TensorFlow / Keras

⁕ NumPy

⁕ Scikit-learn

⁕ Matplotlib

⁕ Seaborn
_______________________________________________________________________________________________________________________________________________________

⚠️ Disclaimer

This project is for research and educational purposes only.

It is not intended for real clinical diagnosis.
_______________________________________________________________________________________________________________________________________________________

👨‍💻 Author

Abdullah Mohamed
AI & Computer Vision Enthusiast

GitHub: https://github.com/AbdUllahMohammedIsmail




