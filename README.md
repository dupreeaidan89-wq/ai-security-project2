\# AI Security Project – MNIST Adversarial Attack



\## Overview

This project demonstrates how machine learning models can be attacked using adversarial techniques and defended using adversarial training.



We use the MNIST dataset (handwritten digits) to train a neural network classifier, apply the FGSM attack to reduce accuracy, and then improve robustness using a defense method.



\---



\## Features

\- Trains a neural network on MNIST dataset

\- Implements Fast Gradient Sign Method (FGSM) attack

\- Demonstrates model performance drop under attack

\- Applies adversarial training as a defense

\- Visualizes results using graphs



\---



\## Results

\- Normal Accuracy: \~96%

\- Under FGSM Attack: accuracy drops significantly

\- After Defense: improved robustness (\~80%+)



\---



\## Technologies Used

\- Python

\- PyTorch

\- Torchvision

\- Matplotlib



\---



\## How to Run



```bash

python train\_model.py

