# Embryonic Ultrasound Health Analyzer using ResNet18

An AI-based desktop application built with Python and PyTorch to analyze embryonic ultrasound images using a ResNet18 deep learning model.

## Features
- Ultrasound image selection and analysis
- Deep learning-based image classification using ResNet18
- Image preprocessing with PyTorch
- Grad-CAM visualization for model interpretability
- Model training and testing scripts

## Tech Stack
- Python
- PyTorch
- Torchvision
- ResNet18
- Tkinter
- OpenCV
- NumPy
- Grad-CAM

## Project Files
- `app.py` – Desktop application and image prediction
- `train.py` – Model training script
- `requirements.txt` – Required Python packages
- `model.pth` – Pre-trained model weights (download separately)

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY_NAME.git
   ```

2. Navigate to the project directory:

   ```bash
   cd YOUR_REPOSITORY_NAME
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Download the Trained Model

Download `model.pth` from Hugging Face:

[Download Model Weights](https://huggingface.co/soulhunter77/embryonic-ultrasound-resnet18)

Place the downloaded `model.pth` file in the project root directory, alongside `app.py`.

## Run the Application

```bash
python app.py
```

## Model Training

Training code is available in `train.py`. The dataset is not included in this repository.

## Disclaimer

This project is developed for educational and research purposes only. It is not intended for clinical use or to replace professional medical diagnosis.
