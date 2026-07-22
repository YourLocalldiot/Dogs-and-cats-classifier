# CatDogClassifier

A simple Python machine learning project for classifying images as cats or dogs using TensorFlow/Keras and a Streamlit web interface.

## Structure

- PetImages/Cat and PetImages/Dog contain the dataset folders.
- src/ includes the data loading, model, training, prediction, and utility modules.
- app.py launches the Streamlit interface.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the model:
   ```bash
   python -m src.train
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```
