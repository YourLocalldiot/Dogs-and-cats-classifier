# Cat vs Dog Image Classification

## Project Overview
This project builds a convolutional neural network (CNN) using TensorFlow to classify images as cats or dogs. It includes a data folder for training images, notebooks for experimentation, a trained model directory, and a Streamlit web app for inference.

## Folder Structure
CatDogClassifier/
├── PetImages/
│   ├── Cat/
│   └── Dog/
├── notebooks/
├── models/
├── app/
└── requirements.txt

## Installation
1. Open a terminal in the project root.
2. Create and activate a virtual environment (recommended):
   - Windows:
     ```bash
     python -m venv .venv
     .venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run the Notebooks
1. Start Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook
   ```
   or
   ```bash
   jupyter lab
   ```
2. Open a notebook from the notebooks/ folder.
3. Run the cells to train, evaluate, and save the model.

## How to Run the Streamlit App
1. Navigate to the app directory:
   ```bash
   cd app
   ```
2. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Open the local URL shown in the terminal in your browser.

## Notes
- Place your image dataset inside the PetImages/Cat and PetImages/Dog folders.
- The trained model can be saved in the models/ directory for reuse.
- The app can be extended to upload images and display predictions.
