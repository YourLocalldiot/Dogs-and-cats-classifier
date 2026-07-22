# Cat vs Dog Image Classifier

## Streamlit link
https://dogs-and-cats-classifier-bikm8igcfybnegbccjhglg.streamlit.app/
## Overview
This project provides a modular and production-ready starter structure for building a Cat vs Dog image classification system with TensorFlow, Keras, Jupyter notebooks, and Streamlit.

## Project Structure
```text
cat_dog_classifier/
├── PetImages/                # Original dataset directory
├── notebooks/                # Jupyter notebooks for data prep, training, and testing
├── models/                   # Saved model artifacts
├── src/                      # Reusable Python modules
├── streamlit_app.py          # Streamlit web app for inference
├── requirements.txt          # Python package dependencies
├── README.md                 # Project documentation
└── .gitignore                # Git ignore rules
```

## Installation
1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start Jupyter Lab:
   ```bash
   jupyter lab
   ```

## Dataset Requirements
The project expects image folders under the dataset root:
- PetImages/Cat/
- PetImages/Dog/

Keep the original dataset intact and only remove corrupted files if necessary.

## Training Workflow
1. Open and run the dataset preparation notebook.
2. Train the model in the training notebook.
3. Test predictions in the evaluation notebook.
4. Launch the Streamlit app for inference.

## Running the Streamlit App
From the project root, run:
```bash
streamlit run streamlit_app.py
```

## Features
- Uses pathlib for file handling.
- Keeps data loading and model logic modular.
- Supports notebook-based experimentation and Streamlit inference.
- Ready for future expansion with additional preprocessing and evaluation steps.
