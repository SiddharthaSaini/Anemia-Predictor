# Anemia Predictor

A Flask web application that predicts anemia based on blood parameters.

## Features

- Predicts anemia based on hemoglobin, MCH, MCHC, and MCV values
- User-friendly web interface
- Detailed explanations of results

## Deployment on Render

1. Fork this repository
2. Connect your GitHub repository to Render
3. Create a new Web Service on Render
4. Use the following settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python app.py`
   - Python Version: 3.11.9

## Local Development

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Linux/Mac)
4. Install dependencies: `pip install -r requirements.txt`
5. Run the application: `python app.py`

## Required Files

Make sure you have the following files in your project root:
- `model.pkl` - Trained machine learning model
- `scaler.pkl` - Data scaler for preprocessing
- `app.py` - Flask application
- `requirements.txt` - Python dependencies

## API Endpoints

- `GET /` - Home page with prediction form
- `POST /predict` - Submit prediction form and get results
