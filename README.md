# 🖥️ Laptop Price Predictor

## Overview
The Laptop Price Predictor is a machine learning-based web application that helps estimate the price of laptops based on various specifications such as company, type, screen resolution, CPU, GPU, operating system, RAM, storage, and weight.  
It utilizes the laptop_data.csv dataset to preprocess features and train a machine learning pipeline combining preprocessing and models for accurate predictions.  
The project uses Jupyter Notebook (notebook.ipynb) for interactive data analysis and scikit-learn for model building, while Matplotlib and Seaborn enhance data visualization.  
The final trained pipeline (pipe.pkl) and cleaned dataset (df.pkl) are saved for deployment, and the Streamlit app (app.py) provides an interactive interface for real-time laptop price prediction.

## Features
- 📊 **Data Visualization**: Uses `Matplotlib` and `Seaborn` to visualize data insights and correlations.
- 🖥️ **Laptop Specification Analysis**: Reads `laptop_data.csv` for training machine learning models based on laptop features.
- 🤖 **Machine Learning Pipeline**: Implements `ColumnTransformer` and `Linear Regression` inside a `Pipeline`.
- 🌐 **Interactive Web App**: Built with `Streamlit` for user-friendly laptop price prediction.
- 🎯 **Performance Evaluation**: Evaluates the model using `R2 Score`, `Mean Absolute Error (MAE)`, and `Root Mean Squared Error (RMSE)`.
- 🛠️ **Result Analysis**: Provides visual comparison between actual and predicted prices and company-wise price distribution.

## Installation
To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/laptop-price-predictor.git
   cd laptop-price-predictor
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the Streamlit app:
   ```bash
   streamlit run app.py

## Project Structure

📂 laptop-price-predictor
│-- app.py # Streamlit app for laptop price prediction

│-- laptop-price-predictor.ipynb # Jupyter Notebook containing full data analysis workflow

│-- laptop_data.csv # Original dataset of laptop specifications

│-- pipe.pkl # Trained ML pipeline (preprocessing + model)

│-- df.pkl # Preprocessed cleaned dataset

│-- requirements.txt # List of required libraries

│-- README.md # Project documentation

## How It Works

1.Load the Dataset: The project reads laptop_data.csv into a pandas DataFrame.

2.Preprocess the Data: Cleans the data by handling categorical variables, converting RAM/Weight, and splitting memory into SSD and HDD.

3.Build the Pipeline: Constructs a Pipeline combining preprocessing steps and Linear Regression modeling.

4.Train the Model: Fits the model on the training data.

5.Evaluate the Model: Computes performance metrics like R2 Score, MAE, RMSE.

6.Visualize Results: Displays actual vs predicted prices scatter plot, heatmaps, and company-wise price analysis.

7.Deploy the App: Use Streamlit to create an interactive web app where users can input laptop specs and predict prices instantly.

## Output
<img width="879" alt="output" src="https://github.com/user-attachments/assets/6f978f25-12f1-4400-b31b-4f598382caaa" />

## Contributing
Feel free to fork this repository and contribute by submitting a pull request with improvements, bug fixes, or additional features!

## License
This project is licensed under the MIT License.



   
