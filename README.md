# 🧠 Sugar Price Forecasting in Dhaka Markets (LSTM-based)

## 📝 Problem Statement

This project aims to forecast the **daily retail price of sugar** in Dhaka’s markets using **5 years of historical price data**. The data was collected from the official website of the [**Trading Corporation of Bangladesh**](https://www.tcb.gov.bd), which regularly publishes the prices of essential commodities.

---

## 🚀 Project Highlights

This end-to-end machine learning pipeline forecasts future sugar prices using **LSTM neural networks**, enabling the detection of pricing trends based on historical patterns. The solution mimics real-world deployment practices — from data extraction and preprocessing to model training and evaluation.

---

## 🔧 Tools & Technologies Used

- **Python** 🐍  
- **Pandas, NumPy** — for data manipulation  
- **BeautifulSoup, Selenium** — for web scraping TCB price data  
- **Scikit-learn** — preprocessing utilities  
- **TensorFlow/Keras** — for building the LSTM model  
- **Matplotlib, Seaborn** — for data visualization and EDA  
- **ML pipeline structuring** — modularized using object-oriented design principles  
- **Logging & Exception handling** — for production-grade robustness  
- **Pickle (.pkl)** and **HDF5 (.h5)** formats — for saving preprocessors and models  

---

## 🔄 Workflow Summary

1. **Data Extraction**  
   Scraped daily sugar prices from [TCB's official website](https://www.tcb.gov.bd), cleaned and structured the data into a time series format.

2. **Exploratory Data Analysis**  
   - Visualized price trends  
   - Identified missing values and market irregularities  
   - Standardized units and resolved data quality issues
   - compared different models like ARIMA, SARIMA, LSTM, PROPHET to choose best model

3. **Data Transformation**  
   - Converted date strings to datetime index  
   - Dropped categorical variables  
   - Applied **MinMax scaling**  
   - Generated LSTM-ready sequences using a 30-day sliding window  

4. **Model Training**  
   - Trained a **univariate LSTM model** on the scaled time series data  
   - Tuned hyperparameters like `window_size`, `epochs`, and `batch_size`  
   - Evaluated model using `RMSE`, `MAE`, and `MAPE`  

5. **Model Evaluation & Saving**  
   - Achieved high accuracy on test data  
   - Saved model as `.h5` file and preprocessing pipeline as `.pkl`  

---

## 📈 Results

The trained LSTM model effectively captured the seasonality and volatility in sugar prices, demonstrating strong performance on out-of-sample test data.

| Metric | Value |
|--------|-------|
| MAE    | ~1.75 |
| MAPE   | ~1.49% |

---

## 📁 Project Structure

<pre><code>├── artifacts/ │ ├── lstm_model.h5 │ └── preprocessor.pkl ├── data/ │ └── raw_sugar_prices.csv ├── src/ │ ├── components/ │ │ ├── data_ingestion.py │ │ ├── data_transformation.py │ │ └── model_trainer.py │ ├── utils.py │ ├── logger.py │ └── exception.py ├── notebooks/ │ └── EDA.ipynb ├── README.md └── requirements.txt </code></pre>


---

## 💼 What I Did

- Designed a full ML pipeline from scratch — extraction to deployment  
- Implemented a deep learning model using **LSTM** for time series forecasting  
- Built reusable, modular Python components with exception handling  
- Applied best practices in code organization and model evaluation  
- Gained hands-on experience with real-world economic data  

---

## 🔍 What's Next?

- Compare LSTM with traditional models (SARIMA, Prophet, etc.)  
- Deploy the model via a REST API or dashboard  
- Extend the model to predict prices of multiple commodities  


