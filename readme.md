
# Time-Series Forecasting with ARIMA

This repository contains the necessary files to perform time-series forecasting using the ARIMA (AutoRegressive Integrated Moving Average) model on a given dataset.

## Project Structure

```
TIME-SERIES-FORECASTING-ARIMA
│
├── data
│   ├── train
│   │   └── train_data.csv
│   └── valid
│       └── valid_data.csv
│
├── myenv
│
├── notebook
│   └── ARIMA and SARIMA models.ipynb
│
├── scripts
│   └── time_series_arima.py
│
├── .gitignore
│
└── readme.md
└── requirements.txt
```

## Installation

Before running the scripts, ensure that you have the following prerequisites installed:

- Python 3
- pandas
- numpy
- matplotlib
- scikit-learn
- statsmodels

You can install all required packages using the following command:

```
pip install -r requirements.txt
```

## Usage

To execute the time-series forecasting:

1. Navigate to the `notebook` directory and open `ARIMA and SARIMA models.ipynb` in Jupyter Notebook or JupyterLab.
2. Execute the cells sequentially to preprocess the data, fit the ARIMA model, and make forecasts.
3. The `scripts` directory contains `time_series_arima.py` if you prefer running a Python script.

## Overview of the Process

The notebook/script includes:

- Importing the required libraries for handling data, calculations, and plotting.
- Loading and preprocessing the training and validation datasets.
- Visualizing the data to understand the trends and patterns.
- Conducting stationarity tests using the Dickey-Fuller and KPSS tests.
- Making the time-series data stationary for better modeling.
- Plotting ACF and PACF charts to identify ARIMA model parameters.
- Fitting the ARIMA model to the training data.
- Forecasting the future points in the time-series and comparing them with the validation set.
- Calculating and printing the RMSE (Root Mean Square Error) to evaluate the model performance.

## Contributing

Feel free to fork this repository and contribute by submitting a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
```

