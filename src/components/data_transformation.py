import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.utils import create_sequences, save_object
from src.logger import logging
from src.exception import CustomException

@dataclass
class DataTransformationConfig:
    scaler_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def preprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all shared preprocessing steps to a DataFrame."""
        df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d")
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)

        cat_cols = [col for col in df.columns if df[col].dtype == 'O']
        df.drop(columns=cat_cols, inplace=True, errors='ignore')

        df.rename(columns={
            'min_price': 'min_price_per_kg',
            'max_price': 'max_price_per_kg'
        }, inplace=True)

        df['avg_price_per_kg'] = np.round(
            (df['min_price_per_kg'] + df['max_price_per_kg']) / 2, 2)

        return df

    def init_data_transformation(self, train_path: str, test_path: str, window_size: int = 30):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test CSVs read successfully.")

            train_df = self.preprocess_df(train_df)
            test_df = self.preprocess_df(test_df)

            logging.info("Preprocessing applied to train and test dataframes.")

            scaler = MinMaxScaler()
            scaled_train = scaler.fit_transform(train_df[['avg_price_per_kg']])
            scaled_test = scaler.transform(test_df[['avg_price_per_kg']])

            logging.info("Scaled average price with MinMaxScaler.")

            X_train, y_train = create_sequences(scaled_train, window_size)
            X_test, y_test = create_sequences(scaled_test, window_size)

            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

            logging.info("Created LSTM-ready sequences.")

            save_object(self.config.scaler_path, scaler)
            logging.info(f"Saved scaler object at {self.config.scaler_path}")

            return X_train, y_train, X_test, y_test, self.config.scaler_path

        except Exception as e:
            raise CustomException(e, sys)
