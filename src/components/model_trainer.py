import sys
import os
from dataclasses import dataclass

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, load_object, evaluate

@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join("artifacts", "lstm_model.h5")
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("Initializing LSTM model training...")

            model = Sequential([
                LSTM(64, activation='relu', return_sequences=False, input_shape=(X_train.shape[1], 1)),
                Dense(1)
            ])

            model.compile(optimizer='adam', loss='mse')
            logging.info("Model compiled successfully.")

            history = model.fit(
                X_train, y_train,
                epochs=30,
                batch_size=16,
                validation_split=0.1,
                verbose=1
            )

            logging.info("Model training completed.")

            save_object(self.config.model_path, model)
            logging.info(f"Trained LSTM model saved to {self.config.model_path}")

            preprocessor = load_object(self.config.preprocessor_path)

            predictions_scaled = model.predict(X_test)
            predictions = preprocessor.inverse_transform(predictions_scaled)
            y_true = preprocessor.inverse_transform(y_test.reshape(-1, 1))

            rmse, mae, mape = evaluate(y_true, predictions)

            logging.info(f"LSTM RMSE: {rmse:.2f}")
            logging.info(f"LSTM MAE: {mae:.2f}")
            logging.info(f"LSTM MAPE: {mape:.2f}%")

            return {
                "rmse": rmse,
                "mae": mae,
                "mape": mape,
                "model_path": self.config.model_path
            }

        except Exception as e:
            raise CustomException(e, sys)
