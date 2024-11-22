import logging 
from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """
    Abstract class definiing strategy for handling data
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[ pd.DataFrame, pd.Series] :
        pass 

class DataPreProcessStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame :
        try:
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ],
                axis=1)
            data.fillna({"product_weight_g": data["product_weight_g"].median()}, inplace=True)
            data.fillna({"product_length_cm": data["product_length_cm"].median()}, inplace=True)
            data.fillna({"product_height_cm": data["product_height_cm"].median()}, inplace=True)
            data.fillna({"product_width_cm": data["product_width_cm"].median()}, inplace=True)
            data.fillna({"review_comment_message": "No review"}, inplace=True)

            #data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
            #data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
            #data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
            #data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)
            #data["review_comment_message"].fillna("No review", inplace=True)

            data = data.select_dtypes(include=[np.number])
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis=1)
            return data
        except Exception as e:
            logging.error("Error in preprocessing data: {}".format(e))
            raise e 

class DataDivideStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> Union[ pd.DataFrame, pd.Series] :
        try:
            x = data.drop(["review_score"], axis=1)
            y = data["review_score"]
            X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)
            return X_train, X_test, Y_train, Y_test
        except Exception as e:
            logging.error("Error in dividing data: {}".format(e))
            raise e
        

class DataCleaning:
    def __init__(self, df: pd.DataFrame, strategy: DataStrategy):
        self.data = df
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in handling data {}".format(e))
            raise e

if __name__=="__main__":
    data = pd.read_csv("")
    data_cleaning = DataCleaning(data, DataPreProcessStrategy())
    data_cleaning.handle_data()