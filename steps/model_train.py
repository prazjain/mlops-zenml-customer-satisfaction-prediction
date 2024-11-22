import logging 
import pandas as pd
from zenml import step 
from src.model_dev import LinearRegressionModel
#from .config import meth
from .config import ModelNameConfig
from sklearn.base import RegressorMixin

#meth()

@step
def train_model(
    x_train: pd.DataFrame
    ,y_train: pd.DataFrame
    ,config: ModelNameConfig
) -> RegressorMixin:
    model = None
    if config.model_name == "LinearRegression":
        model = LinearRegressionModel()
        trained_model = model.train(x_train, y_train)
        return trained_model 
    #elif config.model_name == "RandomForestRegressor":
    #    model = RandomForestRegressorModel()
    #    trained_model = model.train(x_train, y_train)
    #    return trained_model
    else:
        raise ValueError("Model {} not supported".format(config.model_name))