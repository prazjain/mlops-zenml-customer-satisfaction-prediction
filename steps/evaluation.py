import logging
import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated

from src.evaluation import MSE, R2Score, RMSE


@step
def evaluate_model(model: RegressorMixin
                   , x_test: pd.DataFrame
                   , y_test: pd.DataFrame) -> Tuple [
                       Annotated[float, "mse"]
                       ,Annotated[float, "r2"]
                       ,Annotated[float, "rmse"]
                   ]:
    try:
        prediction = model.predict(x_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, prediction)

        r2_class = R2Score()
        r2 = r2_class.calculate_scores(y_test, prediction)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test, prediction)
        
        return mse, r2, rmse
    except Exception as e:
        logging.error("Error occured while calculating scores {}".format(e))
        raise e
     
