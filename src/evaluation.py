import logging
from abc import ABC, abstractmethod
import numpy as np 
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates the scores for the model
        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        pass  

class MSE(Evaluation):
    """
    Evaluation strategy that uses Mean Squared Error 
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info("MSE: {}".format(mse))
            return mse
        except Exception as e:
            logging.error("Error while calculating MSE score {}".format(e))
            raise e 


class R2Score(Evaluation):
    """
    Evaluation strategy that uses R2 score 
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating R2 Score")
            r2 = r2_score(y_true, y_pred)
            logging.info("MSE: {}".format(r2))
            return r2
        except Exception as e:
            logging.error("Error while calculating R2 score {}".format(e))
            raise e 
        

class RMSE(Evaluation):
    """
    Evaluation strategy that uses Root mean squared error 
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating RMSE")
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info("MSE: {}".format(rmse))
            return rmse
        except Exception as e:
            logging.error("Error while calculating RMSE {}".format(e))
            raise e 
        
