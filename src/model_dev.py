import logging 
from abc import ABC, abstractmethod 
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    Abstract class for all models
    """

    @abstractmethod
    def train(self, X_train, Y_train, **kwargs):
        pass 


class LinearRegressionModel(Model):
    def train(self, X_train, Y_train, **kwargs):
        try: 
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, Y_train)
            logging.info("Model training completed")
            return reg 
        except Exception as e:
            logging.error("Error occurred while training linear regression model {}".format(e))
            raise e 
     
