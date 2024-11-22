import logging

from pipelines.training_pipeline import train_pipeline 

if __name__ == "__main__":
    print("about to train pipeline")
    train_pipeline(data_path="/Users/Prashant/LOCAL/Data/github/python/mlops-zenml-customer-satisfaction-prediction/archive/olist_customers_dataset.csv")
    print("pipeline trained")