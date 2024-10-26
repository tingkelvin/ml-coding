import logging
import numpy as np

from sklearn.metrics import mean_squared_error
from google.cloud import aiplatform

from config import *
from delpoyment_config import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    
    logging.info("Making predictions from endpoint.")
    endpoint = aiplatform.Endpoint(str(END_POINT))
    test_data = np.load("data/X_test.npy")
    prediction = endpoint.predict(instances=test_data.tolist())
    logging.info("Predictions made successfully.")
    
    logging.info("Evaluating model.")
    grand_truth = np.load("data/y_test.npy")
    mse = mean_squared_error(grand_truth, prediction.predictions)
    logging.info(f"Model evaluation complete. MSE: {mse}")
