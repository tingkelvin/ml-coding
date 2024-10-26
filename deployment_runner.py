import subprocess
import sys
import os
import json

import numpy as np

from typing import Optional
from config import *
from delpoyment_config import *
from google.cloud import aiplatform
from sklearn.metrics import mean_squared_error

class Deployment_runner:
    def __init__(self):
        aiplatform.init(project=PROJECT_ID, location=LOCATION)

    def upload_model_sample(
        self,
        display_name: str,
        serving_container_image_uri: str,
        artifact_uri: Optional[str] = None,
        sync: bool = True,
    ):

        model = aiplatform.Model.upload(
            display_name=display_name,
            artifact_uri=artifact_uri,
            serving_container_image_uri=serving_container_image_uri,
            sync=sync,
            location=LOCATION
        )

        print(model.display_name)
        print(model.resource_name)
        print(model.name)
        return model

    def deploy_model_to_endpoint(self, model_name: str):
        # Deploy model to endpoint
        subprocess.check_call([
            'gcloud', 'ai', 'endpoints', 'deploy-model', str(END_POINT),
            '--region=australia-southeast1',
            '--model=' + model_name,
            '--display-name=housing_price_prediction',
            '--min-replica-count=1',
            '--max-replica-count=1',
            '--traffic-split=0=100'
        ], stderr=sys.stdout)

    def endpoint_predict_sample(self):
        endpoint = aiplatform.Endpoint(str(END_POINT))
        test_data = np.load("X_test.npy")
        prediction = endpoint.predict(instances=test_data.tolist())

        return prediction.predictions

    def evaluate_mode(self, predictions):
        grand_truth = np.load("y_test.npy")
        return mean_squared_error(grand_truth, predictions)

# Unit testing
if __name__ == "__main__":
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "vertex-ai-key.json"
    runner = Deployment_runner()

    # Read the best configuration from the JSON file
    with open(BEST_CONFIG_FILE, 'r') as f:
        best_config = json.load(f)
    runner.deploy_model_to_endpoint(model_name=best_config.trial_id)
    predictions = runner.endpoint_predict_sample()
    print(len(predictions))
    accuracy = runner.evaluate_mode(predictions=predictions)
    print(accuracy)
