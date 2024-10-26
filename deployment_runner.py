import subprocess
import sys
import json
import logging
import numpy as np

from typing import Optional
from config import *
from delpoyment_config import *
from google.cloud import aiplatform
from sklearn.metrics import mean_squared_error

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Deployment_runner:
    def __init__(self):
        aiplatform.init(project=PROJECT_ID, location=LOCATION)
        logging.info("AI Platform initialized.")

    def upload_model_sample(
        self,
        display_name: str,
        serving_container_image_uri: str,
        artifact_uri: Optional[str] = None,
        sync: bool = True,
    ):
        logging.info(f"Uploading model: {display_name}")
        model = aiplatform.Model.upload(
            display_name=display_name,
            artifact_uri=artifact_uri,
            serving_container_image_uri=serving_container_image_uri,
            sync=sync,
            location=LOCATION
        )
        logging.info(f"Model uploaded: {model.display_name} - Resource Name: {model.resource_name}")
        return model

    def deploy_model_to_endpoint(self, model_name: str):
        # Deploy model to endpoint
        logging.info(f"Deploying model to endpoint: {model_name}")
        try:
            subprocess.check_call([
                'gcloud', 'ai', 'endpoints', 'deploy-model', str(END_POINT),
                '--region=australia-southeast1',
                '--model=' + model_name,
                '--display-name=housing_price_prediction',
                '--min-replica-count=1',
                '--max-replica-count=1',
                '--traffic-split=0=100'
            ], stderr=sys.stdout)
            logging.info("Model deployed successfully.")
        except subprocess.CalledProcessError as e:
            logging.error("Error deploying model to endpoint.", exc_info=e)


# Unit testing
if __name__ == "__main__":
    runner = Deployment_runner()


    subprocess.check_call(['gcloud', 'storage', "cp", f"{BUCKET}/{CURRENT_CONFIG_FILE}", f"{CURRENT_CONFIG_FILE}" ], stderr=sys.stdout)

    # Read the best configuration from the JSON file
    with open(BEST_CONFIG_FILE, 'r') as f:
        best_config = json.load(f)

    with open(CURRENT_CONFIG_FILE, 'r') as f:
        current_config = json.load(f)

    if best_config["metric_value"] < current_config["metric_value"]:
        trail_id = best_config["trial_id"]
        model = runner.upload_model_sample(
                            display_name="prediction",
                            serving_container_image_uri=DEPLOY_IMAGE,
                            artifact_uri=f"{BUCKET}/aiplatform-custom-job/{trail_id}/model"
                        )
        
        runner.deploy_model_to_endpoint(model_name=model.name)
        subprocess.check_call(['gcloud', 'storage', "cp",  f"{BEST_CONFIG_FILE}", f"{BUCKET}/{CURRENT_CONFIG_FILE}"], stderr=sys.stdout)
        logging.info("Best configuration file copied to bucket.")

