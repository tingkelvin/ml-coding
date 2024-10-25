import os
import subprocess
from hp_tunning_runner import Hp_Tunning_Runner
from deployment_runner import Deployment_runner

from config import *
from delpoyment_config import *

if __name__ == "__main__":
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "vertex-ai-key.json"

    # Step 0
    tar_file = 'custom.tar'
    tar_gz_file = 'custom.tar.gz'
    directory_to_archive = 'custom'
    # Remove existing files if they exist
    for file in [tar_file, tar_gz_file]:
        if os.path.exists(file):
            os.remove(file)
    # Create a tar file
    subprocess.run(['tar', 'cvf', tar_file, directory_to_archive], check=True)
    # Compress the tar file with gzip
    subprocess.run(['gzip', tar_file], check=True)
    # Copy the gzipped tar file to Google Cloud Storage
    subprocess.run(['gsutil', 'cp', tar_gz_file, f'{BUCKET}/trainer_boston.tar.gz'], check=True)

    # Step 1
    hp_runner = Hp_Tunning_Runner()
    best_model_id = hp_runner.create_and_run_hp_tuning_job()

    # Step 2
    deployment_runner = Deployment_runner()
    uploaded_model = deployment_runner.upload_model_sample(
                display_name="prediction",
                serving_container_image_uri=DEPLOY_IMAGE,
                artifact_uri=f"{BUCKET}/aiplatform-custom-job/{best_model_id}/model"
            )
    
    deployment_runner.deploy_model_to_endpoint(model_name=uploaded_model.name)
    predictions = deployment_runner.endpoint_predict_sample()
    accuracy = deployment_runner.evaluate_mode(predictions=predictions)
    print(accuracy)
