import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv('PROJECT_ID')
LOCATION = os.getenv('LOCATION')
BUCKET= os.getenv('BUCKET')
TRAIN_VERSION = os.getenv('TRAIN_VERSION')
MODEL_DIR = os.getenv('MODEL_DIR')
DATASET_DIR = os.getenv('DATASET_DIR')
DISPLAY_NAME = os.getenv('DISPLAY_NAME')
CONTAINER_URI = os.getenv('CONTAINER_URI')
# Version configurations
TRAIN_VERSION="xgboost-cpu.1-1"

# Image URIs
TRAIN_IMAGE = os.getenv('TRAIN_IMAGE')

# Command line arguments
CMDARGS = [
    "--dataset_X_train_url=" + DATASET_DIR + "/X_train.npy",
    "--dataset_y_train_url=" + DATASET_DIR + "/y_train.npy",
    "--dataset_X_val_url=" + DATASET_DIR + "/X_val.npy",
    "--dataset_y_val_url=" + DATASET_DIR + "/y_val.npy",
]

# Worker pool specifications
WORKER_POOL_SPEC = [
    {
        "replica_count": 1,
        "machine_spec": {
            "machine_type": "n1-standard-8",
        },
        "python_package_spec": {
            "executor_image_uri": TRAIN_IMAGE,
            "package_uris": [BUCKET + "/trainer_boston.tar.gz"],
            "python_module": "trainer.task",
            "args": CMDARGS,
        },
    }
]