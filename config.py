from datetime import datetime
# config.py

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
PROJECT_ID = "ml-coding-439503"
LOCATION = "australia-southeast1"
BUCKET = "gs://ml-coding"
DISPLAY_NAME = f"xgboost_model"
CONTAINER_URI = f"gcr.io/{PROJECT_ID}/xgboost-training:latest"

# Version configurations
TRAIN_VERSION = "xgboost-cpu.1-1"

# Image URIs
TRAIN_IMAGE = "gcr.io/cloud-aiplatform/training/{}:latest".format(TRAIN_VERSION)


# Directory configurations
MODEL_DIR = "{}/aiplatform-custom-job".format(BUCKET)
DATASET_DIR = "gs://boston_housing_data"

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