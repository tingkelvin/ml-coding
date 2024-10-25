# trainer/task.py
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import logging
import hypertune
from google.cloud import storage
import subprocess
import sys

SEED=123

def get_data(test_size=0.2, random_state=SEED):
    # Check Missing data
    if not args.run_locally:
        logging.info("Copy training data from gs bucket: {}, {}".format(args.dataset_X_train_url, args.dataset_y_train_url))
        # gsutil outputs everything to stderr. Hence, the need to divert it to stdout.
        subprocess.check_call(['gsutil', 'cp', args.dataset_X_train_url, 'X_train.npy' ], stderr=sys.stdout)
        subprocess.check_call(['gsutil', 'cp', args.dataset_y_train_url, 'y_train.npy'], stderr=sys.stdout)
        subprocess.check_call(['gsutil', 'cp', args.dataset_X_val_url, 'X_val.npy' ], stderr=sys.stdout)
        subprocess.check_call(['gsutil', 'cp', args.dataset_y_val_url, 'y_val.npy'], stderr=sys.stdout)
        
        logging.info("Finished Copying")
        
    X_train, X_val, y_train, y_val = np.load('X_train.npy'), np.load('X_val.npy'), np.load('y_train.npy'), np.load('y_val.npy')

    return X_train, X_val, y_train, y_val

def train_xgboost(args):
    """
    Train XGBoost model with given hyperparameters
    """
    
    # Split data
    X_train, X_val, y_train, y_val = get_data()
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Set up parameters
    params = {
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'subsample': args.subsample,
        'objective': 'reg:squarederror',  # Change based on your problem
    }
    
    # Training parameters
    training_params = {
        'params': params,
        'dtrain': dtrain,
        'num_boost_round': args.n_estimators,
        'evals': [(dtrain, 'train'), (dval, 'val')],
        'early_stopping_rounds': 10,
        'verbose_eval': True
    }
    
    # Train model
    model = xgb.train(**training_params)
    
    # Calculate validation error
    val_pred = model.predict(dval)
    val_loss = mean_squared_error(y_val, val_pred)

    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='loss',
        metric_value=val_loss
    )
    
    # model.save_model(f"/gcs/{args.model_dir}_model.bst")

    if not args.run_locally:
        # GCSFuse conversion
        gs_prefix = 'gs://'
        gcsfuse_prefix = '/gcs/'
        if args.model_dir.startswith(gs_prefix):
            args.model_dir = args.model_dir.replace(gs_prefix, gcsfuse_prefix)
            dirpath = os.path.split(args.model_dir)[0]
            if not os.path.isdir(dirpath):
                os.makedirs(dirpath)

        # Export the classifier to a file
        gcs_model_path = os.path.join(args.model_dir, 'model.bst')
        logging.info("Saving model artifacts to {}". format(gcs_model_path))
        # model.save_model(gcs_model_path)

        # GCSFuse conversion
        gs_prefix = 'gs://'
        gcsfuse_prefix = '/gcs/'
        if args.model_dir.startswith(gs_prefix):
            args.model_dir = args.model_dir.replace(gs_prefix, gcsfuse_prefix)
            dirpath = os.path.split(args.model_dir)[0]
            if not os.path.isdir(dirpath):
                os.makedirs(dirpath)

        # Export the classifier to a file
        gcs_model_path = os.path.join(args.model_dir, 'model.bst')
        logging.info("Saving model artifacts to {}". format(gcs_model_path))
        model.save_model(gcs_model_path)

        logging.info("Saving metrics to {}/loss.json". format(args.model_dir))
        gcs_metrics_path = os.path.join(args.model_dir, 'loss.json')
        with open(gcs_metrics_path, "w") as f:
            f.write(f"loss: {val_loss}")
        
    return model, val_loss

def get_args():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--model_dir', dest='model_dir',
        default=os.getenv('AIP_MODEL_DIR'), 
        type=str, help='Model dir.'
    )
    parser.add_argument(
        '--max_depth',
        type=int,
        default=6,
        help='Maximum tree depth'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.1,
        help='Learning rate'
    )
    parser.add_argument(
        '--subsample',
        type=float,
        default=1.0,
        help='Subsample ratio of training instances'
    )
    parser.add_argument(
        '--n_estimators',
        type=int,
        default=100,
        help='Number of boosting rounds'
    )
    parser.add_argument(
        "--dataset_X_train_url", 
        dest="dataset_X_train_url",
        type=str, help="Download url for the training data."
    )
    parser.add_argument(
        "--dataset_y_train_url", 
        dest="dataset_y_train_url",
        type=str, 
        help="Download url for the training data labels."
    )
    parser.add_argument(
        "--dataset_X_val_url", 
        dest="dataset_X_val_url",
        type=str, help="Download url for the training data."
    )
    parser.add_argument(
        "--dataset_y_val_url", 
        dest="dataset_y_val_url",
        type=str, 
        help="Download url for the training data labels."
    )

    parser.add_argument(
        "--run_locally", 
        default=False,
        type=bool, 
        help="Run script locally for testing."
    )
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    model, val_loss = train_xgboost(args)
    print(args)