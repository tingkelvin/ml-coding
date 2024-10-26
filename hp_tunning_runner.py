import yaml
import json
import os

from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt
from typing import Dict
from config import *

from datetime import datetime
# config.py

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

class Hp_Tunning_Runner:
    def __init__(self):
        aiplatform.init(
            project=PROJECT_ID,
            location=LOCATION,
            staging_bucket=BUCKET
        )

    def create_and_run_hp_tuning_job(self):
        """
        Creates and runs a hyperparameter tuning job on AI Platform.
        """
        # Initialize AI Platform


        # Job Configurations are loaded in config.py
        custom_job = aiplatform.CustomJob(
            display_name=f"DISPLAY_NAME_{timestamp}",
            worker_pool_specs=WORKER_POOL_SPEC ,
            staging_bucket=BUCKET,
            base_output_dir=MODEL_DIR
        )

        # Hyperparameter Configurations are loaded in hyperparameter_tune_setting.yaml
        config = self.parse_hyperparameter_config()
        hyperparameter_tune_settings = config['hyperparameter_tune_settings']
        parameter_config = hyperparameter_tune_settings['parameter_spec']
        parameter_settings = self.parse_parameter_settings(parameter_config)
        print(parameter_settings)

        # Create the hyperparameter tuning job
        hp_job = aiplatform.HyperparameterTuningJob(
            display_name='HP-XGBoostTunner',
            custom_job=custom_job,
            metric_spec=hyperparameter_tune_settings['metric_spec'],
            parameter_spec=parameter_settings,
            max_trial_count=hyperparameter_tune_settings['num_trials'],
            parallel_trial_count=hyperparameter_tune_settings['parallel_trials'],
            max_failed_trial_count=hyperparameter_tune_settings['max_failed_trials'],
            search_algorithm=None,
        )
        
        # Run the job
        hp_job.run()

        best = (None, None, None, None, None, float('inf'))
        # Iterate through the trails and update the best configuration
        for trial in hp_job.trials:
            print(trial)
            # Keep track of the best outcome
            if float(trial.final_measurement.metrics[0].value) < best[5]:
                best = (
                    trial.id,
                    float(trial.parameters[0].value),
                    float(trial.parameters[1].value),
                    float(trial.parameters[2].value),
                    float(trial.parameters[3].value),
                    float(trial.final_measurement.metrics[0].value),
                )

        # Print details of the best configuration
        print("Best Configuration:", best)

        # Write the best configuration to a JSON file
        with open(BEST_CONFIG_FILE, 'w') as f:
            json.dump({
                'trial_id': best[0],
                'learning_rate': best[1],
                'max_depth': best[2],
                'subsample': best[3],
                'parameter_3': best[4],
                'metric_value': best[5]
            }, f)

        print(f'Best configuration saved to {BEST_CONFIG_FILE}')
        

    def parse_hyperparameter_config(self, config_path="hyperparameter_tune_setting.yaml") -> Dict:
        with open(config_path, 'r') as file:
            config_dict = yaml.safe_load(file)

        # Now you can access your config as a dictionary
        return config_dict

    def parse_parameter_settings(self, parameter_config) -> Dict:
        parameter_settings = {}
        for paramter in parameter_config:
            min_value = parameter_config[paramter]['min']
            max_value = parameter_config[paramter]['max']
            scale = parameter_config[paramter]['scale']
            print(paramter, min_value, max_value, scale)
            if scale == "linear":
                parameter_settings[paramter] = hpt.IntegerParameterSpec(min_value, max_value, scale)
            if scale == "log":
                parameter_settings[paramter] = hpt.DoubleParameterSpec(min_value, max_value, scale)
        return parameter_settings

# Unit testing
if __name__ == "__main__":
    runner = Hp_Tunning_Runner()
    runner.create_and_run_hp_tuning_job()
    
    
    
