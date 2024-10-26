# Machine Learning Pipeline and CI/CD Process Documentation

## Overview
This documentation outlines the implementation of a machine learning pipeline for housing price prediction, including data preparation, model training, hyperparameter tuning, deployment, and continuous integration/continuous deployment (CI/CD) processes.

## 1. Data Preparation and Training Process

### Data Processing
- **Data Structure**: Implemented custom data arrangement where alternate rows contain last 2 features and housing prices
- **Data Partitioning**: 
  - Training set: 80%
  - Validation set: 15%
  - Testing set: 5%

### Feature Engineering
- **Target Variable Treatment**: Applied logarithmic transformation to housing prices to address right-skewed distribution
- **Feature Scaling**: Minimal data normalization implemented as XGBoost performance showed no significant improvement with scaling

### Implementation Details
- Data is serialized to NumPy format and stored in cloud storage for hyperparameter tuning
- Primary training logic is implemented in `custom/trainer/task.py`
- Model evaluation uses Mean Squared Error (MSE) as the primary metric
- Complete process flow is documented in `run.py`, which serves as a reference for unit testing

## 2. Hyperparameter Tuning Framework

### Process Flow
1. Execution managed by `hyperparameter_tuner_runner.py`
2. Custom training package is compressed and uploaded to Google Cloud Storage
3. Hyperparameter tuning job is initiated with specified configuration

### Key Components
- **Model Artifacts**: Each training iteration saves model artifacts and metrics to Google Cloud Storage
- **Model Selection**: Best configurations are selected based on trial performance metrics
- **Configuration Management**:
  - Core settings (Project ID, Location, Training Version) managed in `config.py`
  - Hyperparameter and search space defined in `hyperparameter_tune_setting.yaml`

## 3. Model Deployment Pipeline

### Deployment Logic
- Implemented in `deployment_runner.py`
- Compares best tuned configuration against current production model using MSE
- Automated deployment triggers on performance improvement

### Configuration
- Deployment parameters managed in `deployment_config.py`
- Supports endpoint-specific configurations

## 4. Testing Framework

### Endpoint Testing
- Implemented in `test_endpoint_prediction.py`
- Performs inference requests using test dataset
- Validates predictions against ground truth data

## 5. CI/CD Pipeline

### Workflow Components
1. Repository checkout
2. Local training validation
3. Custom package preparation and cloud upload
4. Dependency management
5. Hyperparameter tuning execution
6. Endpoint testing and validation
7. Performance evaluation and logging

### Implementation
- Workflow defined in GitHub Actions YAML configuration
- Automates entire pipeline from testing to deployment
- Includes comprehensive logging and validation steps

## 6. Current Limitations and Future Improvements

### Technical Challenges
- Google Cloud Platform integration complexity
- GitHub Actions workflow optimization
- Resource constraints due to full-time SWE commitments

### Proposed Enhancements
- Implement comprehensive model configuration logging
- Isolated CI/CD process by components modification, instead of trigger by push
- Introduce parallel pipeline execution
- Enhance monitoring and observability
- Implement automated rollback mechanisms
- Add more sophisticated validation gates

## 7. Documentation and Resources
- **Video Documentation**: [Loom Walkthrough](https://www.loom.com/share/59c870995a6b4f54800e567787fb75f2)
- **Source Code**: Available in project repository
- **Configuration Templates**: Available in respective YAML files

## 8. Conclusion
This implementation demonstrates a production-grade machine learning pipeline with automated training, tuning, and deployment capabilities. The CI/CD integration ensures reliable and consistent model updates while maintaining code quality and performance standards.
