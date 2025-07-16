# Capstone Project for the MLOps Zoomcamp: Predicting Heart Attack Risks

## Problem statement

Cardiovascular diseases, including heart attacks, are the leading cause of death globally, claiming millions of lives each year. Early detection and prevention are critical to reducing the mortality rate associated with heart diseases. Despite advancements in medical technology, there remains a significant need for accessible and accurate tools that can predict the risk of heart attacks, enabling timely intervention and potentially saving lives.

To address this pressing issue, I developed a comprehensive heart attack risk prediction service as part of my capstone project for the DataTalksClub MLOps course ('https://github.com/DataTalksClub/mlops-zoomcamp'). This project utilizes a dataset of heart attack risk factors to build and deploy a machine learning model capable of predicting an individual's risk of experiencing a heart attack. The service is designed to be user-friendly and accessible, requiring only 13 input measurements to generate a prediction.

The project leverages state-of-the-art MLOps techniques to ensure robustness, scalability, and maintainability. Key features include:

- __Experiment Tracking and Model Registry__: Hyperparameter tuning and model training are tracked using MLflow, with the best model automatically registered for deployment.
- __Workflow Orchestration__: Prefect is used to orchestrate workflows, including model training, deployment, and monitoring.
- __Model Deployment__: The trained model is deployed as a web service using Flask, allowing users to access the prediction service via a web browser.
- __Model Monitoring__: The service includes comprehensive monitoring to detect data drift and ensure model performance, with alerts sent via email if anomalies are detected.
- __Reproducibility and Best Practices__: The project follows best practices in software development, including the use of Docker for containerization, pipenv for dependency management, and unit tests to ensure code quality.

This repository showcases the application of MLOps principles to a real-world problem, providing a valuable tool for predicting heart attack risk and demonstrating the effectiveness of modern machine learning operations.

## Instructions for use

### A. Installation
1. Install Git and docker-compose (or docker compose) 
2. Pull from GitHub using `git clone https://github.com/mohamedhamd32/2025_mlops_project_heart_attack.git`
3. Move to repo `cd 2025_mlops_project_heart_attack.git`
4. `docker-compose build` (or 'docker compose build') 
5. `docker-compose up` (or 'docker compose up') 
<img width="226" height="61" alt="Image" src="https://github.com/user-attachments/assets/9a040b62-d46d-4139-bee8-4b144332eb7c" />
<img width="1239" height="118" alt="Image" src="https://github.com/user-attachments/assets/f103e769-ac79-4483-88cd-173883a24654" />


### B. Training a Model
1. Train a model by opening prefect over `http://localhost:4200/`.
2. Go to `Deployments`
3. For the deployment `train-heart-attack-model` start `Quick run`. Now a hyperparameter tuning is performed and the best model is registered via MLflow.

perfect ui:

   <img width="1280" height="598" alt="Image" src="https://github.com/user-attachments/assets/179fd9d5-b0fc-4aa5-8344-cda0d6b327a0" />

mlflow tracking for best model: 

   <img width="1278" height="563" alt="Image" src="https://github.com/user-attachments/assets/6bdb27bb-093c-4af4-a033-3e930db0f94e" />

### C. Heart Attack Risk Prediction
1. __Important: You have to train a model first!__ Open `http://localhost:8000/` in your browser. Now you can input your health data an it returns the probability of increased heart attack risk using your best trained model.

mlflow ui with best model performance : 

  <img width="686" height="394" alt="Image" src="https://github.com/user-attachments/assets/2b3ba168-9bf3-4226-9325-bc95b64f27fa" /> 

### D. Monitoring Data Drift (only Gmail for sending supported!)
1. Register app password under https://myaccount.google.com/apppasswords
2. Create new prefect block with your email address and app password (You can also refer to `python src/create_email_block.py --sender your_email@gmail.com --sender_password your_gmail_app_password` for a programmatic solution)
   - Open prefect over `http://localhost:4200/`
   - Click on `Block`
   - Add block via `Add Block+`
   - Choose type `Email Server Credentials`
   - Set `Block Name` to `gmail`
   - Input your gmail username via `Username` and your app password set register in step 1 via `Password`
   - Set `SMPTServer` to `smtp.gmail.com`, `SMTP Type` to `SSL`, and `SMTP Port` to `465`
3. Monitor dataset ad hoc (you can also create a schedule) by going to `Deyploments` again. Start a custom run of deployment `monitor-heart-attack-data-drift`. Set `recipient` to an email address you want to send the data drift alert to.

here is example of generated report by evidently : 

   <img width="2270" height="677" alt="image" src="https://github.com/user-attachments/assets/38509b2e-98f2-49c7-9556-6ddd530cf40a" />


## Repository Contents

### Data
- __data/__: Contains all data.
  - __heart.csv__: The training and test data for heart attack prediction.
  - __reference.csv__: This is the reference dataset that is saved for monitoring. This can be exchanged with any dataset to measure data drift using 'src/monitor.py'. In the current case, the data set is a subset of heart.csv, where I deleted observations in such a way that a data drift is simulated.

### Scripts
- __src/__
  - __hyperopt_register_model.py__: This script performs automated hyperparameter optimization for a Support Vector Classifier (SVC) using Hyperopt. It utilizes MLflow for experiment tracking and model management, logging metrics and registering the best model found. The workflow, orchestrated with Prefect, includes data loading, preprocessing, scaling, and evaluating SVC models on a heart attack prediction dataset.
  - __helper.py__: The code defines a data preprocessing pipeline for a machine learning project using Prefect for workflow management. It includes tasks for reading a dataset, removing duplicates, splitting features and targets, splitting data into training and testing sets, and scaling features using standardization. 
  - __app.py__: Hosts a Flask prediction web service over http://localhost:8000/. Takes the best model registered before and uses it for prediction.
  - __templates/__: Stores the template used by the Flask app.
    - __index.html__: Simple web form for heart attack risk prediction.
  - __monitor.py__: Calculates metrics between current data (data/heart.csv) and reference data set (data/reference.csv) and sends out email.
  - __create_email_block.py__: Python helper to create a prefect email block to send monitoring alerts.

### Tests
- __tests/__: Contains a unit and an integration test.
  - __\_\_init\_\_.py__: init file for testing.
  - __unit_test.py__: Unit tests for data preprocessing. Tests the function 'preprocess_data' that is stored in src/helper.py and is used for training (hyperopt_register_model.py) and prediction (app.py).

### Environment
- __.gitignore__: Contains all files and directories that should be ignored for GitHub commits.
- __docker-compose.yaml__: This Docker Compose configuration defines three services (mlflow, prefect, and app) each built from their respective Dockerfiles and joined to a common network. The mlflow service exposes port 5000 and the prefect service exposes port 4200, while the app service exposes port 8000. All three services share a volume named mlflow-data mounted at /app/artifacts to facilitate data sharing between them. The PREFECT_API_URL environment variable is set for the prefect service to point to its API endpoint. The mlflow-data volume is defined to persist data, and a network is created to allow inter-service communication.
- __Dockerfile.app__: This Dockerfile uses pipenv to manage dependencies. It designates /app as the working directory, copies Pipfile and Pipfile.lock to install the necessary packages, and then copies the entire application code into the container. The Dockerfile exposes port 8000 for the web service and uses pipenv to run gunicorn with four workers to serve the application located at src.app:app on 0.0.0.0:8000.
- __Dockerfile.mlflow__: Containerizes the mlflow server that is accessed by training (to find and register best model) and prediction (to retrieve best model).
- __Dockerfile.prefect__: This Dockerfile sets up a Prefect environment and installs dependencies with pipenv. The Prefect server is exposed on port 4200. Upon starting the container, it runs a script to start the Prefect server, ensures the existence of a work pool, deploys two flows (hyperopt_register_model.py for training a model and monitor.py for monitoring data drift), and finally starts a Prefect worker within the pool.
- __Pipfile__: Contains the dependencies.
- __Pipfile.lock__: Contains the exact versions of all dependencies and their dependencies.
- __prefect.yaml__: Contains the .yaml file that stores this .git location to pull.
- __README.md__: This file.


