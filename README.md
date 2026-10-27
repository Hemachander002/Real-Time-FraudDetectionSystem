==> Data_source : https://www.kaggle.com/datasets/ealaxi/paysim1
==> I added a new column called tx_id to this dataset 
For Transaction and outcome simulation , i used the test split X of this dataset with the tx_id and created an another csv with the Test split Y but with the tx_id for the outcome simulation
### TO RUN THIS SYSTEM locally:
1. create a conda virtual environment  and install the libraries in the req.txt
2. install docker and run docker compose up -d and verify it with docker compose ps
3. install kafka-python and create an event "transactions"
4. run this command set MLFLOW_ALLOW_FILE_STORE=true in ur terminal before running the pipeline  
5. place ur transactions simulator data and the outcomes for it in the paths accordingly
6. once u train the initial model by running the pipeline (python -m fraud_pipeline.train) in the terminal
7. if u wanna perform the nightly exportation in s3 bucket , u can add the export_labeldata.py script in the run_system.bat ( refer to the code used to run the other scripts in the batch file )
8. run the run-system.bat to run the system
9. Troubleshooting
If you encounter errors:
  =>Check the comments in the relevant Python scripts for configuration requirements and execution notes.
  =>Do verify all the file paths in the python scripts and the bat file.
  =>Verify that all required services and Docker containers are running.
  =>Check database connectivity and Kafka broker configuration.
  =>Confirm that the input files and trained model exist at the configured paths.
  =>Review the terminal output and service logs to identify the failing component.

# Real-Time Fraud Detection & MLOps Platform
<img width="1913" height="1001" alt="dashboard4" src="https://github.com/user-attachments/assets/150cc684-d81a-4f8f-bbf6-4f481c990cf3" />

An end-to-end machine learning platform for detecting potentially fraudulent financial transactions in real time, monitoring inference and model performance, collecting delayed ground-truth outcomes, and retraining models through a champion–challenger evaluation workflow.
The project combines event-driven data processing, machine learning inference, operational monitoring, cloud data storage, and automated model lifecycle management.
### Overview
Traditional machine learning projects often stop after model training and evaluation. This project focuses on the complete lifecycle of a fraud detection system, from transactions ingestion and real-time prediction to delayed outcome collection, historical data management, model evaluation, and retraining.
The system simulates a continuous transaction stream as it occurs in real life but at 25 transactions per second in our case , processes transactions through Apache Kafka, generates fraud predictions using a trained ML model, and tracks actual outcomes when they become available.
Historical transaction and outcome data are archived in Amazon S3 for future training and evaluation.
### Transaction Processing Workflow
#### 1. Transaction Simulation
The transactions.py script simulates real-world transaction activity at approximately 25 transactions per second.
Generated transactions are inserted into the database, creating a continuous stream of records for downstream processing.
Each transaction has a unique transaction identifier (tx_id) that allows it to be tracked throughout the system.
#### 2. Kafka Event Publishing
The kafka_producer.py script retrieves unpublished transactions from the database and publishes them to the Kafka topic named transactions.
Kafka decouples transaction ingestion from model inference, allowing the two stages to operate independently.
#### 3. Real-Time Model Inference
The inference_consumer.py script consumes transaction events from Kafka broker.
The consumer separates the transaction identifier from the model features because the identifier is used for tracking rather than prediction.
The trained model generates a fraud prediction, and the inference result is stored alongside the associated transaction information for later analysis.
#### 4. Delayed Ground-Truth Collection
In real-world fraud detection, the actual outcome of a transaction may take hours or days to become available.
The outcome simulator reproduces this behavior by writing the actual outcomes associated with transaction IDs into the outcomes table.
This enables the system to evaluate historical predictions against subsequently observed outcomes.
#### 5. Uploading the Data for Future use
Every night we upload the transactions data that has been occured in the day with its ground truth in the S3 bucket and we call it as day_data 
#### 6. Model Performance Monitoring
SQL queries in grafana combine inference results with actual outcomes to calculate model evaluation metrics.
The dashboard provides visibility into:
=> Total transactions processed
=> Potentially fraudulent transactions
=> Flagged transaction IDs
=> Inference latency
=> Model performance metrics
=> Transaction and inference activity
=> Health and status of simulator and processing services
=> Model evaluation uses transactions whose actual outcomes have arrived, rather than treating unlabelled transactions as confirmed successes or failures.
<img width="1910" height="1035" alt="dashboard3" src="https://github.com/user-attachments/assets/5f209ca8-5f72-4062-9be9-8f23f0baba40" />
<img width="1917" height="993" alt="dashboard2" src="https://github.com/user-attachments/assets/71e2e42a-c4d8-4fb2-bc61-c165007ace8c" />
<img width="1909" height="943" alt="dashboard1" src="https://github.com/user-attachments/assets/1ae47499-66f8-4b2a-87f9-03be4e225cbd" />
#### 7. Retraining
Once if the model performance drops
=> we merge all the day data with the ultimate data we have trained our current model with 
=> and we call it as new ultimate data or version it with DVC
=> we start the training pipeline with the new data and register the new champion model in the MLflow
<img width="1918" height="1045" alt="MLFLOW" src="https://github.com/user-attachments/assets/357ed593-c4c1-43d7-8c4c-dd56d65bc25f" />
## Cloud Deployment and Limitations I have faced
I explored deploying the system using AWS services, including Amazon RDS, Amazon S3, Amazon EC2, and Amazon SageMaker.
- **Amazon RDS:** Successfully integrated for database operations.
- **Amazon S3:** Successfully integrated for storing daily transaction and outcome datasets.
- **Amazon EC2:** Tested for running the system's scripts, but limited compute resources affected performance.
- **Amazon SageMaker and AWS-managed Grafana:** Full integration was limited by service availability and resource constraints associated with my AWS account.
Due to these limitations, I currently run the resource-intensive processing, training pipeline, and monitoring components locally while retaining the working AWS integrations.
This hybrid approach allowed me to continue developing and testing the end-to-end system while exploring cloud deployment options. A fully cloud-hosted deployment remains a potential future improvement.
