
import yaml
import os
import logging
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

class MLflowManager:
    def __init__(self,config_path:str'/usr/local/airflow/include/config/ml_config.yaml'):
        with open(config_path,'r') as file:
            self.config = yaml.safe_load(file)
        mlflow_config = self.config['mlflow']
        self.tracking_uri = get_mlflow_endpoint()
        self.experiment_name = mlflow_config['experiment_name']
        self.registry_name = mlflow_config['registry_name']
        mlflow.set_tracking_uri(self.tracking_uri)
        try:
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            logger.warning(f"Experiment {self.experiment_name} failed to set: {e}")
            if 'mlflow' in self.tracking_uri:
                self.tracking_uri = 'http://localhost:5001'
                mlflow.set_tracking_uri(self.tracking_uri)
                os.environ['MLFLOW_TRACKING_URI'] = self.tracking_uri
                logger.info(f"Retrying with localhost: {self.tracking_uri}")
                try :
                    mlflow.set_experiment(self.experiment_name)
                except Exception as e2:
                    logger.error(f"Experiment {self.experiment_name} failed to set again: {e2}")
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = get_minio_endpoint()
        