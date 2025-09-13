import yaml
class ModelTrainer:
    def __init__(self,config_path:str'/usr/local/airflow/include/config/ml_config.yaml'):
        with open(config_path,'r') as file:
            self.config = yaml.safe_load(file)
        self.models_config = self.config['models']
        self.training_config = self.config['training']
        self.mlflow_manager = MLflowManager(config_path)
                
