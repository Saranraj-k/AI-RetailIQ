import yaml
class ModelTrainer:
    def __init__(self,config_path:str'/usr/local/airflow/include/config/ml_config.yaml'):
        with open(config_path,'r') as file:
            self.config = yaml.safe_load(file)
        pass
