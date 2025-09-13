import yaml
from include.utils.mlflow_utils import MLflowManager
from include.feature_engineering import FeatureEngineer
from include.data_validation.validators import DataValidator
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import yaml
import joblib
import logging
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

import xgboost as xgb
import lightgbm as lgb
from prophet import Prophet
import optuna
import mlflow
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self,config_path:str'/usr/local/airflow/include/config/ml_config.yaml'):
        with open(config_path,'r') as file:
            self.config = yaml.safe_load(file)
        self.models_config = self.config['models']
        self.training_config = self.config['training']
        self.mlflow_manager = MLflowManager(config_path)
        self.feature_engineer = FeatureEngineer(config_path)
        self.data_validator =  DataValidator(config_path)

        self.models = {}
        self.scalers = {}
        self.encoders = {}

    def prepare_data(self,df : pd.DataFrame, target_col:str = 'sales',date_col:str = 'date'
                     , group_cols: Optional[List[str]] = None, categorical_cols: Optional[List[str]] = None):
        logger.info("Preparing data for training...")
        required_cols = ['date', target_col]
        if group_cols:
            required_cols.extend(group_cols)
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        df_features = self.feature_engineer.create_all_features(
            df, date_col=date_col, target_col=target_col,
            group_cols=group_cols, categorical_cols=categorical_cols
        )
        #Split the data chronologically for the time series
        df_sorted = df_features.sort_values(by=date_col)
        train_size = int(len(df_sorted) * (1-self.training_config['test_size']-self.training_config['validation_size']))
        val_size = int(len(df_sorted) * self.training_config['validation_size'])
        train_df = df_sorted[:train_size]
        val_df = df_sorted[train_size:train_size+val_size]
        test_df = df_sorted[train_size+val_size:]
        train_df = train_df.dropna(subset=[target_col])
        val_df = val_df.dropna(subset=[target_col])
        test_df = test_df.dropna(subset=[target_col])
        logger.info(f"Data split into train({len(train_df)}), val({len(val_df)}), test({len(test_df)})")
        return train_df, val_df, test_df
    
    def preprocess_features(self,train_df:pd.DataFrame, val_df:pd.DataFrame, test_df:pd.DataFrame
                            , target_col:str , exclude_cols: List[str] = ['date']):
        logger.info("Preprocessing features...")
        feature_cols = [col for col in train_df.columns if col not in exclude_cols + [target_col]]
        X_train = train_df[feature_cols].copy()
        y_train = train_df[target_col].values()
        X_val = val_df[feature_cols].copy()
        y_val = val_df[target_col].values()
        X_test = test_df[feature_cols].copy()
        y_test = test_df[target_col].values()
        #encode categorical vars
        categorical_cols = X_train.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                X_train.loc[:,col] = self.encoders[col].fit_transform(X_train[col].astype(str))
            else:
                X_train.loc[:,col] = self.encoders[col].transform(X_train[col].astype(str))
            X_val.loc[:,col] = self.encoders[col].transform(X_val[col].astype(str))
            X_test.loc[:,col] = self.encoders[col].transform(X_test[col].astype(str))
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_cols, index=X_val.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
        self.scalers['standard'] = scaler
        self.feature_cols = feature_cols
        return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test