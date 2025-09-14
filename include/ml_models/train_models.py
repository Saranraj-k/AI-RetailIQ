import yaml
from include.utils.mlflow_utils import MLflowManager
from include.feature_engineering import FeatureEngineer
from include.data_validation.validators import DataValidator
from include.ml_models.ensemble_model import EnsembleModel
from include.ml_models.model_visualization import ModelVisualizer
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
    
    def train_xgboost(self,X_train:pd.DataFrame, y_train:pd.Series, X_val:pd.DataFrame, y_val:pd.Series,
                      use_optuna:bool = True):
        logger.info("Training XGBoost model...")
        if use_optuna:
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 0.5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
                    'random_state': 42
                }
                params['early_stopping_rounds'] = 50
                models = xgb.XGBRegressor(**params)
                models.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                y_pred = models.predict(X_val)
                return mean_squared_error(y_val, y_pred)
            study = optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=42),
                                        pruner=optuna.pruners.MedianPruner())
            study.optimize(objective, n_trials=self.config['training'].get('optuna_trials',50))
            best_params = study.best_params
            best_params['random_state'] = 42
        else:
            best_params = self.model_config['xgboost']['params']
        
        best_params['early_stopping_rounds'] = 50
        model = xgb.XGBRegressor(**best_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
        
        self.models['xgboost'] = model
        return model
    
    def caculate_metrics(self,y_true:np.ndarray, y_pred:np.ndarray):
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
        return metrics
    
    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray,
                      use_optuna: bool = True) -> lgb.LGBMRegressor:
        
        logger.info("Training LightGBM model")
        
        if use_optuna:
            def objective(trial):
                params = {
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
                    'random_state': 42,
                    'verbosity': -1,
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt'
                }
                
                model = lgb.LGBMRegressor(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                         callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
                
                y_pred = model.predict(X_val)
                return np.sqrt(mean_squared_error(y_val, y_pred))
            
            study = optuna.create_study(
                direction='minimize',
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner()
            )
            study.optimize(objective, n_trials=self.config['training'].get('optuna_trials', 50))
            
            best_params = study.best_params
            best_params['random_state'] = 42
            best_params['verbosity'] = -1
        else:
            best_params = self.model_config['lightgbm']['params']
        
        model = lgb.LGBMRegressor(**best_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                 callbacks=[lgb.early_stopping(50)])
        
        self.models['lightgbm'] = model
        return model
    
    def train_prophet(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                     date_col: str = 'date', target_col: str = 'sales') -> Prophet:
        
        logger.info("Training Prophet model")
        
        # Prepare data for Prophet
        prophet_train = train_df[[date_col, target_col]].rename(
            columns={date_col: 'ds', target_col: 'y'}
        )
        
        # Remove any NaN values
        prophet_train = prophet_train.dropna()
        
        # Ensure dates are sorted
        prophet_train = prophet_train.sort_values('ds')
        
        # Initialize Prophet with simplified parameters to avoid memory issues
        prophet_params = self.model_config['prophet']['params'].copy()
        
        # Override some parameters for stability
        prophet_params.update({
            'stan_backend': 'CMDSTANPY',  # Use cmdstanpy backend
            'mcmc_samples': 0,  # Disable MCMC for speed and stability
            'uncertainty_samples': 100,  # Reduce uncertainty samples
        })
        
        try:
            model = Prophet(**prophet_params)
            
            # Add only essential regressors to reduce complexity
            numeric_cols = train_df.select_dtypes(include=[np.number]).columns
            regressor_cols = [col for col in numeric_cols if col not in [target_col, 'year', 'month', 'day', 'week', 'quarter']]
            
            # Limit to top 5 most important regressors based on variance
            if len(regressor_cols) > 5:
                variances = {col: train_df[col].var() for col in regressor_cols}
                regressor_cols = sorted(variances.keys(), key=lambda x: variances[x], reverse=True)[:5]
            
            for col in regressor_cols:
                if train_df[col].std() > 0:  # Only add regressors with variance
                    model.add_regressor(col)
                    prophet_train[col] = train_df[col]
            
            # Fit the model with error handling
            model.fit(prophet_train)
            
            self.models['prophet'] = model
            return model
            
        except Exception as e:
            logger.error(f"Prophet training failed with parameters: {e}")
            # Try with minimal configuration
            logger.info("Retrying Prophet with minimal configuration...")
            
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
                uncertainty_samples=50,
                mcmc_samples=0
            )
            
            # Train without any additional regressors
            model.fit(prophet_train[['ds', 'y']])
            
            self.models['prophet'] = model
            return model

    def train_all_models(self,train_df:pd.DataFrame, val_df:pd.DataFrame, test_df: pd.DataFrame,
                         target_col:str = 'sales',use_optuna:bool = True):
        results={}
        run_id = self.mlflow_manager.start_run(
            run_name=f"sales_forecast_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags={'model_type':'ensemble','use_optuna':str(use_optuna)}
        )       
        logger.info("Starting training for all models...")
        try:
            X_train, y_train, X_val, y_val, X_test, y_test = self.preprocess_features(
                train_df, val_df, test_df, target_col=target_col
            )
            self.mlflow_manager.log_params({
                'train_size':len(train_df),
                'val_size':len(val_df),
                'test_size':len(test_df),
                'n_features':len(self.feature_cols)
            })
            ##Train XGB
            xgbmodel = self.train_xgboost(X_train, y_train, X_val, y_val, use_optuna=use_optuna)
            xgb_pred = xgbmodel.predict(X_test)
            xgb_metrics = self.calculate_metrics(y_test, xgb_pred)
            self.mlflow_manager.log_metrics({
                f"xgboost_{k}": v for k, v in xgb_metrics.items()
            })
            self.mlflow_manager.log_model(xgbmodel, "xgboost",input_example=X_train.iloc[:5])
            feature_importance = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': xgbmodel.feature_importances_
            }).sort_values(by='importance', ascending=False)
            logger.info(f"Top XGBoost features:\n{feature_importance.to_string(index=False)}")
            self.mlflow_manager.log_params({f"xgb_top_feature_{i}": f"{row['feature']} ({row['importance']:.4f})" 
                                           for i, (_, row) in enumerate(feature_importance.iterrows())})
            
            results['xgboost'] = {
                'model': xgb_model,
                'metrics': xgb_metrics,
                'predictions': xgb_pred
            }
             # Train LightGBM
            lgb_model = self.train_lightgbm(X_train, y_train, X_val, y_val, use_optuna)
            lgb_pred = lgb_model.predict(X_test)
            lgb_metrics = self.calculate_metrics(y_test, lgb_pred)
            
            self.mlflow_manager.log_metrics({f"lightgbm_{k}": v for k, v in lgb_metrics.items()})
            self.mlflow_manager.log_model(lgb_model, "lightgbm",
                                         input_example=X_train.iloc[:5])
            
            # Log feature importance for LightGBM
            lgb_importance = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': lgb_model.feature_importances_
            }).sort_values('importance', ascending=False).head(20)
            
            logger.info(f"Top LightGBM features:\n{lgb_importance.to_string()}")
            
            results['lightgbm'] = {
                'model': lgb_model,
                'metrics': lgb_metrics,
                'predictions': lgb_pred
            }

            #weighted ensemble based on individual model performance
            xgb_val_pred = xgbmodel.predict(X_val)
            lgb_val_pred = lgb_model.predict(X_val)
            xgb_val_r2 = r2_score(y_val, xgb_val_pred)
            lgb_val_r2 = r2_score(y_val, lgb_val_pred)
            min_weight = 0.2
            xgb_weight = max(min_weight, xgb_val_r2/(xgb_val_r2 + lgb_val_r2))
            lgb_weight = max(min_weight, lgb_val_r2/(xgb_val_r2 + lgb_val_r2))
            total_weight = xgb_weight + lgb_weight
            xgb_weight /= total_weight
            lgb_weight /= total_weight
            logger.info(f"Ensemble weights - XGBoost: {xgb_weight:.2f}, LightGBM: {lgb_weight:.2f}")
            ensemble_weights = {'xgboost': xgb_weight, 'lightgbm': lgb_weight}
            ensemble_pred = (xgb_weight * xgb_pred) + (lgb_weight * lgb_pred)
            #create ensemble model object
            ensemble_models={
                'xgboost': xgbmodel,
                'lightgbm': lgb_model
            }
            ensemble_models = EnsembleModel(ensemble_models, ensemble_weights)
            self.models['ensemble'] = ensemble_models
            ensemble_metrics = self.calculate_metrics(y_test, ensemble_pred)
            self.mlflow_manager.log_metrics({f"ensemble_{k}": v for k, v in ensemble_metrics.items()})
            self.mlflow_manager.log_model(ensemble_models, "ensemble_model",
                                         input_example=X_train.iloc[:5])
            results['ensemble'] = {
                'model': ensemble_models,
                'metrics':ensemble_metrics,
                'predictions': ensemble_pred}
            
            logger.info("Generating Visualizations for model comparison..")
            try:
                self._generate_and_log_visualizations(results, X_test, target_col=target_col)
            except Exception as ve:
                logger.error(f"Error generating visualizations: {ve}")

            self.save_artifacts()
            current_run_id = mlflow.active_run().info.run_id
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            self.mlflow_manager.end_run(run_id, status='FAILED')
            raise e
    
    
    def _generate_and_log_visualizations(self, results: Dict[str, Any], 
                                       test_df: pd.DataFrame, 
                                       target_col: str = 'sales') -> None:
        """Generate and log model comparison visualizations to MLflow"""
        try:
            from ml_models.model_visualization import ModelVisualizer
            import tempfile
            import os
            
            logger.info("Starting visualization generation...")
            visualizer = ModelVisualizer()
            
            # Extract metrics
            metrics_dict = {}
            for model_name, model_results in results.items():
                if 'metrics' in model_results:
                    metrics_dict[model_name] = model_results['metrics']
            
            # Prepare predictions data
            predictions_dict = {}
            for model_name, model_results in results.items():
                if 'predictions' in model_results and model_results['predictions'] is not None:
                    pred_df = test_df[['date']].copy()
                    pred_df['prediction'] = model_results['predictions']
                    predictions_dict[model_name] = pred_df
            
            # Extract feature importance if available
            feature_importance_dict = {}
            for model_name, model_results in results.items():
                if model_name in ['xgboost', 'lightgbm'] and 'model' in model_results:
                    model = model_results['model']
                    if hasattr(model, 'feature_importances_'):
                        importance_df = pd.DataFrame({
                            'feature': self.feature_cols,
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False)
                        feature_importance_dict[model_name] = importance_df
            
            # Create temporary directory for visualizations
            with tempfile.TemporaryDirectory() as temp_dir:
                logger.info(f"Creating visualizations in temporary directory: {temp_dir}")
                
                # Generate all visualizations
                saved_files = visualizer.create_comprehensive_report(
                    metrics_dict=metrics_dict,
                    predictions_dict=predictions_dict,
                    actual_data=test_df,
                    feature_importance_dict=feature_importance_dict if feature_importance_dict else None,
                    save_dir=temp_dir
                )
                
                logger.info(f"Generated {len(saved_files)} visualization files: {list(saved_files.keys())}")
                
                # Log each visualization to MLflow
                for viz_name, file_path in saved_files.items():
                    if os.path.exists(file_path):
                        mlflow.log_artifact(file_path, "visualizations")
                        logger.info(f"Logged visualization: {viz_name} from {file_path}")
                    else:
                        logger.warning(f"Visualization file not found: {file_path}")
                
                # Also create a combined HTML report
                self._create_combined_html_report(saved_files, temp_dir)
                
                # Log the combined report
                combined_report = os.path.join(temp_dir, 'model_comparison_report.html')
                if os.path.exists(combined_report):
                    mlflow.log_artifact(combined_report, "reports")
                    logger.info("Logged combined HTML report")
                    
        except Exception as e:
            logger.error(f"Failed to generate visualizations: {e}")
            # Don't fail the entire training if visualization fails

    def save_artifacts(self):
        # Save scalers and encoders
        joblib.dump(self.scalers, '/tmp/scalers.pkl')
        joblib.dump(self.encoders, '/tmp/encoders.pkl')
        joblib.dump(self.feature_cols, '/tmp/feature_cols.pkl')
        
        # Save individual models in the expected format
        import os
        os.makedirs('/tmp/models/xgboost', exist_ok=True)
        os.makedirs('/tmp/models/lightgbm', exist_ok=True)
        os.makedirs('/tmp/models/ensemble', exist_ok=True)
        
        if 'xgboost' in self.models:
            joblib.dump(self.models['xgboost'], '/tmp/models/xgboost/xgboost_model.pkl')
        
        if 'lightgbm' in self.models:
            joblib.dump(self.models['lightgbm'], '/tmp/models/lightgbm/lightgbm_model.pkl')
            
        if 'ensemble' in self.models:
            joblib.dump(self.models['ensemble'], '/tmp/models/ensemble/ensemble_model.pkl')
        
        self.mlflow_manager.log_artifacts('/tmp/')
        
        logger.info("Artifacts saved successfully")