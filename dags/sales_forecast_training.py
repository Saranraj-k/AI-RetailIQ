from datetime import datetime, timedelta
from airflow.decorators import dag, task
import pandas as pd
import sys
sys.path.append("/usr/local/airflow/include")
from utils.data_generator import RealisticSalesDataGenerator
import logging

logger = logging.getLogger(__name__)

default_args = {
    'owner' : 'AI Governance Officer',
    'depends_on_past' : False,
    'start_date' : datetime(2025, 9, 1),
    'retries' : 1,
    'retry_delay' : timedelta(minutes=1),
    'catchup' : False,
    'schedule' : '@weekly'
}

@dag(
    default_args=default_args,
    description = 'Sales Forecasting Traiing DAG',
    tags = ['ml','training','sales_forecast','sales']
)

def sales_forecast_training():
    # Define the training workflow here
    @task()
    def extract_data_task():
        data_output_dir = '/tmp/sales_data'
        generator = RealisticSalesDataGenerator(
            start_date="2023-01-01",
            end_date="2025-09-01",
        )
        print("Generating real sales data...............")
        file_paths = generator.generate_sales_data(output_dir=data_output_dir)
        total_files = sum(len(paths) for paths in file_paths.values())
        print(f"Generated{total_files} files")
        for data_type, paths in file_paths.items():
            print(f"{data_type}:{len(paths)} files")
        return {
            'data_output_dir': data_output_dir,
            'file_paths': file_paths,
            'total_files':total_files
        }
    @task()
    def validate_data_task(extract_result):
        file_paths = extract_result['file_paths']
        total_rows = 0
        issues_found = []
        logger.info(f"Validate {len(file_paths["sales"])} sales files...")
        for i,sales_file in enumerate(file_paths['sales'][:10]):
            df=pd.read_parquet(sales_file)
            if i == 0:
                logger.info(f"sales data columns:{df.columns.tolist()}")
            if df.empty:
                issues_found.append(f"sales file {sales_file} is empty")
                continue
            required_cols = [
                'date',
                'store_id',
                'product_id',
                'quantity_sold',
                'revenue'
            ]
            missing_cols = set(required_cols)-set(df.columns)
            if missing_cols:
                issues_found.append(f"sales data {sales_file} missing cols: {missing_cols}")
                continue

            total_rows+=len(df)
            if df['quantity_sold'].min()<0:
                issues_found.append(f"sales data {sales_file} has negative quantity sold")
            if df['revenue'].min()<0:
                issues_found.append(f"sales data {sales_file} has negative revenue")
            for data_type in ['promotions','customer_traffic','store_events']:
                if data_type in file_paths and file_paths[data_type]:
                    sample_file = file_paths[data_type][0]
                    df=pd.read_parquet(sample_file)
                    logger.info(f"{data_type} data shape: {df.shape}")
                    logger.info(f"{data_type} data columns: {df.columns.tolist()}")
            validation_summary ={
                'total files validate':len(file_paths['sales'][:10]),
                'total_rows': total_rows,
                'issues_found':issues_found,
                'issues_count':issues_found[:5],
                'file_paths':file_paths
            }
            if issues_found:
                logger.error(f"validation sumary:{validation_summary}")
                for issue in issues_found:
                    logger.error(issue)
                raise Exception("Validation issues found")
            else:
                logger.info(f"validation summary:{validation_summary}")
        
    extract_result = extract_data_task()
    validation_summary = validate_data_task(extract_result)
sales_forecast_training()
        
