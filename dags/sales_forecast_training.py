from datetime import datetime, timedelta
from airflow.decorators import dag, task
import pandas as pd
import sys
sys.path.append("/usr/local/airflow/include")
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
        from utils.data_generator import RealisticSalesDataGenerator
        data_output_dir = '/tmp/sales_data'
        generator = RealisticSalesDataGenerator(
            start_date="2023-01-01",
            end_date="2025-09-01",
        )
        print("Generating real sales data...............")
        file_paths = generator.generate_sales_data(output_dir=data_output_dir)
        
