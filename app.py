from src.mlproject.exception import CustomException
from src.mlproject.logger import logging 
import sys
from src.mlproject.components.data_ingestion import DataIngestion,DataIngestionConfig
from src.mlproject.components.data_transformation import DataTransformationConfig,DataTransformation

if __name__ == "__main__":
    try:
        # data_ingestion_config = DataIngestionConfig()
        data_ingestion = DataIngestion()
        train_data_path,test_data_path = data_ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        data_transformation.initiate_data_transformation(train_data_path,test_data_path)

    except Exception as e :
        logging.info("Custom Exception")
        raise CustomException(e,sys)