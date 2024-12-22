import os 
from pathlib import Path
import logging


logging.basicConfig(level = logging.INFO)


project_name = 'mlproject'

list_of_file = [
    ".github/workflow/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_monitering.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/training_pipeline.py",
    f"src/{project_name}/pipeline/prediction_pipeline.py",
    f"src/{project_name}/exception.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/utils.py",
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py"
]


for filepath in list_of_file:
    filepath = Path(filepath)
    print(filepath)
    filedir,filename = os.path.split(filepath)
    print(filedir,filename)

    if filedir != "":
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath,'w') as f:
            pass
        logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info(f"{filename} is already exists")