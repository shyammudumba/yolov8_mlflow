import os
from pathlib import Path

import yaml
from ultralytics import YOLO

from utils import save_metrics_and_params, save_model


root_dir = Path(__file__).resolve().parents[1]  # root directory absolute path
data_dir = os.path.join(root_dir, "data")
data_yaml_path = os.path.join(root_dir, "data.yaml")
metrics_path = os.path.join(root_dir, 'reports/train_metrics.json')


if __name__ == '__main__':

    # load the configuration file 
    with open(r"params.yaml") as f:
        params = yaml.safe_load(f)
    
    pre_trained_model = YOLO(params['model_type'])

    # train 
    model = pre_trained_model.train(
        data=data_yaml_path,
        imgsz=params['imgsz'],
        batch=params['batch'],
        epochs=params['epochs'],
        optimizer=params['optimizer'],
        lr0=params['lr0'],
        seed=params['seed'],
        pretrained=params['pretrained'],
        name=params['name']
    )

    # save model
    save_model(experiment_name=params['name'])

    # save metrics csv file and training params 
    save_metrics_and_params(experiment_name=params['name'])



         










