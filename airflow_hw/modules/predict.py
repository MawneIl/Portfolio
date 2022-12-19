# <YOUR_IMPORTS>
from pathlib import Path

import dill
import json
import os
import pandas as pd
from datetime import datetime

path = os.environ.get('PROJECT_PATH', '..')


# <YOUR_CODE>

def predict():
    model_filename = f"{path}/data/models/cars_pipe_{datetime.now().strftime('%Y%m%d%H%M')}.pkl"
    dir_with_json = f'{path}/data/test'

    def model_load(model_filename):
        with open(model_filename, 'rb') as file:
            model = dill.load(file)
        return model

    def test_files_list(directory):
        files = sorted(Path(directory).glob('*.json'))
        files = list(map(str, files))
        return files

    def data_load(files):
        temp = []
        for file in files:
            with open(file, 'r') as f: js = json.load(f)
            data = pd.DataFrame.from_dict([js])
            temp.append(data)
        return pd.concat(temp, ignore_index=True)

    model = model_load(model_filename)
    files = test_files_list(dir_with_json)
    data = data_load(files)
    y = model.predict(data)
    data['pred'] = y
    path_to_csv = f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv'
    with open(path_to_csv, 'w') as file: data[['id', 'pred']].to_csv(file, index=False)
    return


if __name__ == '__main__':
    predict()
