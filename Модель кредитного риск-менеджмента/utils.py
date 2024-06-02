import os
import pandas as pd
import tqdm
from natsort import natsorted
import re

def preprocess (data):
    """
    Заменяет пустые значения нулем. 
    Преобразовывает все категориальные переменные в фиктивные / индикаторные переменные.
    Объединяет объекты с одним ID в один вектор признаков.
    Возвращает дата фрейм.
    """
    # заменяем все пустые значения на 0
    data.fillna('0', inplace=True)

    # Преобразуем бинаризированные категориальные переменные в фиктивные / индикаторные переменные.
    feat_data =list(data.columns.values)
    feat_data.remove('id'), feat_data.remove('rn')

    dummies = pd.get_dummies(data[feat_data], columns=feat_data)
    data_dummies = pd.concat([data, dummies], axis=1).drop(feat_data, axis=1)

    features = data_dummies.groupby('id').sum()

    return features

def read_parquet_dataset_from_local(path_to_dataset: str, 
                                    start_from: int = 0,
                                    num_parts_to_read: int = 2, 
                                    columns=None, 
                                    verbose=False
                                    ) -> pd.DataFrame:
    """
    читает num_parts_to_read партиций, преобразовывает их к pd.DataFrame и возвращает
    :param path_to_dataset: путь до директории с партициями
    :param start_from: номер партиции, с которой нужно начать чтение
    :param num_parts_to_read: количество партиций, которые требуется прочитать
    :param columns: список колонок, которые нужно прочитать из партиции
    :return: pd.DataFrame
    """

    res = []
    dataset_paths = natsorted([os.path.join(path_to_dataset, filename) for filename in os.listdir(path_to_dataset)
                              if filename.startswith('train')]) # использовал natsorted чтобы сортировка проводилась нативно
    print('\n'.join(dataset_paths))

    start_from = max(0, start_from)
    chunks = dataset_paths[start_from: start_from + num_parts_to_read]
    if verbose:
        print('Reading chunks:\n')
        for chunk in chunks:
            print(chunk)
    for chunk_path in tqdm.tqdm_notebook(chunks, desc="Reading dataset with pandas"):
        print('chunk_path', chunk_path)
        chunk = pd.read_parquet(chunk_path,columns=columns)
        res.append(chunk)

    return pd.concat(res).reset_index(drop=True)


def prepare_transactions_dataset(path_to_dataset: str, 
                                 num_parts_to_preprocess_at_once: int = 1, 
                                 num_parts_total: int=50,
                                 save_to_path=None, 
                                 verbose: bool=False
                                 ):
    """
    возвращает готовый pd.DataFrame с признаками, на которых можно учить модель для целевой задачи
    path_to_dataset: str
        путь до датасета с партициями
    num_parts_to_preprocess_at_once: int
        количество партиций, которые будут одновременно держаться и обрабатываться в памяти
    num_parts_total: int
        общее количество партиций, которые нужно обработать
    save_to_path: str
        путь до папки, в которой будет сохранён каждый обработанный блок в .parquet-формате; если None, то не будет сохранён
    verbose: bool
        логирует каждую обрабатываемую часть данных
    """
    preprocessed_frames = []

    for step in tqdm.tqdm_notebook(range(0, num_parts_total, num_parts_to_preprocess_at_once),
                                   desc="Transforming transactions data"):
        transactions_frame = read_parquet_dataset_from_local(path_to_dataset, step, num_parts_to_preprocess_at_once,
                                                             verbose=verbose)


   #здесь должен быть препроцессинг данных
        transactions_frame = preprocess(transactions_frame)

   #записываем подготовленные данные в файл
        if save_to_path:
            block_as_str = str(step)
            if len(block_as_str) == 1:
                block_as_str = '00' + block_as_str
            else:
                block_as_str = '0' + block_as_str
            transactions_frame.to_parquet(os.path.join(save_to_path, f'processed_chunk_{block_as_str}.parquet'))

        preprocessed_frames.append(transactions_frame)
    return pd.concat(preprocessed_frames).fillna(0)


def feat_drop (df, feats=False, deleteOnce=False):
        cols = list(df.columns.values)
        if not feats:
            feats=set([re.sub('_\d+$', '', item) for item in cols])       
        cols_to_del = set()
        for feat in feats:
            for i in range(len(cols)):
                if cols[i].startswith(feat):
                    cols_to_del.add(cols[i])
                    if deleteOnce:
                        break
        return [x for x in cols if x not in list(cols_to_del)]

def drop_first_feat (df):
    cols = feat_drop(df, deleteOnce=True)
    return df[cols]

def remove_unimportant_features(df):
    cols = feat_drop(df, feats=['is_zero_util', 'pre_loans530', 'pre_loans_total_overdue', 'rn'])
    return df[cols]