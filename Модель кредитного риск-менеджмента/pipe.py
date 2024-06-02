# imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
import pandas as pd
import joblib
import re

from utils import prepare_transactions_dataset
 
# functions
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

def main():
    # variables
    path = 'data/'
    X = prepare_transactions_dataset(path, 3, 12)
    y = pd.read_csv('train_target.csv')['flag'].ravel()

    # pipeline
    preprocessor = Pipeline(steps=[
        ('drop_first_feat', FunctionTransformer(drop_first_feat)),
        ('remove_unimportant_features', FunctionTransformer(remove_unimportant_features))   
    ])


    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', HistGradientBoostingClassifier(
                                                random_state=38,
                                                class_weight='balanced',
                                                scoring='roc_auc',
                                                l2_regularization=0.6,
                                                max_iter=200, 
                                                min_samples_leaf=25
                                                ))
    ])

    pipe.fit(X,y)

    joblib.dump(pipe, 'models/pipe.pkl')


if __name__ == '__main__':
    main()