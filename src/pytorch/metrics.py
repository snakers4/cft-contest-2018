import numpy as np
from sklearn.metrics import f1_score

def score_task1(df_true, df_pred):
    df_true = df_true.set_index('id')
    df_pred = df_pred.set_index('id')
    df_pred = df_pred.loc[df_true.index]
    return f1_score(df_true.target, df_pred.target, average='macro')


def score_task2(df_true, df_pred):
    df_true = df_true.set_index('id')
    df_pred = df_pred.set_index('id')
    df_pred = df_pred.loc[df_true.index]

    # false correction if no target == 1
    df_pred.loc[df_pred.target != 1] = ''
    return np.mean(df_true.loc[df_true.target == 1, 'fullname_true']
                   == df_pred.loc[df_true.target == 1, 'fullname_true'])