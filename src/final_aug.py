import pickle 
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from generate_mistakes import generate_mistakes

with open('nearest_dict_dx', 'rb') as f:
    nearest_dict = pickle.load(f)
    
def list_multiprocessing(param_lst,
                         func,
                         **kwargs):
    
    workers = kwargs.pop('workers')

    with Pool(workers) as p:
        apply_lst = [([params], func, i, kwargs) for i,params in enumerate(param_lst)]
        result = list(tqdm(p.imap(_apply_lst, apply_lst), total=len(apply_lst)))

    # lists do not need such sorting, but this can be useful later
    result=sorted(result,key=lambda x:x[0])
    return [_[1] for _ in result]

def _apply_lst(args):
    params, func, num, kwargs = args
    return num, func(*params,**kwargs)

def generate_mistakes_wrapper(name):
    global nearest_dict
    try:
        return generate_mistakes(name,nearest_dict)
    except:
        return name    
    
# re-generate the augs on the train dataset
trn_df = pd.read_csv('../data/proc_trn.csv')
trn_df_copy_corr = trn_df.copy()
trn_df_copy_corr = trn_df_copy_corr.set_index('id')

# remove the second class
trn_df_copy_corr = trn_df_copy_corr[trn_df_copy_corr.target.isin([0,1])]

# leave only correct names
trn_df_copy_corr['fullname'] = trn_df_copy_corr['fullname_true']
trn_df_copy_corr['target'] = 0
assert len(trn_df_copy_corr) == (trn_df_copy_corr.fullname==trn_df_copy_corr.fullname_true).sum()

# only trash dataframe
trn_df_copy_trash = trn_df[trn_df.target==2].copy().set_index('id')

# old errors
trn_df_copy_old_errors = trn_df[trn_df.target==1].copy().set_index('id')
trn_df_copy_old_errors.index = trn_df_copy_old_errors.index+10000000

# generate new errors 
trn_df_copy_new_errors = trn_df[trn_df.target!=2].copy().set_index('id')
trn_df_copy_new_errors['target'] = 1

names = list(trn_df_copy_new_errors.fullname_true.values)
aug_names = list_multiprocessing(names,generate_mistakes_wrapper, workers=6)
trn_df_copy_new_errors['fullname'] = aug_names

trn_df_copy_new_errors.index = trn_df_copy_new_errors.index+20000000

new_trn_df = pd.concat([trn_df_copy_corr,trn_df_copy_trash,trn_df_copy_old_errors,trn_df_copy_new_errors])

# augment the best ensemble now
test_df = pd.read_csv('../data/test.csv')
test_df = test_df.set_index('id')
best_ensemble = pd.read_csv('predictions/stack_v4.csv')

best_ensemble = best_ensemble.set_index('id')
blank_ids = best_ensemble[pd.isnull(best_ensemble.fullname_true)].index.values
best_ensemble.loc[blank_ids,'fullname_true'] = test_df.loc[blank_ids,'fullname']
best_ensemble['country'] = 44

# take all correct values
best_ensemble_correct = best_ensemble[best_ensemble.target!=2].copy()
best_ensemble_correct['fullname'] = best_ensemble_correct['fullname_true'] 

cols = ['fullname', 'fullname_true', 'target', 'country']
best_ensemble_correct = best_ensemble_correct[cols]

best_ensemble_correct.index = best_ensemble_correct.index+30000000
best_ensemble_correct['target'] = 0

best_ensemble_errors = best_ensemble[best_ensemble.target!=2].copy()
best_ensemble_errors['target'] = 1

names = list(best_ensemble_errors.fullname_true.values)
aug_names = list_multiprocessing(names,generate_mistakes_wrapper, workers=6)

best_ensemble_errors['fullname'] = aug_names

best_ensemble_errors.index = best_ensemble_errors.index+40000000

# take all trash
best_ensemble_trash = best_ensemble[best_ensemble.target==2].copy()
best_ensemble_trash.index = best_ensemble_trash.index+50000000
best_ensemble_trash['fullname'] = best_ensemble_trash['fullname_true']

cols = ['fullname', 'fullname_true', 'target', 'country']
best_ensemble_trash = best_ensemble_trash[cols]
best_ensemble_correct = best_ensemble_correct[cols]
best_ensemble_errors = best_ensemble_errors[cols]

new_pseudo_df = pd.concat([best_ensemble_trash,best_ensemble_correct,best_ensemble_errors])

new_dataset = pd.concat([new_trn_df,new_pseudo_df])

new_pseudo_df.to_csv('../data/new_proc_train_only_pseudo.csv')
new_dataset.to_csv('../data/new_proc_train_and_pseudo.csv')
new_trn_df.to_csv('../data/new_proc_train.csv')