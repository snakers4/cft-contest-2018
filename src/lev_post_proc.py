import pickle
import numpy as np
import pandas as pd
import editdistance
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Pool
from matplotlib_venn import venn3
from sklearn.feature_extraction.text import CountVectorizer

def pckl(obj,path):
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def upkl(path):
    with open(path, 'rb') as handle:
        _ = pickle.load(handle)
    return _

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

train_df = pd.read_csv('../data/proc_train.csv')
pred_df = pd.read_csv('predictions/stack_v1_full.csv')

clean_names = []

# only non-nonsense queries
l1 = list(train_df[train_df.target.isin([0,1])].fullname_true)
# only "clean queries"
l2 = list(pred_df[pred_df.target==0].fullname_true)

clean_names.extend(l1)
clean_names.extend(l2)
print(len(l1),len(l2),len(clean_names))

cv = CountVectorizer(lowercase=False,
                     ngram_range=(1,1),
                     analyzer='word')
cv.fit(clean_names)

vocab = cv.vocabulary_

cv = CountVectorizer(lowercase=False,
                     ngram_range=(1,1),
                     analyzer='word')
cv.fit(l1)

vocab_train = cv.vocabulary_

cv = CountVectorizer(lowercase=False,
                     ngram_range=(1,1),
                     analyzer='word')
cv.fit(l2)

vocab_test = cv.vocabulary_

print(len(vocab),len(vocab_train),len(vocab_test))

def count_new_words(text):
    global vocab
    new_words = 0
    words = text.split()
    
    for word in words:
        if word not in vocab:
            new_words += 1
            
    return new_words

def correct_name_dist(phrase,
                      len_diff=1):
    global vocab_arr,vocab_list_len
    
    # print(phrase)
    words = phrase.split()
    correct_phrase = []

    try:
        for word in words:
            if word not in vocab:
                similar_vocab = vocab_arr[(vocab_list_len>len(word)-len_diff-1)*(vocab_list_len<len(word)+len_diff+1)]
                distances = [editdistance.eval(name,word) for name in similar_vocab]
                idx = np.argsort(np.array(distances))[:5]
                correct_phrase.append(similar_vocab[idx[0]])
                # correct_phrase.append(get_close_matches(word, list(similar_vocab), n=1))
            else:
                correct_phrase.append(word)
    except:
        return phrase
    
    # print(' '.join(correct_phrase))
    
    return ' '.join(correct_phrase)

pred_df['new_words'] = pred_df['fullname_true'].apply(lambda x: count_new_words(x))

pred_df = pred_df.set_index('id')
vocab_list = list(vocab.keys())
cond = (pred_df.target==1)&(pred_df.new_words>0)
vocab_list_len = [len(_) for _ in vocab_list]
vocab_list_len = np.array(vocab_list_len)
vocab_arr = np.array(vocab_list)

ids = list(pred_df[cond].index.values)
phrases = list(pred_df.loc[ids,'fullname_true'])

pckl(ids,'lev_ids_to_fix')
pckl(phrases,'lev_phrases_to_fix')

correct_phrases = list_multiprocessing(phrases,
                                       correct_name_dist,
                                       workers=8)

pckl(correct_phrases,'lev_corr_phrases_to_fix')