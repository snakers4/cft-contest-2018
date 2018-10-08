import torch
import numpy as np
import pandas as pd
import torch.utils.data as data
from sklearn.model_selection import StratifiedKFold

from utils.text_utils.text_utils import (
    pckl,
    upkl
    )

class TransformerDataset(data.Dataset):
    def __init__(self,
                 df_path='data/all_corpuses_2018_09_06_processed_le.feather', 
                 w2i_path='data/2018_09_07_encoding_w2i.pickle',
                 mode='train', # train, val or test
                 max_len=10,
                 fold=0,
                 target_column='s_id_encoded',
                 keyword_column='keyword',
                 source_column='source',
                 trainval_sources =['SEO_cluster','SEO_dash','Synonyms'],
                 test_source=['order','select','tag'],
                 random_state=42,
                 return_meta=False,
                 return_string=False):
        
        df = pd.read_feather(df_path)
        df['len'] = df[keyword_column].apply(lambda x: len(x.split()))
        
        train_df = df[df[source_column].isin(trainval_sources)]
        test_df = df[df[source_column].isin(test_source)]
        del df
        
        self.w2i = upkl(w2i_path)
        
        self.mode = mode
        self.fold = fold
        self.return_meta = return_meta
        self.return_string = return_string
        self.random_state = random_state

        self.train_texts = train_df[keyword_column].values
        self.train_targets = train_df[target_column].values
        self.train_lens = train_df['len'].values
        self.train_sources = train_df[source_column].values
        
        self.test_texts = test_df[keyword_column].values
        self.test_targets = test_df[target_column].values
        self.test_lens = test_df['len'].values
        self.test_sources = test_df[source_column].values
        
        self.max_len = max_len
        
        # check that unique tokens are present
        # otherwise add them to the dataset
        
        for new_tag in ['_start_','_pad_','_unk_','_classify_']:
            if new_tag not in self.w2i:
                self.w2i[new_tag] = len(self.w2i)
                print('{} tag added to the dataset with index {}'.format(new_tag,
                                                                         len(self.w2i)-1))               
        self.clf_token = self.w2i['_classify_']
        self.vocab_len = len(self.w2i)     
            
        # add position embeddings
        for i in range(0,self.max_len):
            pos_emb = '_pos{}_'.format(i)
            if pos_emb not in self.w2i:
                self.w2i[pos_emb] = len(self.w2i)
                print('{} tag added to the dataset with index {}'.format(pos_emb,
                                                                     len(self.w2i)-1))                   

        skf = StratifiedKFold(n_splits = 4,
                              shuffle = True,
                              random_state = self.random_state)
        """
        f1, f2, f3, f4 = skf.split(list(self.train_texts),
                                   list(self.train_targets))
 
        folds = [f1, f2, f3, f4]
        self.train_idx = folds[self.fold][0]
        self.val_idx = folds[self.fold][1] 
        """
    def __getitem__(self, idx):
        if self.mode == 'train':
            _idx = self.train_idx[idx]
        elif self.mode == 'val':
            _idx = self.val_idx[idx]

        if self.mode in ('train','val'):
            text,target,_len,source = (self.train_texts[_idx],
                                       self.train_targets[_idx],
                                       self.train_lens[_idx],
                                       self.train_sources[_idx])
            
        elif self.mode == 'test':
            text, target,_len,source = (self.test_texts[idx],
                                        self.test_targets[idx],
                                        self.test_lens[idx],
                                        self.test_sources[idx],
                                       )
        original_text = text
        text = text.split()
        
        # one token is "lost" for _classify_ token
        # one toke is "lost" for _start_ token 
        if len(text) > self.max_len-2:
            text = text[:self.max_len-2]

        # start with the _start_ token
        processed = [self.w2i['_start_']] * 1              

        processed.extend([self.w2i[w] if w in self.w2i
                          else self.w2i['_unk_'] for w in text])

        # add classification token for the transformer model after each phrase
        processed.extend([self.w2i['_classify_']] * 1)         
        
        l = len(processed)
        
        if len(processed) < self.max_len:
            processed.extend([self.w2i['_pad_']] * (self.max_len - len(processed)))
        
        processed = torch.tensor(processed).squeeze()
        
        # also embed positional information and masks
        # also account for the classification token
        mmb = np.zeros((self.max_len))
        mmb[:l] = 1
        mmb = torch.from_numpy(mmb)
        
        # position information that is added to the input embeddings in the TransformerModel
        # also account for the classification token
        
        pmb = np.arange(self.vocab_len, self.vocab_len + self.max_len)
        
        processed = torch.stack((processed,torch.from_numpy(pmb)),dim=1)
        
        if self.return_meta:
            return_tuple = [processed.float(),mmb.float(),target,_len,source]
        else:
            return_tuple = [processed.float(),mmb.float(),target]
            
        if self.return_string:
            return_tuple.append(original_text)
        
        return return_tuple
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_idx)
        elif self.mode == 'val':
            return len(self.val_idx)
        elif self.mode == 'test':
            return len(self.test_texts)  