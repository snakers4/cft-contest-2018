import torch
import numpy as np
import pandas as pd
import torch.utils.data as data
from pytorch.symbols import symbol_list
from sklearn.model_selection import StratifiedKFold

symbol_dict = dict(zip(sorted(symbol_list),
                       range(0,len(symbol_list))))

class MisprintDataset(data.Dataset):
    def __init__(self,
                 train_df_path='../data/train.csv',
                 test_df_path='../data/test.csv',
                 symbol_dict=symbol_dict, # only symbols in the pre-defined dict
                 mode='train', # train, val or test
                 
                 max_len=100, # good enough length for attention
                 fold=0,
                 
                 target_column='target',
                 name_column='fullname',
                 gt_name_column='fullname_true',
                 country_column='country',
                 
                 random_state=42,
                 return_meta=False,
                 return_string=False,
                 add_cn_embeddings=False):
        
        train_df = pd.read_csv(train_df_path)
        test_df = pd.read_csv(test_df_path)
        self.s2i = symbol_dict.copy()
        
        self.mode = mode
        self.fold = fold
        self.max_len = max_len
        self.random_state = random_state
        self.add_cn_embeddings = add_cn_embeddings

        
        cns = train_df.country.value_counts()
        top_countries = list(cns[cns>100].index)
        
        train_df['country'] = train_df['country'].apply(lambda x: x if x in top_countries else 'OTHER')
        train_df.loc[pd.isnull(train_df.fullname_true),
                     'fullname_true'] = train_df.loc[pd.isnull(train_df.fullname_true),
                                                     'fullname']
        
        test_df['country'] = test_df['country'].apply(lambda x: x if x in top_countries else 'OTHER')

        self.cn_dict = dict(zip(sorted(list(train_df['country'].unique())),
                                range(train_df['country'].nunique())))

        train_df['country'] = train_df['country'].apply(lambda x: self.cn_dict[x])
        test_df['country'] = test_df['country'].apply(lambda x: self.cn_dict[x])        
        

        self.train_names = train_df[name_column].values
        self.train_gt_names = train_df[gt_name_column].values
        self.train_targets = train_df[target_column].values
        self.train_countries = train_df[country_column].values
        
        self.test_names = test_df[name_column].values
        self.test_countries = test_df[country_column].values

        del train_df,test_df
        
        # check that unique tokens are present
        # otherwise add them to the dataset
        
        for new_tag in ['_start_','_pad_','_unk_','_classify_']:
            if new_tag not in self.s2i:
                self.s2i[new_tag] = len(self.s2i)
                print('{} tag added to the dataset with index {}'.format(new_tag,
                                                                         len(self.s2i)-1))               
        self.clf_token = self.s2i['_classify_']
        self.vocab_len = len(self.s2i)     
            
        # add position embeddings
        for i in range(0,self.max_len):
            pos_emb = '_pos{}_'.format(i)
            if pos_emb not in self.s2i:
                self.s2i[pos_emb] = len(self.s2i)
                print('{} tag added to the dataset with index {}'.format(pos_emb,
                                                                     len(self.s2i)-1))
                
        # add cn embeddings
        if add_cn_embeddings:
            for cn_key,i in self.cn_dict.items():
                cn_emb = '_cn{}_'.format(i)
                if cn_emb not in self.s2i:
                    self.s2i[cn_emb] = len(self.s2i)
                    print('{} tag added to the dataset with index {}'.format(cn_emb,
                                                                         len(self.s2i)-1))                
                

        skf = StratifiedKFold(n_splits = 4,
                              shuffle = True,
                              random_state = self.random_state)

        f1, f2, f3, f4 = skf.split(list(self.train_names),
                                   list(self.train_countries))
 
        folds = [f1, f2, f3, f4]
        self.train_idx = folds[self.fold][0]
        self.val_idx = folds[self.fold][1] 

    def __getitem__(self, idx):
        if self.mode == 'train':
            _idx = self.train_idx[idx]
        elif self.mode == 'val':
            _idx = self.val_idx[idx]

        if self.mode in ['train','val']:
            name,gt_name,target,country = (self.train_names[_idx],
                                           self.train_gt_names[_idx],
                                           self.train_targets[_idx],
                                           self.train_countries[_idx])
            
            name_processed,name_mmb = self.process_name(name,country)
            gt_name_processed,gt_name_mmb = self.process_name(gt_name,country)  
            
            return_tuple = [name,
                            gt_name,
                            name_processed.long(),
                            gt_name_processed.long(),
                            gt_name_mmb.float(),
                            target]
            
        elif self.mode == 'test':
            # idx, not _idx
            name,country = (self.train_names[idx],
                            self.test_countries[idx])
            
            name_processed,name_mmb = self.process_name(name,country)
            
            return_tuple = [name,
                            name_processed.long()]            
        
        return return_tuple
    
    def process_name(self,
                     name,
                     country):
        # start with the _start_ token
        processed = [self.s2i['_start_']] * 1              

        processed.extend([self.s2i[s] 
                          if s in self.s2i
                          else self.s2i['_unk_']
                          for s in name])
        
        # add classification token for the transformer model after the end of the name
        processed.extend([self.s2i['_classify_']] * 1)         
        
        l = len(processed)
        
        if len(processed) < self.max_len:
            processed.extend([self.s2i['_pad_']] 
                             * (self.max_len - len(processed)))
        
        processed = torch.tensor(processed).squeeze()
        
        # also embed positional information and masks
        # also account for the classification token
        mmb = np.zeros((self.max_len))
        mmb[:l] = 1
        mmb = torch.from_numpy(mmb)
        
        # position information that is added to the input embeddings in the TransformerModel
        # also account for the classification token
        
        pmb = np.arange(self.vocab_len, self.vocab_len + self.max_len)
        
        if self.add_cn_embeddings:
            # add country embeddings
            # just repeat the country embedding
            # having different country embedding for each position is questionable
            
            cmb = np.asarray([self.vocab_len + self.max_len + country] * self.max_len)
            processed = torch.stack((processed,
                                     torch.from_numpy(pmb),
                                     torch.from_numpy(cmb)),dim=1)    
        else:
            processed = torch.stack((processed,torch.from_numpy(pmb)),dim=1)
        
        return processed,mmb
        
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_idx)
        elif self.mode == 'val':
            return len(self.val_idx)
        elif self.mode == 'test':
            return len(self.test_names)  