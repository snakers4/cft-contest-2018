import re
import pickle
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm

def generate_mistakes(name, nearest_dict):
    used = []
    num_words = len(name.split(' '))
    name = list(name)
    name_copy = name[:]
    num_mists = np.random.choice([1,2,3], p=[0.93, 0.06, 0.01])
    num = 0
    while num < num_mists:
    
        mist_type = np.random.choice(['insert', 'change', 'swap', 'delete_space', 'delete_letter'], p=[0.5, 0.3, 0.05, 0.075, 0.075])

    
        if (mist_type == 'delete_space') & (num_words > 1):
            while True:
                idx = np.random.randint(len(name))
                if (name[idx] == ' ') & (idx not in used):
                    break
            num += 1
            num_words -= 1
            used.append(idx)
            name_copy[idx] = ''
            continue
        
        while True:
            idx = np.random.randint(len(name))
            
            if (name[idx] != ' ') & (idx not in used):
                    break
        num += 1
        used.append(idx)
        
        if mist_type == 'insert':
            new_letter = pd.DataFrame(nearest_dict[name[idx]])
            new_letter = new_letter.sample(1, weights=new_letter[1]).values[0][0]
            if np.random.rand() < 0.5:
                name_copy[idx] = new_letter + name_copy[idx]
            else:
                name_copy[idx] = name_copy[idx] + new_letter
            
            
        if mist_type == 'change':
            new_letter = pd.DataFrame(nearest_dict[name[idx]])
            new_letter = new_letter.sample(1, weights=new_letter[1]).values[0][0]
            name_copy[idx] = new_letter
        
        if mist_type == 'swap':
            if (idx == 0) :
                name_copy[idx], name_copy[idx+1] = name_copy[idx+1], name_copy[idx]
            if idx == (len(name_copy) -1):
                name_copy[idx], name_copy[idx-1] = name_copy[idx-1], name_copy[idx]
            else:
                if name[idx-1] == ' ':
                    name_copy[idx], name_copy[idx+1] = name_copy[idx+1], name_copy[idx]
                if name[idx+1] == ' ':
                    name_copy[idx], name_copy[idx-1] = name_copy[idx-1], name_copy[idx]
                else:
                    if np.random.rand() < 0.5:
                        name_copy[idx], name_copy[idx+1] = name_copy[idx+1], name_copy[idx]
                    else:
                        name_copy[idx], name_copy[idx-1] = name_copy[idx-1], name_copy[idx]
                    
        if mist_type == 'delete_letter':
            name_copy[idx] = ''
        
    return ''.join(name_copy)