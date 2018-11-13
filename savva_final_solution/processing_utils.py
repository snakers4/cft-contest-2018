import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

seed = 7961730
print("Seed: {}".format(seed))

PAD_TOK = ""
STR_TOK = "\t"
END_TOK = "\n"
CHARS_SET = {' ', "'", '-', '.', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 
             'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '`', 'Ё', 'І', 'Ї', 'А', 'Б',
             'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У',
             'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я'}

INX2CHAR = [PAD_TOK, STR_TOK, END_TOK] + sorted(list(CHARS_SET))
CHAR2INX = dict((c, i) for i, c in enumerate(INX2CHAR))
VOCAB_SIZE = len(CHAR2INX)
print("Vocab size: {}".format(VOCAB_SIZE))

def cleaner(text):
    new_text = text
    new_text = re.sub("[_\n/12345?[|,]", " ", new_text)
    new_text = re.sub("Є", "Е", new_text)
    return new_text

def print_len_stats(len_arr):
    len_mean = len_arr.mean(axis=0)
    len_std = len_arr.std(axis=0)

    print("len_min: {}".format(len_arr.min()))
    print("len_max: {}".format(len_arr.max()))
    print("len_mean: {}".format(len_mean))
    print("len_std: {}".format(len_std))
    print("len_mean + 2*len_std: {}".format(len_mean+2*len_std))
    print("len_mean + 3*len_std: {}".format(len_mean+3*len_std))
    print()

def chars2seq(text):
    return [CHAR2INX[STR_TOK]] + [CHAR2INX[c] for c in list(text)] + [CHAR2INX[END_TOK]]
    
def ohe_seq2chars(ohe_seq):
    return ("".join([INX2CHAR[np.argmax(ohe)] for ohe in ohe_seq])).split("\n")[0].strip()

def seq2chars(seq):
    return ("".join([INX2CHAR[inx] for inx in seq])).split("\n")[0].strip()