import sys, os, gc, re, datetime, unidecode, pickle, tqdm, math, random, itertools, collections
import difflib
from IPython.lib.display import FileLink

from joblib import Parallel, delayed
import multiprocessing
cpu_cores = multiprocessing.cpu_count()

from processing_utils import *
import nltk
import scipy
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow as tf
import keras
from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras import optimizers
from keras import backend as K
from keras.utils import Sequence, to_categorical
from keras.preprocessing.sequence import pad_sequences

cfg = K.tf.ConfigProto()
cfg.gpu_options.allow_growth = True
K.set_session(K.tf.Session(config=cfg))

data_dir = 'data/'
models_dir = 'models/'
results_dir = 'results/'

def predict(model, seq, convert2text=ohe_seq2chars):
    spell_pred = []
    names_pred = []
    
    for i in tqdm.tqdm(range(0, len(seq))):
        batch_spell_pred, batch_names_pred = model.predict_on_batch(seq.__getitem__(i))

        spell_pred.append(batch_spell_pred)
        names_pred += [convert2text(pred) for pred in batch_names_pred]
    
    spell_pred = np.concatenate(spell_pred)
    names_pred = np.stack(names_pred)
    return spell_pred, names_pred

def seq2seq_predict(model, seq, convert2text=ohe_seq2chars):
    spell_pred = []
    names_pred = []
    
    for i in tqdm.tqdm(range(0, len(seq))):
        batch_spell_pred, batch_names_pred = model.predict_on_batch(seq.__getitem__(i))

        spell_pred.append(batch_spell_pred)
        names_pred += [convert2text(pred) for pred in batch_names_pred]
    
    spell_pred = np.concatenate(spell_pred)
    names_pred = np.stack(names_pred)
    return spell_pred, names_pred

def seq2seq_pred_sample(model, text, enc_convert2seq, enc_vocab_size, convert2text):
    input_seq = to_categorical(pad_sequences([enc_convert2seq(text)], maxlen=len(text), padding="pre", truncating="post"),
                               enc_vocab_size)
    spell_pred, names_pred = model.predict_on_batch(input_seq)
    spell_pred = np.argmax(spell_pred[0])
    names_pred = convert2text(names_pred[0])

    return spell_pred, names_pred

def seq2seq_enc_dec_predict(encoder_model, decoder_model, seq, convert2text,
                            str_inx, vocab_size, max_len):
    spell_pred = []
    names_pred = []
    
    for i in tqdm.tqdm(range(0, len(seq))):
        batch_spell_pred, batch_state_h_value, batch_state_c_value = encoder_model.predict_on_batch(seq.__getitem__(i))
        spell_pred.append(batch_spell_pred)
            
        inp_toks = np.zeros((batch_spell_pred.shape[0], 1, vocab_size))
        inp_toks[:, 0, str_inx] = 1.
        batch_names_pred = []
        states_values = [batch_state_h_value, batch_state_c_value]
        for _ in range(max_len):
            toks_pred, dec_state_h, dec_state_c = decoder_model.predict_on_batch([inp_toks] + states_values)
            batch_names_pred.append(toks_pred)

            inp_toks = np.zeros((toks_pred.shape[0], 1, vocab_size))
            inp_toks[:, 0][toks_pred[:, 0] >= np.max(toks_pred, axis=-1)] = 1.
            states_values = [dec_state_h, dec_state_c]
        
        names_pred.append([convert2text(pred) for pred in np.concatenate(batch_names_pred, axis=1)])
    
    spell_pred = np.concatenate(spell_pred)
    names_pred = np.concatenate(names_pred)
    return spell_pred, names_pred

class Evaluation(Callback):
    def __init__(self, seq, y, names_true, name, convert2text=ohe_seq2chars, interval=1):
        super(Callback, self).__init__()

        self.seq = seq
        self.y = y
        self.names_true = names_true
        self.convert2text = convert2text
        self.name = name
        self.interval = interval

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            spell_prob, names_pred = predict(self.model, self.seq, self.convert2text)
            spell_pred = np.argmax(spell_prob, axis=1)
            
            task1_score = score_task1(self.y, spell_pred)
            task2_score = score_task2(self.names_true, names_pred, self.y, spell_pred)
            score = (task1_score+task2_score)/2
            
            logs[self.name+"_task1_score"] = task1_score
            logs[self.name+"_task2_score"] = task2_score
            logs[self.name+"_score"] = score
            print((self.name+"_task1_score: {:.8f}; "+self.name+"_task2_score: {:.8f};").format(task1_score, task2_score))
            print((self.name+"_score: {:.8f}").format(score))
            
class SimpleGenerator(Sequence):
    
    def __init__(self, names, names_true, y, vocab_size, convert2seq,
                 ohe_input=True, ohe_output=True, max_len=None, conv_extra=2,
                 batch_size=256, shuffle=False):
        
        self.names = names
        self.names_true = names_true
        self.y = y
        self.vocab_size = vocab_size
        self.convert2seq = convert2seq
        
        self.ohe_input = ohe_input
        self.ohe_output = ohe_output
        self.batch_size = batch_size
        self.max_len = max_len
        self.conv_extra = conv_extra
        
        self.inx = np.arange(0, len(self.y))
        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.inx)

    def __len__(self):
        return math.ceil(self.inx.shape[0] / self.batch_size)

    def __getitem__(self, i):
        batch_inx = self.inx[i*self.batch_size:(i+1)*self.batch_size]
        
        names_batch = self.names[batch_inx]
        names_true_batch = self.names_true[batch_inx]
        y_batch = to_categorical(self.y[batch_inx], 3)
        
        names_seqs_batch = [self.convert2seq(name) for name in names_batch]
        if self.conv_extra != 0:
            names_true_seqs_batch = [self.convert2seq(name)[self.conv_extra:-self.conv_extra] for name in names_true_batch]
        else:
            names_true_seqs_batch = [self.convert2seq(name) for name in names_true_batch]
        
        max_len = max([len(seq) for seq in names_seqs_batch+names_true_seqs_batch])
        if self.max_len != None:
            max_len = min([self.max_len, max_len])
        
        x_inp = pad_sequences(names_seqs_batch, maxlen=max_len+(self.conv_extra*2), padding="post", truncating="post")
        x_out = pad_sequences(names_true_seqs_batch, maxlen=max_len, padding="post", truncating="post")
        
        if self.ohe_input:
            x_inp = to_categorical(x_inp, self.vocab_size)
        if self.ohe_output:
            x_out = to_categorical(x_out, self.vocab_size)
        else:
            x_out = np.expand_dims(x_out, -1)
        
        return x_inp, [y_batch, x_out]
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.inx)
            
class SimplePredGenerator(Sequence):
    
    def __init__(self, names, convert2seq, vocab_size=None, conv_extra=2,
                 ohe_input=True, max_len=None, batch_size=256):
        
        self.names = names
        self.convert2seq = convert2seq
        self.vocab_size = vocab_size
        self.conv_extra = conv_extra
        
        self.ohe_input = ohe_input
        self.batch_size = batch_size
        self.max_len = max_len
        
        self.inx = np.arange(0, len(self.names))

    def __len__(self):
        return math.ceil(self.inx.shape[0] / self.batch_size)

    def __getitem__(self, i):
        batch_inx = self.inx[i*self.batch_size:(i+1)*self.batch_size]
        
        names_batch = self.names[batch_inx]
        names_seqs_batch = [self.convert2seq(name) for name in names_batch]
            
        max_len = max([len(seq) for seq in names_seqs_batch])
        if self.max_len != None:
            max_len = min([self.max_len, max_len])
        
        x_inp = pad_sequences(names_seqs_batch, maxlen=max_len+(self.conv_extra*2), padding="post", truncating="post")
        if self.ohe_input:
            x_inp = to_categorical(x_inp, self.vocab_size)
        
        return x_inp
            
def score_task1(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred, average='macro')

def score_task2(names_true, names_pred, y_true, y_pred):
    names_pred_copy = names_pred.copy()
    names_pred_copy[y_pred != 1] = ''
    return np.mean(names_true[y_true == 1] == names_pred_copy[y_true == 1])

def evaluate(y_true, names_true, y_pred, names_pred):
    task1_score = score_task1(y_true, y_pred)
    task2_score = score_task2(names_true, names_pred, y_true, y_pred)
    score = (task1_score+task2_score)/2

    print("val_task1_score: {:.5f}".format(task1_score))
    print("val_task2_score: {:.5f}".format(task2_score))
    print("val_final_score: {:.5f}".format(score))
    
def convert2float(string):
    floats = string.replace('[','').replace(']','').strip().replace('  ',' ').split()
    floats = np.array([float(f) for f in floats])

    return floats

def convert2prob(x):
    return logit2prob(convert2float(x))

def logit2prob(x):
    odds = np.exp(x)
    prob = odds/(1+odds)
    return prob

def load_l4_df(filename, select_cols=["id", "prob", "pred", "fullname_pred"]):
    print("Loading " + filename + " ...")
    l4_df = pd.read_csv(results_dir+filename)\
        .rename(columns={"fullname_true": "fullname_pred", "target": "prob"})\
        .sort_values("id")\
        .reset_index(drop=True)
     
    l4_df["prob"] = l4_df["prob"].apply(convert2prob)
    l4_df["pred"] = l4_df["prob"].apply(np.argmax)

    return l4_df[select_cols]

def pckl(obj, path):
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

    with multiprocessing.Pool(workers) as p:
        apply_lst = [([params], func, i, kwargs) for i,params in enumerate(param_lst)]
        result = list(tqdm.tqdm(p.imap(_apply_lst, apply_lst), total=len(apply_lst)))

    # lists do not need such sorting, but this can be useful later
    result=sorted(result,key=lambda x:x[0])
    return [_[1] for _ in result]

def _apply_lst(args):
    params, func, num, kwargs = args
    return num, func(*params,**kwargs)

class AttentionWeightedAverage(Layer):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(** kwargs)
        
    def get_config(self):
        config = {
            'return_attention': self.return_attention,
        }
        base_config = super(AttentionWeightedAverage, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_W'.format(self.name),
                                 initializer=self.init)
        self.trainable_weights = [self.W]
        super(AttentionWeightedAverage, self).build(input_shape)

    def call(self, x, mask=None):
        # computes a probability distribution over the timesteps
        # uses 'max trick' for numerical stability
        # reshape is done to avoid issue with Tensorflow
        # and 1-dimensional weights
        logits = K.dot(x, self.W)
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ai = ai * mask
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None
        
class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.
    
    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```    
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1
        K.set_value(self.model.optimizer.lr, self.clr())

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)