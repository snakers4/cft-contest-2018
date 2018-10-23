import os
import re
import math
import copy
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data, datasets
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """
    def __init__(self, src, trg,
                 id,clf,cn,
                 pad_index=0):
        
        src, src_lengths = src
        
        self.id = id
        self.clf = clf
        self.cn = cn
        
        self.src = src
        self.src_lengths = src_lengths
        self.src_mask = (src != pad_index).unsqueeze(-2)
        self.nseqs = src.size(0)
        
        self.trg = None
        self.trg_y = None
        self.trg_mask = None
        self.trg_lengths = None
        self.ntokens = None

        if trg is not None:
            trg, trg_lengths = trg
            self.trg = trg[:, :-1]
            self.trg_lengths = trg_lengths
            self.trg_y = trg[:, 1:]
            self.trg_mask = (self.trg_y != pad_index)
            self.ntokens = (self.trg_y != pad_index).data.sum().item()
        
        if os.environ.get("USE_CUDA") == 'True':
            self.src = self.src.cuda()
            self.src_mask = self.src_mask.cuda()

            if trg is not None:
                self.trg = self.trg.cuda()
                self.trg_y = self.trg_y.cuda()
                self.trg_mask = self.trg_mask.cuda()

class SimpleLossCompute:
    """A simple loss compute and train function."""

    def __init__(self, generator,
                 criterion,clf_criterion,
                 opt=None,
                 clf_coeff = 1):
        self.generator = generator
        self.criterion = criterion
        self.clf_criterion = clf_criterion
        self.opt = opt
        self.clf_coeff = clf_coeff

    def __call__(self, x, y, norm,
                 clf_logits, clf_gts):
        x = self.generator(x)
        
        # normalize the lm loss by number of tokens
        lm_loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                                 y.contiguous().view(-1))
        lm_loss = lm_loss / norm
        
        # normalize the clf loss by number of sentences
        clf_loss = self.clf_criterion(clf_logits, clf_gts)
        clf_loss = self.clf_coeff * clf_loss
        
        loss = lm_loss + clf_loss
        
        if self.opt is not None:
            loss.backward()          
            self.opt.step()
            self.opt.zero_grad()

        # return only one loss
        return lm_loss.data.item() * norm, clf_loss.data.item()
    
def cleaner(text):
    new_text = text
    new_text = re.sub("[_\n/]", " ", new_text)
    new_text = re.sub("[12345?[|]", "", new_text)
    new_text = re.sub("Є", "Е", new_text)
    return new_text

def tokenize(text):
    return list(cleaner(text))

def greedy_decode_batch(model,
                        src, src_mask, src_lengths,
                        max_len=100,
                        sos_index=1, eos_index=None,
                        return_logits=False,
                        cn=None):
    """Greedily decode a sentence."""
    batch_size = src.size(0)
    
    with torch.no_grad():
        encoder_hidden, encoder_final = model.encode(src, src_mask, src_lengths, cn)
        clf_logits = model.classifier(encoder_hidden)
        if return_logits:
            pred_classes = clf_logits
        else:
            _, pred_classes = torch.max(clf_logits, dim=1)
        pred_classes = pred_classes.data.cpu().numpy()
        prev_y = torch.ones(batch_size, 1).fill_(sos_index).type_as(src)
        trg_mask = torch.ones_like(prev_y)

    output = []
    # attention_scores = []
    hidden = None

    for i in range(max_len):
        with torch.no_grad():
            out, hidden, pre_output = model.decode(
              encoder_hidden, encoder_final, src_mask,
              prev_y, trg_mask, hidden,
              cn=cn)

            # we predict from the pre-output layer, which is
            # a combination of Decoder state, prev emb, and context
            prob = model.generator(pre_output[:, -1])

        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data
        output.append(next_word.cpu().numpy())
        prev_y = next_word.unsqueeze(dim=1)
        # attention_scores.append(model.decoder.attention.alphas.cpu().numpy())
    
    output = np.array(output)
        
    # cut off everything starting from </s> 
    # (only when eos_index provided)
    # if eos_index is not None:
    #    first_eos = np.where(output==eos_index)[0]
    #    if len(first_eos) > 0:
    #        output = output[:first_eos[0]]      
    
    output = np.stack(output).T
    
    return output,pred_classes 
    #, np.concatenate(attention_scores, axis=1)

def lookup_words(x, vocab=None):
    if vocab is not None:
        x = [vocab.itos[i] for i in x]

    return [str(t) for t in x]

def run_epoch(data_iter,
              model,
              loss_compute,
              print_every=50,
              num_batches=100,
              epoch_no=0):
    """Standard Training and Logging Function"""

    start = time.time()
    total_tokens = 0
    total_loss = 0
    total_clf_loss = 0
    print_tokens = 0
    total_sentences = 0
    
    lm_losses = AverageMeter()
    clf_losses = AverageMeter()    
    
    with tqdm(total=num_batches) as pbar:
        for i, batch in enumerate(data_iter, 1):
            pbar.set_description('EPOCH %i' % epoch_no)
            batch_size = batch.src.size(0)
            
            (out, _, pre_output),clf_logits = model.forward(batch.src, batch.trg,
                                                            batch.src_mask, batch.trg_mask,
                                                            batch.src_lengths, batch.trg_lengths,
                                                            batch.cn)
            loss, clf_loss = loss_compute(pre_output, batch.trg_y, batch.nseqs,
                                          clf_logits,batch.clf)
            
            total_loss += loss
            total_clf_loss += clf_loss
            total_tokens += batch.ntokens
            print_tokens += batch.ntokens
            total_sentences += batch_size
            
            lm_losses.update(loss / batch.nseqs, batch_size)
            clf_losses.update(clf_loss, batch_size)

            if model.training and i % print_every == 0:
                elapsed = time.time() - start
                
                pbar.set_postfix(loss=(lm_losses.avg,lm_losses.val),
                                 clf_loss=(clf_losses.avg,clf_losses.val),
                                 tkns_per_sec=print_tokens / elapsed)
                
                """
                print("Epoch Step: %d Loss: %f CLF Loss: %f Tokens per Sec: %f" %
                        (i,
                         loss / batch.nseqs, clf_loss,
                         print_tokens / elapsed))
                """
                start = time.time()
                print_tokens = 0
            
            pbar.update(1)
                
    return math.exp(total_loss / float(total_tokens)), total_clf_loss / float(total_sentences)

def print_examples(example_iter, model, n=2, max_len=100, 
                   sos_index=1, 
                   src_eos_index=None, 
                   trg_eos_index=None, 
                   src_vocab=None, trg_vocab=None):
    """Prints N examples. Assumes batch size of 1."""

    model.eval()
    count = 0
    print()
    
    if src_vocab is not None and trg_vocab is not None:
        src_eos_index = src_vocab.stoi[EOS_TOKEN]
        trg_sos_index = trg_vocab.stoi[SOS_TOKEN]
        trg_eos_index = trg_vocab.stoi[EOS_TOKEN]
    else:
        src_eos_index = None
        trg_sos_index = 1
        trg_eos_index = None
        
    for i, batch in enumerate(example_iter):
      
        src = batch.src.cpu().numpy()[0, :]
        trg = batch.trg_y.cpu().numpy()[0, :]

        # remove </s> (if it is there)
        src = src[:-1] if src[-1] == src_eos_index else src
        trg = trg[:-1] if trg[-1] == trg_eos_index else trg      
      
        result, _ = greedy_decode(
          model, batch.src, batch.src_mask, batch.src_lengths,
          max_len=max_len, sos_index=trg_sos_index, eos_index=trg_eos_index)
        print("Example #%d" % (i+1))
        print("Src : ", " ".join(lookup_words(src, vocab=src_vocab)))
        print("Trg : ", " ".join(lookup_words(trg, vocab=trg_vocab)))
        print("Pred: ", " ".join(lookup_words(result, vocab=trg_vocab)))
        print()
        
        count += 1
        if count == n:
            break

def rebatch(pad_idx, batch):
    """Wrap torchtext batch into our own Batch class for pre-processing"""
    return Batch(batch.src, batch.trg,
                 batch.id,batch.clf,batch.cn,
                 pad_idx)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count