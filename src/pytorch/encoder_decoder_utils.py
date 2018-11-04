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

import sentencepiece as spm

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
                 clf_coeff = 1,
                 seq_penalize_only_errors=False):
        self.generator = generator
        self.criterion = criterion
        self.clf_criterion = clf_criterion
        self.opt = opt
        self.clf_coeff = clf_coeff
        self.seq_penalize_only_errors = seq_penalize_only_errors

    def __call__(self, x, y, norm,
                 clf_logits, clf_gts):
        x = self.generator(x)

        # normalize the lm loss by number of tokens
        if self.seq_penalize_only_errors:
            # TODO add code here
            pass
        else:
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
    new_text = re.sub("[_\n/12345?[|,]", " ", new_text)
    new_text = re.sub("Є", "Е", new_text)
    return new_text

def tokenize(text):
    return list(text)

class Tokenizer(object):
    def __init__(self,
                 model_path,
                 models_dir='bpe_models/'):
        sp = spm.SentencePieceProcessor()
        sp.Load(os.path.join(models_dir,model_path))
        self.sp = sp
        
    def tokenize(self,text):
        return self.sp.EncodeAsPieces(text)
    
    def enc_text2seq(self,text):
        return self.sp.EncodeAsIds(text)

    def enc_text2pieces(self,text):
        return self.sp.EncodeAsPieces(text)

    def dec_pieces2text(self,piece_list):
        return self.sp.DecodePieces(piece_list)

def greedy_decode_batch(model,
                        src, src_mask, src_lengths,
                        max_len=100,
                        sos_index=1, eos_index=None,
                        return_logits=False,
                        cn=None,add_skip=False):
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
            if add_skip:
                out, hidden, pre_output = model.decode(
                  encoder_hidden, encoder_final, src_mask,
                  prev_y, trg_mask, hidden,
                  cn=cn, skip=src)                
            else:
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

def get_beam_probs(encoder_hidden, encoder_final, src_mask,
                   prev_y, trg_mask, hidden,
                   cn=None, model=None,
                   add_skip=False, src=None):
    
    with torch.no_grad():
        if add_skip:
            out, hidden, pre_output = model.decode(
                encoder_hidden, encoder_final, src_mask,
                prev_y, trg_mask, hidden,
                cn=cn, skip=src)            
        else:
            out, hidden, pre_output = model.decode(
                encoder_hidden, encoder_final, src_mask,
                prev_y, trg_mask, hidden,
                cn=cn)

        # we predict from the pre-output layer, which is
        # a combination of Decoder state, prev emb, and context
        prob = model.generator(pre_output[:, -1])
        prob = F.softmax(prob, dim=1)
        
        """
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data
        prev_y = next_word.unsqueeze(dim=1)        
        
        print(prev_y.shape)
        """
    return prob,hidden
        
def beam_decode_batch(model,
                      src, src_mask, src_lengths,
                      max_len=100,
                      sos_index=1, eos_index=None,
                      return_logits=False,
                      cn=None,
                      beam_width=3, device=None, values_to_return=1,
                      debug=False, add_skip=False):
    """Use beam search to decode a sentence."""
    batch_size = src.size(0)
    if debug:
        end = time.time()
    
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

    if debug:
        print('Encoder - {}'.format(time.time() - end))
        end = time.time()

    # output = []
    # attention_scores = []
    # hidden = None

    # init tensors used for batch-based beam search
    hidden_tensor = None # (num_layers * num_directions, batch, hidden_size, sequences)
    score_tensor  = None # (batch,num_sequences)
    seq_tensor    = None # (batch,num_sequences,max_len)
    
    prev_y_eos = torch.ones(batch_size).fill_(eos_index).long().to(device)
    break_mask = torch.zeros(batch_size).byte().to(device)
    
    for i in range(max_len):
        if debug:
            print('i = {}'.format(i))
        # for first iteration asume there is already one sequence
        if seq_tensor is None:
            seq_len = 1
        else:
            seq_len = seq_tensor.size(1)
        
        # sequences is dynamically updated
        # iterate over sequences
        # for each sequence do the magic
        
        for j in range(seq_len):
            if debug:
                print('j = {}'.format(j))
            if hidden_tensor is None:
                # first hidden is initialized as None
                hidden = None
            else:
                # we store only the current hidden state
                hidden = hidden_tensor[:,:,:,j].contiguous() #.to(device)
                if debug:
                    print(hidden_tensor.shape)
                # do not forget to pass the last symbol
                prev_y = seq_tensor[:,j,i-1:i].contiguous() #.to(device)
                
                if i==0:
                    prev_y = prev_y.unsqueeze(dim=1)
                

            
            # use wrapper for readability
            prob, hidden = get_beam_probs(encoder_hidden, encoder_final, src_mask,
                                          prev_y, trg_mask, hidden,
                                          cn=cn, model=model, add_skip=add_skip, src=src)
           
            # these tensors are (batch,beam_width)
            scores, indices = torch.topk(input=prob,
                                         k=beam_width,
                                         dim=1)
            if debug:
                print(scores[0])
                print("".join(lookup_words(list(indices[0]),
                                           vocab=TRG_NAMES.vocab)))

            
            # scores = -torch.log(scores.detach().cpu())
            scores = -torch.log(scores)
            
            assert scores.size(0) == batch_size
            assert indices.size(0) == batch_size
            assert scores.size(1) == beam_width
            assert indices.size(1) == beam_width            
            
            # collect hidden states
            # (batch,num_sequences,hidden_size)
            if hidden_tensor is not None:
                # hidden sizes are shared across sequences
                # we store only the last ones on each step
                """
                hidden_tensor = torch.cat([hidden_tensor]
                                          +beam_width*[hidden.detach().cpu().unsqueeze(dim=3)],
                                          dim=3)
                """
                hidden_tensor = torch.cat([hidden_tensor]
                                          +beam_width*[hidden.unsqueeze(dim=3)],
                                          dim=3).contiguous()                
                
            else:
                """
                hidden_tensor = torch.cat(beam_width*[hidden.detach().cpu().unsqueeze(dim=3)],
                                          dim=3)                
                """
                hidden_tensor = torch.cat(beam_width*[hidden.unsqueeze(dim=3)],
                                          dim=3).contiguous() 
            
            # collect indexes
            # (batch,beam_width) => (batch,num_sequences,max_len)
            if seq_tensor is not None:
                # pad sequence tensor with -1
                if seq_tensor.size(2) == i:
                    # we assume that at the beginning of each operation
                    # sequences tensor has only beam_width sequences
                    seq_tensor = torch.cat([seq_tensor,
                                            torch.ones(batch_size,
                                                       beam_width,
                                                       1,dtype=torch.long).to(device)*(-1)],
                                           dim=2)
                
                # take old indices
                old_seq_indices = seq_tensor[:,j:j+1,:].clone()
                # repeat beam_width times to create several new sequences
                new_indices = torch.cat(beam_width*[old_seq_indices],
                                        dim=1)
                # add new indices  
                new_indices[:,:,i] = indices # .detach().cpu()
                # merge with seq tensor to create new sequences
                seq_tensor = torch.cat([seq_tensor,new_indices],
                                       dim=1)
            else:
                # convert first beam into sequences
                seq_tensor = indices.unsqueeze(dim=2) # .detach().cpu().unsqueeze(dim=2)
                assert seq_tensor.size() == (batch_size,beam_width,1)
            
            # update scores
            # (batch,beam_width) => (batch,num_sequences)
            if score_tensor is not None:
                # multiply parent sequence's score by its children
                parent_score = score_tensor[:,j:j+1].clone()
                parent_score = torch.cat(beam_width*[parent_score],
                                         dim=1)
                children_scores = parent_score + scores
                score_tensor = torch.cat([score_tensor,
                                          children_scores],
                                         dim=1)
            else:
                score_tensor = scores
                
        # print(seq_tensor.size())

        # remove the first beam_width sequences for simplicity
        # but only after the first pass
        if i>0:
            hidden_tensor = hidden_tensor[:,:,:,beam_width:]
            score_tensor = score_tensor[:,beam_width:] 
            seq_tensor = seq_tensor[:,beam_width:,:] 

        # select the best sequences to survive!
        _, seq_indices = torch.topk(input=score_tensor,
                                    k=beam_width,
                                    dim=1,
                                    largest=False)

        # print(score_tensor[0])
        
        # batch and sequence dimension are equal
        assert hidden_tensor.size(1) == score_tensor.size(0)
        assert hidden_tensor.size(3) == score_tensor.size(1)
        assert score_tensor.size()[0:2] == seq_tensor.size()[0:2]

        # print(hidden_tensor.shape)
        # print(seq_tensor.shape)
        # print(score_tensor.shape)
        # print(seq_indices.shape)           
        
        # torch gather is not applicable due to different shape
        # for simplicity re-index manually
        hidden_tensor = torch.cat([hidden_tensor[:,
                                                 _:_+1,
                                                 :,
                                                 seq_indices[_]] 
                                   for _ in range(batch_size)],
                                  dim=1)
        
        score_tensor = torch.gather(input=score_tensor,
                                    dim=1,
                                    index=seq_indices)
        
        seq_tensor = torch.cat([seq_tensor[_:_+1,
                                           seq_indices[_],
                                           :]
                                for _ in range(batch_size)],
                               dim=0).long()
        
        # break out of the cycle if all the last predictions were eos_index
        prev_y = seq_tensor[:,j,i-1:i].long()
        if i==0:
            # to pass to the model on the next step
            prev_y = prev_y.unsqueeze(dim=1)
            
        mask = (prev_y.squeeze(dim=1) == prev_y_eos)
        break_mask += mask
        
        if (break_mask>=1).sum()==512:
            if debug:
                print('Breaking out of cycle early')
            break
        
        if debug:
            print('Iteration {} - {}'.format(i,time.time() - end))
            end = time.time()
      
    # output = np.array(output)
    # output = np.stack(output).T
    
    # select only the best of the best
    _, seq_indices = torch.topk(input=score_tensor,
                                k=values_to_return,
                                dim=1,
                                largest=False)
    
    seq_tensor = torch.cat([seq_tensor[_:_+1,
                                       seq_indices[_],
                                       :]
                            for _ in range(batch_size)],
                           dim=0).long()
    
    # shed extra dimension for compatibility
    if values_to_return==1:
        assert seq_tensor.size(1)==1
        seq_tensor = seq_tensor.squeeze(dim=1)
    
    seq_tensor = seq_tensor.cpu().detach().numpy()
    
    if debug:
        print('Postprocessing - {}'.format(time.time() - end))
        end = time.time()
    
    return seq_tensor,pred_classes 