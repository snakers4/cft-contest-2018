import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import math, copy, time
import torch.nn.functional as F
from pytorch.ann_transformer import subsequent_mask

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if len(mask) > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        # return self.criterion(x, Variable(true_dist, requires_grad=False))
        return self.criterion(x, true_dist)
    
class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0,
                 id=None,clf=None):
        self.id = id
        self.clf = clf
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum().float()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        # tgt_mask = tgt_mask & Variable(
        #    subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask

def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    # src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    src, trg = batch.src, batch.trg
    return Batch(src, trg, pad_idx,
                 batch.id,batch.clf)

def greedy_decode(model, src, src_mask, max_len, start_symbol,
                  return_logits=False, end_symbol=None, device=None):
    
    batch_size = src.size(0)
    
    # early stopping 
    break_mask = torch.zeros(batch_size).byte().to(device)
    prev_y_eos = torch.ones(batch_size).fill_(end_symbol).long().to(device)
    
    memory = model.encode(src, src_mask)
    clf_logits = model.classifier(memory)
    
    if return_logits:
        pred_classes = clf_logits
    else:
        _, pred_classes = torch.max(clf_logits, dim=1)    
    pred_classes = pred_classes.data.cpu().numpy()
    
    ys = torch.ones(batch_size, 1).fill_(start_symbol).type_as(src.data)
    
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           ys, 
                           subsequent_mask(ys.size(1)).type_as(src.data))
        # prob = model.generator(out[:, -1])
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)

        ys = torch.cat([ys, 
                        next_word.unsqueeze(dim=1)], dim=1)        
        
        mask = (next_word == prev_y_eos)
        break_mask += mask        
        
        if (break_mask>=1).sum()==batch_size:
            # print('Breaking out of cycle early')
            break            
    
    ys = ys.data.cpu().numpy()
    
    return ys,pred_classes

def run_epoch(data_iter, model, loss_compute,
              print_every=50,num_batches=100,epoch_no=0):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    
    total_clf_loss = 0
    total_sentences = 0
    
    lm_losses = AverageMeter()
    clf_losses = AverageMeter()     
    
    with tqdm(total=num_batches) as pbar:
        for i, batch in enumerate(data_iter):
            pbar.set_description('EPOCH %i' % epoch_no)
            
            batch_size = batch.src.size(0)
            
            out,clf_logits = model.forward(batch.src, batch.trg, 
                                           batch.src_mask, batch.trg_mask)
            loss,clf_loss = loss_compute(out, batch.trg_y, batch.ntokens,
                                         clf_logits,batch.clf)
            total_loss += loss
            total_clf_loss += clf_loss
            total_tokens += batch.ntokens
            tokens += batch.ntokens
            total_sentences += batch_size
            
            lm_losses.update(loss.cpu() / float(batch.ntokens.cpu()), batch_size)
            clf_losses.update(clf_loss, batch_size)            
            
            if model.training and i % print_every == 0:
                elapsed = time.time() - start

                # print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                #        (i, loss / batch.ntokens, tokens / elapsed))

                pbar.set_postfix(loss=(float(lm_losses.avg),float(lm_losses.val)),
                                 clf_loss=(clf_losses.avg,clf_losses.val))            

                start = time.time()
                tokens = 0
            pbar.update(1)
    return total_loss / total_tokens,  total_clf_loss / float(total_sentences)

class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator,
                 criterion,clf_criterion,
                 opt=None,clf_coeff = 1):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        self.clf_criterion = clf_criterion
        self.clf_coeff = clf_coeff
        
    def __call__(self, x, y, norm,
                 clf_logits, clf_gts):
        x = self.generator(x)
        
        lm_loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                                 y.contiguous().view(-1)) / norm
        
        # normalize the clf loss by number of sentences
        clf_loss = self.clf_criterion(clf_logits, clf_gts)
        clf_loss = self.clf_coeff * clf_loss
       
        loss = lm_loss + clf_loss        
        
        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return lm_loss.cpu().data.item() * norm, clf_loss.cpu().data.item()
    
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