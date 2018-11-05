# basic imports
import re
import gc
import ast
import copy
import time
import math
import glob
import os,sys
import shutil
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torchtext import data, datasets
from torch.optim.lr_scheduler import ReduceLROnPlateau

# task specific custom modules
from pytorch.encoder_decoder import make_model
from pytorch.transformer_train_utils import str2bool
from pytorch.encoder_decoder_utils import (Batch,rebatch,
                                           SimpleLossCompute,
                                           cleaner,
                                           tokenize,
                                           greedy_decode_batch,
                                           beam_decode_batch,
                                           lookup_words,
                                           run_epoch,
                                           Tokenizer
                                           )

from pytorch.metrics import (score_task1,
                             score_task2)

# utils
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser()

# model
parser.add_argument('--num_layers',     type=int,   default=2)
parser.add_argument('--num_classes',    type=int,   default=3)
parser.add_argument('--emb_size',       type=int,   default=50)
parser.add_argument('--hidden_size',    type=int,   default=256)
parser.add_argument('--dropout',        type=float, default=0.2)
parser.add_argument('--cn_emb_size',    type=int,   default=0)
parser.add_argument('--num_cn',         type=int,   default=0)
parser.add_argument('--beam_width',     type=int,   default=0)
parser.add_argument('--heavy_decoder',  type=str2bool, default=False)
parser.add_argument('--add_skip',       type=str2bool, default=False)

parser.add_argument('--load_optimizer_state',  type=str2bool, default=False)

# dataset
parser.add_argument('--batch_size',         type=int,      default=256)
parser.add_argument('--train_df_path',      type=str,      default='../data/proc_train.csv') # only to build a vocab
parser.add_argument('--trn_df_path',        type=str,      default='../data/proc_trn.csv') # pre-made trainval splits
parser.add_argument('--val_df_path',        type=str,      default='../data/proc_val.csv')
parser.add_argument('--test_df_path',       type=str,      default='../data/proc_test.csv')
parser.add_argument('--bpe_model',          type=str,      default='')

parser.add_argument('--min_freq',           type=int,      default=0)

# optimizer
parser.add_argument('--lr',           type=float,    default=1e-3)
parser.add_argument('--epochs',       type=int,      default=50)
parser.add_argument('--seed',         type=int,      default=42)
parser.add_argument('--clf_coeff',    type=float,    default=1.0)
parser.add_argument('--lr_factor',    type=float,    default=0.5)

parser.add_argument('--seq_penalize_only_errors', type=str2bool, default=False)

# logging
parser.add_argument('--print_freq',  default=10,     type=int)
parser.add_argument('--tensorboard', default=False, type=str2bool, help='Tensorboard logging (default False)')
parser.add_argument('--tb_name',     default=None,  help='Name for tb logs')

# evaluation
parser.add_argument('--resume',      default='',        type=str)
parser.add_argument('--evaluate',    dest='evaluate',   action='store_true')
parser.add_argument('--predict',     dest='predict',   action='store_true')

# global vars
best_met = 0
valid_minib_counter = 0

args = parser.parse_args()

# we will use CUDA if it is available
USE_CUDA = torch.cuda.is_available()
DEVICE=torch.device('cuda:0') # or set to 'cpu'
print("CUDA:", USE_CUDA)
print(DEVICE)

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ["USE_CUDA"] = str(USE_CUDA)

tb_name = args.tb_name

def main():
    global USE_CUDA,DEVICE
    global UNK_TOKEN,PAD_TOKEN,SOS_TOKEN,EOS_TOKEN,TRG_NAMES,LOWER,PAD_INDEX,NAMES,MIN_FREQ
    global args,best_met,valid_minib_counter
    global pieces2text

    label_writers = []

    UNK_TOKEN = "!"
    PAD_TOKEN = "_"    
    SOS_TOKEN = "["
    EOS_TOKEN = "]"
    LOWER = False
    
    # use sentence piece instead of plain list
    if args.bpe_model != '':
        tokenizer   = Tokenizer(args.bpe_model)
        text2pieces = tokenizer.enc_text2pieces
        pieces2text = tokenizer.dec_pieces2text
        tokenize    = text2pieces
        print('BPE Tokenizer loaded!')
        print(args.bpe_model)
    else:
        from pytorch.encoder_decoder_utils import tokenize        
    
    ID = data.Field(sequential=False,
                    use_vocab=False)

    NAMES = data.Field(tokenize=tokenize,
                       batch_first=True,
                       lower=LOWER,
                       include_lengths=True,
                       unk_token=UNK_TOKEN,
                       pad_token=PAD_TOKEN,
                       init_token=None,
                       eos_token=EOS_TOKEN)

    TRG_NAMES = data.Field(tokenize=tokenize, 
                           batch_first=True,
                           lower=LOWER,
                           include_lengths=True,
                           unk_token=UNK_TOKEN,
                           pad_token=PAD_TOKEN,
                           init_token=SOS_TOKEN,
                           eos_token=EOS_TOKEN)

    LBL = data.Field(sequential=False,
                     use_vocab=False)

    CNT = data.Field(sequential=False,
                     use_vocab=False)

    datafields = [("id", ID),
                  ("src", NAMES),
                  ("trg", TRG_NAMES),
                  ("clf", LBL),
                  ("cn", CNT)
                 ]

    trainval_data = data.TabularDataset(path=args.train_df_path,
                                     format='csv',
                                     skip_header=True,
                                     fields=datafields)
    
    if not args.predict and not args.evaluate:
        train_data = data.TabularDataset(path=args.trn_df_path,
                                       format='csv',
                                       skip_header=True,
                                       fields=datafields)

        val_data = data.TabularDataset(path=args.val_df_path,
                                       format='csv',
                                       skip_header=True,
                                       fields=datafields)
        
        print('Train length {}, val length {}'.format(len(train_data),len(val_data)))        
    
    if args.predict:
        test_data = data.TabularDataset(path=args.test_df_path,
                                        format='csv',
                                        skip_header=True,
                                        fields=datafields)    

    MIN_FREQ = args.min_freq  # NOTE: we limit the vocabulary to frequent words for speed
    NAMES.build_vocab(trainval_data.src, min_freq=MIN_FREQ)
    TRG_NAMES.build_vocab(trainval_data.trg, min_freq=MIN_FREQ)
    PAD_INDEX = TRG_NAMES.vocab.stoi[PAD_TOKEN]
    
    del trainval_data
    gc.collect()

    if not args.predict and not args.evaluate:
        train_iter = data.BucketIterator(train_data,
                                         batch_size=args.batch_size,
                                         train=True, 
                                         sort_within_batch=True, 
                                         sort_key=lambda x: (len(x.src), len(x.trg)),
                                         repeat=False,
                                         device=DEVICE,
                                         shuffle=True)

        valid_iter_batch = data.Iterator(val_data,
                                   batch_size=args.batch_size,
                                   train=False,
                                   sort_within_batch=True,
                                   sort_key=lambda x: (len(x.src), len(x.trg)),
                                   repeat=False, 
                                   device=DEVICE,
                                   shuffle=False)
        
        val_ids = []
        for b in valid_iter_batch:
            val_ids.extend(list(b.id.cpu().numpy()))
            
        print('Preparing data for validation')

        train_df = pd.read_csv('../data/proc_train.csv')
        train_df = train_df.set_index('id')

        # val_gts = train_df.loc[val_ids,'fullname_true'].values
        # val_ors = train_df.loc[val_ids,'fullname'].values
        # incorrect_idx = list(train_df[train_df.target==1].index.values)
        # incorrect_val_ids = list(set(val_ids).intersection(set(incorrect_idx)))
        # correct_val_ids = list(set(val_ids)-set(incorrect_val_ids))

        print('Making dictionaries')

        id2gt = dict(train_df['fullname_true'])
        id2clf_gt = dict(train_df['target'])
        val_gts = [id2gt[_] for _ in val_ids]
        val_clf_gts = [id2clf_gt[_] for _ in val_ids]    
        del train_df
        gc.collect()    
        
    if args.evaluate:
        val_data = data.TabularDataset(path=args.val_df_path,
                                       format='csv',
                                       skip_header=True,
                                       fields=datafields)
        
        valid_iter_batch = data.Iterator(val_data,
                                   batch_size=args.batch_size,
                                   train=False,
                                   sort_within_batch=True,
                                   sort_key=lambda x: (len(x.src), len(x.trg)),
                                   repeat=False, 
                                   device=DEVICE,
                                   shuffle=False)
        
        val_ids = []
        for b in valid_iter_batch:
            val_ids.extend(list(b.id.cpu().numpy()))
            
        print('Preparing data for validation')

        train_df = pd.read_csv('../data/proc_train.csv')
        train_df = train_df.set_index('id')
        
        print('Making dictionaries')

        id2gt = dict(train_df['fullname_true'])
        id2clf_gt = dict(train_df['target'])
        val_gts = [id2gt[_] for _ in val_ids]
        val_clf_gts = [id2clf_gt[_] for _ in val_ids]    
        del train_df
        gc.collect()
        
    if args.predict:
        test_iter_batch = data.Iterator(test_data,
                                        batch_size=args.batch_size,
                                        train=False,
                                        sort_within_batch=True,
                                        sort_key=lambda x: (len(x.src), len(x.trg)),
                                        repeat=False, 
                                        device=DEVICE,
                                        shuffle=False)    


        
        test_ids = []
        for b in test_iter_batch:
            test_ids.extend(list(b.id.cpu().numpy()))         

    
    model = make_model(len(NAMES.vocab),
                       len(TRG_NAMES.vocab),
                       device=DEVICE,
                       emb_size=args.emb_size,
                       hidden_size=args.hidden_size,
                       num_layers=args.num_layers,
                       dropout=args.dropout,
                       num_classes=args.num_classes,
                       num_cn=args.num_cn,
                       cn_emb_size=args.cn_emb_size,
                       heavy_decoder=args.heavy_decoder,
                       add_input_skip=args.add_skip)
    
    loaded_from_checkpoint = False

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume,
                                    map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            best_met = checkpoint['best_met']
            model.load_state_dict(checkpoint['state_dict'])           
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
            
            loaded_from_checkpoint = True
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        args.start_epoch = 0
    
    criterion = nn.CrossEntropyLoss(reduce=False).to(DEVICE)
    
    if args.tensorboard:
        writer = SummaryWriter('runs_encdec/{}'.format(tb_name))
    
    if args.evaluate:
        print('Running prediction on val set')
        if not os.path.exists('eval/'):
            os.makedirs('eval/')
            
        preds,clf_preds = predict((rebatch(PAD_INDEX, x) for x in valid_iter_batch),
                                  model, max_len=70, src_vocab=NAMES.vocab, trg_vocab=TRG_NAMES.vocab,
                                  num_batches=len(valid_iter_batch),return_logits=True)
        
        predict_df = pd.DataFrame(
            {'id': val_ids,
             'target': clf_preds,
             'fullname_true':preds
            })
        
        predict_df.set_index('id').to_csv('eval/{}.csv'.format(args.tb_name)) 
    if args.predict:
        print('Running prediction on test set')
        if not os.path.exists('predictions/'):
            os.makedirs('predictions/')
            
        preds,clf_preds = predict((rebatch(PAD_INDEX, x) for x in test_iter_batch),
                                  model, max_len=70, src_vocab=NAMES.vocab, trg_vocab=TRG_NAMES.vocab,
                                  num_batches=len(test_iter_batch),return_logits=True)
        
        predict_df = pd.DataFrame(
            {'id': test_ids,
             'target': clf_preds,
             'fullname_true':preds
            })
        
        predict_df.set_index('id').to_csv('predictions/{}.csv'.format(args.tb_name))
    if not args.predict and not args.evaluate:
        print('Training starts...') 
        dev_perplexity,dev_clf_loss,preds,clf_preds = train(model,
                                                            lr=args.lr,
                                                            num_epochs=args.epochs,
                                                            print_every=args.print_freq,
                                                            train_iter=train_iter,
                                                            valid_iter_batch=valid_iter_batch,
                                                            val_ids=val_ids,
                                                            val_clf_gts=val_clf_gts,
                                                            val_gts=val_gts,
                                                            writer=writer)        

def train(model,
          num_epochs=10,
          lr=1e-3,
          print_every=10,
          train_iter=None,
          valid_iter_batch=None,
          val_ids=None,
          val_clf_gts=None,
          val_gts=None,
          writer=None):
    
    global best_met
    global UNK_TOKEN,PAD_TOKEN,SOS_TOKEN,EOS_TOKEN,TRG_NAMES,LOWER,PAD_INDEX,NAMES,MIN_FREQ
        
    # optionally add label smoothing; see the Annotated Transformer
    # criterion = nn.NLLLoss(reduce=, ignore_index=PAD_INDEX)
 
    criterion = nn.NLLLoss(size_average=False, ignore_index=PAD_INDEX)
    clf_criterion = nn.CrossEntropyLoss()
    
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    
    if args.load_optimizer_state:
        optim.load_state_dict(checkpoint['optimizer'])
    else:
        print('Optimizer state NOT loaded')     
        
    scheduler = ReduceLROnPlateau(optim,
                                  mode='max',
                                  factor=args.lr_factor,
                                  patience=5,
                                  verbose=True,
                                  min_lr=1e-7)    
    
    # dev_perplexities = []
    # dev_preds = []

    for epoch in range(args.start_epoch, num_epochs):
      
        print("Epoch", epoch)
        print('Training the model')
        model.train()
        
        train_perplexity, train_clf_loss = run_epoch((rebatch(PAD_INDEX, b) for b in train_iter), 
                                                     model,
                                                     SimpleLossCompute(model.generator,
                                                                       criterion,clf_criterion,
                                                                       optim,
                                                                       seq_penalize_only_errors=args.seq_penalize_only_errors,
                                                                       clf_coeff=args.clf_coeff),
                                                     print_every=print_every,
                                                     num_batches=len(train_iter),
                                                     epoch_no=epoch)
       
        model.eval()
        with torch.no_grad():
            print('Evaluating the model')
            """
            print_examples((rebatch(PAD_INDEX, x) for x in valid_iter), 
                           model, n=10, src_vocab=NAMES.vocab, trg_vocab=TRG_NAMES.vocab)        
            """
            dev_perplexity, dev_clf_loss = run_epoch((rebatch(PAD_INDEX, b) for b in valid_iter_batch), 
                                                     model, 
                                                     SimpleLossCompute(model.generator,
                                                                       criterion, clf_criterion,
                                                                       None),
                                                     num_batches=len(valid_iter_batch),
                                                     epoch_no=epoch)
            

            preds,clf_preds = predict((rebatch(PAD_INDEX, x) for x in valid_iter_batch),
                                      model, max_len=70, src_vocab=NAMES.vocab, trg_vocab=TRG_NAMES.vocab,
                                      num_batches=len(valid_iter_batch)) 
            
            df_true = pd.DataFrame(
                {'id': val_ids,
                 'target': val_clf_gts,
                 'fullname_true':val_gts
                })
            df_pred = pd.DataFrame(
                {'id': val_ids,
                 'target': clf_preds,
                 'fullname_true':preds
                })

            score1 = score_task1(df_true, df_pred)
            score2 = score_task2(df_true, df_pred)
            score = (score1 + score2)/2
            
            print("Validation CLF Loss: %f" % dev_clf_loss)
            print("Validation perplexity: %f" % dev_perplexity)
            print("score {} / Score 1 {} / Score 2 {}".format(score,score1,score2))
            
            if args.tensorboard:
                writer.add_scalars('clf_losses', {'dev_clf_loss': dev_clf_loss,
                                                  'train_clf_loss': train_clf_loss},epoch+1) 

                writer.add_scalars('lm_losses', {'dev_perplexity': dev_perplexity,
                                                 'train_perplexity': train_perplexity},epoch+1)            

                writer.add_scalars('scores', {'score1': score1,
                                              'score2': score2,
                                              'score': score,},epoch+1)

            scheduler.step(score)

            is_best = score > best_met
            best_acc1 = max(score, best_met)
            if not os.path.exists('weights/'):
                os.makedirs('weights/')
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optim.state_dict(),                
                    'best_met': best_met,
                    },
                    is_best,
                    'weights/{}_checkpoint.pth.tar'.format(str(tb_name)),
                    'weights/{}_best.pth.tar'.format(str(tb_name))
                    )
        
    return dev_perplexity,dev_clf_loss,preds,clf_preds, score1,score2

def predict(example_iter, model, max_len=100, 
            sos_index=1, 
            src_eos_index=None, 
            trg_eos_index=None, 
            src_vocab=None, trg_vocab=None,
            num_batches=100,
            return_logits=False):

    global UNK_TOKEN,PAD_TOKEN,SOS_TOKEN,EOS_TOKEN,TRG_NAMES,LOWER
    global args, DEVICE
    global pieces2text
    model.eval()
    count = 0
    
    if src_vocab is not None and trg_vocab is not None:
        src_eos_index = src_vocab.stoi[EOS_TOKEN]
        trg_sos_index = trg_vocab.stoi[SOS_TOKEN]
        trg_eos_index = trg_vocab.stoi[EOS_TOKEN]
    else:
        src_eos_index = None
        trg_sos_index = 1
        trg_eos_index = None

    preds = []
    clf_preds = []

    with tqdm(total=num_batches) as pbar:
        for i, batch in enumerate(example_iter):
            
            if args.beam_width==0:
                output, pred_classes = greedy_decode_batch(
                    model, batch.src, batch.src_mask, batch.src_lengths,
                    max_len=max_len, sos_index=trg_sos_index, eos_index=trg_eos_index,
                    return_logits=return_logits,cn=batch.cn,
                    add_skip=args.add_skip
                )
            else:
                output, pred_classes = beam_decode_batch(
                    model, batch.src, batch.src_mask, batch.src_lengths,
                    max_len=max_len, sos_index=trg_sos_index, eos_index=trg_eos_index,
                    return_logits=return_logits,cn=batch.cn,
                    device=DEVICE,beam_width=args.beam_width,values_to_return=1,
                    add_skip=args.add_skip
                )

            clf_preds.extend(list(pred_classes))
            
            # cut off everything starting from </s> 
            # (only when eos_index provided)
            if trg_eos_index is not None:
                # iterate over sentence predictions and cut off from eos
                for pred in output:
                    first_eos = np.where(pred==trg_eos_index)[0]
                    if len(first_eos) > 0:
                        # produce sentences
                        if args.bpe_model != '':
                            _ = lookup_words(pred[:first_eos[0]],
                                             vocab=TRG_NAMES.vocab)                           
                            _ = pieces2text(_)
                            preds.append(_)
                        else:
                            preds.append("".join(lookup_words(pred[:first_eos[0]],
                                             vocab=TRG_NAMES.vocab)))
                    else:
                        if args.bpe_model != '':
                            _ = lookup_words(pred[:],
                                             vocab=TRG_NAMES.vocab)
                            _ = pieces2text(_)
                            preds.append(_)
                        else:
                            preds.append("".join(lookup_words(pred[:],
                                                              vocab=TRG_NAMES.vocab)))                            
            pbar.update(1)
    return preds,clf_preds            

def save_checkpoint(state, is_best, filename, best_filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)

if __name__ == '__main__':
    main()  