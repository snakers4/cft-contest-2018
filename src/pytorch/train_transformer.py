# basic imports
import tqdm
import re
import ast
import copy
import time
import glob
import os,sys
import shutil
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd

# fiddle with python path to import from parent folders
module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

# task specific custom modules
from optimizer import OpenAIAdam
from utils.text_utils.text_utils import pckl,upkl
from transformer_train_utils import train,validate,evaluate
from transformer_loss import MultipleChoiceLossCompute
from utils.text_utils.transformer_dataset import TransformerDataset
from transformer_model import (
    DoubleHeadModel,
    DEFAULT_CONFIG,
    dotdict
    )
from classifier.train_utils import str2bool

# utils
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser()

# transformer specific loss and optimization params
# adam or openai_adam 
parser.add_argument('--optimizer',   type=str,   default='openai_adam') 
parser.add_argument('--lr',          type=float, default=6.25e-5)
parser.add_argument('--freeze_emb',  type=str2bool, default=False)
# only for plain adam
parser.add_argument('--emb_lr',      type=float, default=6.25e-6)

# openai warmup schedules or plateau decay
parser.add_argument('--lr_schedule', type=str,   default='warmup_linear') 

# all the below flags are related to open-ai adam only
# default values from openai adam
parser.add_argument('--l2',            type=float, default=0.01)
parser.add_argument('--max_grad_norm', type=int,   default=1)
parser.add_argument('--lr_warmup',     type=float, default=0.002)
parser.add_argument('--b1',            type=float,    default=0.9)
parser.add_argument('--b2',            type=float,    default=0.999)
parser.add_argument('--e',             type=float,    default=1e-8)
parser.add_argument('--lm_coef',       type=float,    default=0.5)
parser.add_argument('--vector_l2',     type=str2bool, default=False)


# transformer model specific params
# change model capacity here

# replaced by max len
# parser.add_argument('--n_ctx', type=int, default=512)
# replaced by emb_size
# parser.add_argument('--n_embd', type=int, default=768)

# embedding size
parser.add_argument('--emb_size',    type=int,   default=300)
# model capacity
parser.add_argument('--n_head',      type=int,   default=12)
parser.add_argument('--n_layer',     type=int,   default=12)

# regularization
parser.add_argument('--embd_pdrop',  type=float, default=0.1)
parser.add_argument('--attn_pdrop',  type=float, default=0.1)
parser.add_argument('--resid_pdrop', type=float, default=0.1)
parser.add_argument('--clf_pdrop',   type=float, default=0.1)
parser.add_argument('--act_fn',      type=str,   default='gelu')

# data loader params
parser.add_argument('--dataset_path',  type=str,      default='data/all_corpuses_2018_09_06_processed_le.feather')
parser.add_argument('--w2i_path',      type=str,      default='data/2018_09_07_encoding_w2i.pickle')
parser.add_argument('--emb_path',      type=str,      default='data/2018_09_07_encoding_embedding.pth.tar')
parser.add_argument('--seed',          type=int,      default=42)
parser.add_argument('--train_sources', type=str,      default="['SEO_cluster','SEO_dash','Synonyms']")
parser.add_argument('--test_sources',  type=str,      default="['order','select','tag']")

# main params

parser.add_argument('--no_cuda',      action='store_true', default=False)
# elbow in distribution after 5-6 words
parser.add_argument('--max_len',      type=int,      default=7)
parser.add_argument('--batch_size',   type=int,      default=480)
parser.add_argument('--num_workers',  type=int,      default=4)
parser.add_argument('--epochs',       type=int,      default=50)
parser.add_argument('--fold',         type=int,      default=0)
# how many classes for the classification head
parser.add_argument('--target_column',  type=str,      default='s_id_encoded')
parser.add_argument('--clf_classes',    type=int,      default=4579)

# logging
parser.add_argument('--print_freq',  default=7,     type=int)
parser.add_argument('--tensorboard', default=False, type=str2bool, help='Tensorboard logging (default False)')
parser.add_argument('--tb_name',     default=None,  help='Name for tb logs')
parser.add_argument('--csv_log',     default=None,  type=str, help='File for CSV logs, True or existing file, default None')
parser.add_argument('--calc_hdice',  default=True, type=str2bool)

# evaluation
parser.add_argument('--resume',              default='',        type=str)
parser.add_argument('--evaluate',            dest='evaluate',   action='store_true')

# global vars
best_acc1 = 0
train_minib_counter = 0
valid_minib_counter = 0
test_minib_counter = 0

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
tb_name = args.tb_name

# override the default transformer config params
transformer_config = dotdict({**DEFAULT_CONFIG, **{'n_embd':args.emb_size,
                                           'n_head':args.n_head,
                                           'n_layer':args.n_layer,
                                           'embd_pdrop':args.embd_pdrop,
                                           'attn_pdrop':args.attn_pdrop,
                                           'resid_pdrop':args.resid_pdrop,
                                           'clf_pdrop':args.clf_pdrop,
                                           'afn':args.act_fn,
                                           'clf_classes':args.clf_classes
                                          }})

base_dset_kwargs = {
    'df_path':args.dataset_path,
    'w2i_path':args.w2i_path,
    'mode':'train',
    'max_len':args.max_len,
    'fold':args.fold,
    'keyword_column':'keyword',
    'source_column':'source',
    'trainval_sources':ast.literal_eval(args.train_sources),
    'test_source':ast.literal_eval(args.test_sources),
    'random_state':args.seed,
    'return_meta':True,
    'target_column':args.target_column,
}

if args.evaluate:
    base_dset_kwargs['return_string'] = True
    
    

def get_datasets(base_dset_kwargs):
    global args
    """
    train_dataset = TransformerDataset(**{**base_dset_kwargs,
                                       **{'mode':'train'}})
    
    val_dataset   = TransformerDataset(**{**base_dset_kwargs,
                                       **{'mode':'val'}})
    
    """
    test_dataset  = TransformerDataset(**{**base_dset_kwargs,
                                       **{'mode':'test'}})    
    if args.no_cuda:
        pin_memory=False
    else:
        pin_memory=True
     
    """
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,        
        shuffle=True,
        sampler=None,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,        
        shuffle=True,
        sampler=None,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False)
    """
    train_dataset = None
    val_dataset = None
    train_loader = None
    val_loader = None
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,        
        shuffle=True,
        sampler=None,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False)
    
    return train_dataset,val_dataset,test_dataset,train_loader,val_loader,test_loader

def main():
    global base_dset_kwargs
    global transformer_config, clf_token    
    global args, best_acc1, device, tb_name
    global train_minib_counter, valid_minib_counter, test_minib_counter

    # TODO refactor here - support training from scratch
    w2i = upkl(args.w2i_path)
    embeds =  torch.load(args.emb_path)

    # check that embeddings have i) correct length ii) size
    # append the pre-processed embedding 
    # check that the index is derived from the same embeddings
    assert len(w2i)      == embeds.size(0)

    # TODO refactor to have less mess
    # add positional embeddings, add classify tag
    embeddings_to_add = args.max_len + 1
    embeds = torch.cat([embeds,torch.zeros(embeddings_to_add,args.emb_size)])
    assert args.emb_size == embeds.size(1)    
    
    label_writers = []
    
    if args.csv_log:
        columns = ['epoch','tb_name',
                   'train_acc1','val_acc1','test_acc1',
                   'train_acc5','val_acc5','test_acc5',
                   'train_lm_hdice','val_lm_hdice','test_lm_hdice'
                  ]
        try:
            csv_log = pd.read_csv(args.csv_log, index_col='options')
        except:
            csv_log = pd.DataFrame(columns=columns)
        csv_log.loc[str(args)] = 0
    
        assert list(csv_log.columns) == columns
    
    
    print('Saving logs to {}'.format(args.csv_log))

    train_dataset,val_dataset,test_dataset,train_loader,val_loader,test_loader = get_datasets(base_dset_kwargs)
    
    # check that the dataset's w2i len is the same as the embeddings we load
    # assert len(train_dataset.w2i) == embeds.size(0)
    assert len(test_dataset.w2i) == embeds.size(0)
    
    # clf_token = train_dataset.clf_token
    clf_token = test_dataset.clf_token
    
    model = DoubleHeadModel(transformer_config,
                            clf_token,
                            'multiple_choice',
                            vocab=embeds.size(0),
                            n_ctx=args.max_len,
                            embeddings=embeds,
                            freeze_emb=args.freeze_emb)

    loaded_from_checkpoint = False
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume,
                                    map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])           
            # optimizer.load_state_dict(checkpoint['optimizer'])            
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
            loaded_from_checkpoint = True
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))    
    
    criterion = nn.CrossEntropyLoss(reduce=False)
    
    if args.optimizer.startswith('adam'):
        # TODO fix duplicated params
        """
        lr_param_list = [
            {'params': filter(lambda p: p.requires_grad, model.transformer.embed.parameters()), 'lr':args.emb_lr},
            {'params': filter(lambda p: p.requires_grad, model.transformer.h.parameters())},     
            {'params': filter(lambda p: p.requires_grad, model.lm_head.parameters())},
            {'params': filter(lambda p: p.requires_grad, model.task_head.parameters())},            

        ]
        """
        lr_param_list = filter(lambda p: p.requires_grad, model.parameters())
        
        optimizer = torch.optim.Adam(lr_param_list,
                                     lr=args.lr)
        
        scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)          
        
    elif args.optimizer.startswith('openai_adam'):
        
        n_updates_total = len(train_loader) * args.epochs
        
        optimizer = OpenAIAdam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.lr,
                           schedule=args.lr_schedule,
                           warmup=args.lr_warmup,
                           t_total=n_updates_total,
                           b1=args.b1,
                           b2=args.b2,
                           e=args.e,
                           l2=args.l2,
                           vector_l2=args.vector_l2,
                           max_grad_norm=args.max_grad_norm)

    # model.to(device)
    # model = nn.DataParallel(model)
    
    if args.no_cuda:
        model = model.to(device)
    else:
        model = torch.nn.DataParallel(model).to(device) 
    
    compute_loss_fct = MultipleChoiceLossCompute(criterion,
                                                 criterion,
                                                 args.lm_coef,
                                                 optimizer,
                                                 calc_lm_hdice = args.calc_hdice)    
    
    
    if args.tensorboard:
        writer = SummaryWriter('runs/{}'.format(tb_name))
    
    if args.evaluate:
        print('Evaluation starts...')
        
        cols = ['dataset',
                'text',
                'target',
                'top_probs',
                'top_indices',
                'top_lm_probs',
                'top_lm_indices']
        
        eval_df = pd.DataFrame(columns=cols)    
        save_path = 'eval/{}'.format(args.tb_name)
        
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        
        """
        for source,loader in zip(['test','train','val'],
                                 [test_loader,train_loader,val_loader]):
            
        """
        for source,loader in zip(['test'],
                                 [test_loader]):   
            
            eval_df = evaluate(loader,
                               model,
                               compute_loss_fct,
                               train_minib_counter,
                               device,
                               args,
                               eval_df,
                               source)
            eval_df.reset_index(drop=True).to_csv('{}_eval.csv'.format(save_path))

        eval_df.reset_index(drop=True).to_csv('{}_eval.csv'.format(save_path))
        
    else:
        print('Training starts...')        
        for epoch in range(args.epochs):
            train_acc1, train_acc5, train_lm_loss, train_clf_loss, train_lm_hdice = train(train_loader,
                                                                          model,
                                                                          compute_loss_fct,
                                                                          epoch,
                                                                          train_minib_counter,
                                                                          writer,
                                                                          device,
                                                                          args)

            val_acc1, val_acc5, val_lm_loss, val_clf_loss, val_lm_hdice = validate(val_loader,
                                                                     model,
                                                                     compute_loss_fct,
                                                                     epoch,
                                                                     valid_minib_counter,
                                                                     writer,
                                                                     device,
                                                                     args)

            test_acc1, test_acc5, test_lm_loss, test_clf_loss, test_lm_hdice = validate(test_loader,
                                                                         model,
                                                                         compute_loss_fct,
                                                                         epoch,
                                                                         test_minib_counter,
                                                                         writer,
                                                                         device,
                                                                         args)        

            # tensorboard logging
            if args.tensorboard:
                writer.add_scalars('epoch/epoch_lm_losses', {'train_clf_loss': train_lm_loss,
                                                             'val_clf_loss': val_lm_loss,
                                                             'test_clf_loss': test_lm_loss},epoch+1) 

                writer.add_scalars('epoch/epoch_clf_losses', {'train_clf_loss': train_clf_loss,
                                                              'val_clf_loss': val_clf_loss,
                                                              'test_clf_loss': test_clf_loss},epoch+1)            

                writer.add_scalars('epoch/epoch_acc1', {'train_acc1': train_acc1,
                                                        'val_acc1': val_acc1,
                                                        'test_acc1': test_acc1},epoch+1)

                writer.add_scalars('epoch/epoch_acc5', {'train_acc5': train_acc5,
                                                        'val_acc5': val_acc5,
                                                        'test_acc5': test_acc5},epoch+1) 

                writer.add_scalars('epoch/lm_hdice', {'train_lm_hdice': train_lm_hdice,
                                                      'val_lm_hdice': val_lm_hdice,
                                                      'test_lm_hdice': test_lm_hdice},epoch+1)            

            if args.csv_log:
                csv_log.loc[str(args), 'epoch'] = epoch
                csv_log.loc[str(args), 'tb_name'] = args.tb_name
                csv_log.loc[str(args), 'train_acc1'] = train_acc1
                csv_log.loc[str(args), 'train_acc5'] = train_acc5
                csv_log.loc[str(args), 'val_acc1'] = val_acc1
                csv_log.loc[str(args), 'val_acc5'] = val_acc5
                csv_log.loc[str(args), 'test_acc1'] = test_acc1
                csv_log.loc[str(args), 'test_acc5'] = test_acc5

                csv_log.loc[str(args), 'train_lm_hdice'] = train_lm_hdice
                csv_log.loc[str(args), 'val_lm_hdice'] = val_lm_hdice
                csv_log.loc[str(args), 'test_lm_hdice'] = test_lm_hdice

            if args.optimizer.startswith('adam'):
                scheduler.step(val_acc1)

            is_best = val_acc1 > best_acc1
            best_acc1 = max(val_acc1, best_acc1)
            if not os.path.exists('weights/'):
                os.makedirs('weights/')
            save_checkpoint({
                    'epoch': epoch + 1,
                    'optimizer': optimizer.state_dict(),
                    'state_dict': model.module.state_dict(),
                    'best_acc1': best_acc1,
                    },
                    is_best,
                    'weights/{}_checkpoint.pth.tar'.format(str(tb_name)),
                    'weights/{}_best.pth.tar'.format(str(tb_name))
                    )

        if args.csv_log:
            csv_log.to_csv(args.csv_log, index_label='options')
    
def save_checkpoint(state, is_best, filename, best_filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)

if __name__ == '__main__':
    main()
