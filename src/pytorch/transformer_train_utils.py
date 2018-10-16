import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

def train(train_loader,
          model,
          compute_loss_fct,
          epoch,
          minib_counter,
          writer,
          device,
          args):
        
    batch_time = AverageMeter()
    data_time = AverageMeter()

    lm_losses = AverageMeter()
    clf_losses = AverageMeter()
    acc1_meter = AverageMeter()
    hdice_meter = AverageMeter()
    lm_acc1_meter = AverageMeter()
    
    # switch to train mode
    model.train()

    end = time.time()
    
    for i,(names,gt_names,
           input,gt,
           msk,target) in enumerate(train_loader):        
        # measure data loading time
        data_time.update(time.time() - end)

        # char sequences
        input = input.long().to(device)
        gt = gt.long().to(device)
        # clf target
        target = target.long().to(device)
        # gt mask
        msk = msk.float().to(device)
        
        lm_logits, clf_logits = model(input)
                
        clf_loss,lm_loss,hdice = compute_loss_fct(input, target,
                                                  msk,gt,
                                                  clf_logits, lm_logits,
                                                  only_return_losses=False,
                                                  return_both_losses=True)            

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()        

        # measure accuracy and record loss
        lm_losses.update(lm_loss, input.size(0))
        clf_losses.update(clf_loss, input.size(0))
        hdice_meter.update(hdice, input.size(0))

        # log the current lr
        current_lr = compute_loss_fct.opt.state_dict()['param_groups'][0]['lr']

        prec1, prec2 = accuracy(clf_logits.detach(), target, topk=(1, 2))
        # predictions are B*SEQxCLASSES
        # have to conform to to B*SEQxCLASSES vs B*SEQ
        lm_prec1, lm_prec2 = accuracy(lm_logits.detach(),
                                      gt[:, :, 0].contiguous().view(-1,1), topk=(1, 2))
        
        acc1_meter.update(prec1.item(), input.size(0))
        lm_acc1_meter.update(lm_prec1.item(), input.size(0))

        # tensorboard logging
        if args.tensorboard:
            # writer.add_scalar('train/train_lm_loss', lm_losses.val, minib_counter)
            # writer.add_scalar('train/train_clf_loss', clf_losses.val, minib_counter)
            writer.add_scalar('train/train_lr', current_lr, minib_counter)                    

        minib_counter += 1

        if i % args.print_freq == 0:
            print('Epoch:   [{0}][{1}/{2}]\t'
                  'Time     {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data     {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss LM  {lm_losses.val:.6f} ({lm_losses.avg:.6f})\t'     
                  'Loss CLF {clf_losses.val:.6f} ({clf_losses.avg:.6f})\t' 
                  'ACC1     {acc1_meter.val:.4f} ({acc1_meter.avg:.4f})\t'
                  'LM ACC1  {lm_acc1_meter.val:.4f} ({lm_acc1_meter.avg:.4f})\t'
                  'LM HDICE {hdice_meter.val:.4f} ({hdice_meter.avg:.4f})\t'.format(
                      epoch,i, len(train_loader),
                      batch_time=batch_time,data_time=data_time,
                      lm_losses=lm_losses,clf_losses=clf_losses,
                      acc1_meter=acc1_meter,lm_acc1_meter=lm_acc1_meter,
                      hdice_meter=hdice_meter))
            
        # break out of cycle early if required
        # must be used with Dataloader shuffle = True
        if args.epoch_fraction < 1.0:
            if i > len(train_loader) * min(args.epoch_fraction,0.2):
                print('Proceed to next epoch on {}/{}'.format(i,len(train_loader)))
                break            

    print(' * Avg Train ACC1 {acc1_meter.avg:.4f}'.format(acc1_meter=acc1_meter))
    print(' * Avg Train LM ACC1 {acc1_meter.avg:.4f}'.format(acc1_meter=acc1_meter))
    
    return acc1_meter.avg, lm_losses.avg, clf_losses.avg, hdice_meter.avg, lm_acc1_meter.avg
    
def validate(val_loader,
             model,
             compute_loss_fct,
             epoch,
             minib_counter,
             writer,
             device,
             args
             ):

    batch_time = AverageMeter()

    lm_losses = AverageMeter()
    clf_losses = AverageMeter()
    acc1_meter = AverageMeter()
    hdice_meter = AverageMeter()
    
    lm_acc1_meter = AverageMeter()
    
    # switch to evaluate mode
    model.eval()
    
    end = time.time()
    
    with torch.no_grad():
        for i,(names,gt_names,
               input,gt,
               msk,target) in enumerate(val_loader):    
            
            input = input.long().to(device)
            gt = gt.long().to(device)
            
            target = target.long().to(device)
            msk = msk.float().to(device)

            lm_logits, clf_logits = model(input)
            clf_loss,lm_loss, hdice = compute_loss_fct(input, target,
                                                       msk,gt,
                                                       clf_logits, lm_logits,
                                                       only_return_losses=True,
                                                       return_both_losses=True)            

            # measure accuracy and record loss
            lm_losses.update(lm_loss, input.size(0))
            clf_losses.update(clf_loss, input.size(0))
            hdice_meter.update(hdice, input.size(0))

            prec1, prec2 = accuracy(clf_logits.detach(), target, topk=(1, 2))
            # predictions are B*SEQxCLASSES
            # have to conform to to B*SEQxCLASSES vs B*SEQ
            lm_prec1, lm_prec2 = accuracy(lm_logits.detach(),
                                          gt[:, :, 0].contiguous().view(-1,1), topk=(1, 2))                
            
            acc1_meter.update(prec1.item(), input.size(0))
            lm_acc1_meter.update(lm_prec1.item(), input.size(0))            

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            minib_counter += 1

            if i % args.print_freq == 0:
                print('            [{0}/{1}]\t'
                      'Time   {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss LM  {lm_losses.val:.6f} ({lm_losses.avg:.6f})\t'     
                      'Loss CLF {clf_losses.val:.6f} ({clf_losses.avg:.6f})\t' 
                      'ACC1     {acc1_meter.val:.4f} ({acc1_meter.avg:.4f})\t'
                      'LM ACC1  {lm_acc1_meter.val:.4f} ({lm_acc1_meter.avg:.4f})\t'                      
                      'LM HDICE {hdice_meter.val:.4f} ({hdice_meter.avg:.4f})\t'.format(
                       i, len(val_loader),
                       batch_time=batch_time, 
                       lm_losses=lm_losses,clf_losses=clf_losses,
                       acc1_meter=acc1_meter,lm_acc1_meter=lm_acc1_meter,
                       hdice_meter=hdice_meter))
                
            # break out of cycle early if required
            # must be used with Dataloader shuffle = True
            if args.epoch_fraction < 1.0:
                if i > len(val_loader) * min(args.epoch_fraction,0.2):
                    print('Proceed to next epoch on {}/{}'.format(i,len(val_loader)))
                    break                     

    print(' * Avg Val ACC1 {acc1_meter.avg:.4f}'.format(acc1_meter=acc1_meter))
    return acc1_meter.avg, lm_losses.avg, clf_losses.avg, hdice_meter.avg, lm_acc1_meter.avg

def evaluate(val_loader,
             model,
             compute_loss_fct,
             minib_counter,
             device,
             args,
             eval_df,
             source
             ):

    batch_time = AverageMeter()

    sm = torch.nn.Softmax(dim=1)
    sm_lm = torch.nn.Softmax(dim=2)
    
    # lm_losses = AverageMeter()
    # clf_losses = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    
    lm_acc1_meter = AverageMeter()
    lm_acc5_meter = AverageMeter()
    lm_acc100_meter = AverageMeter()
    
    # switch to evaluate mode
    model.eval()
    
    end = time.time()
    
    with torch.no_grad():
        with tqdm(total=int(len(val_loader))) as pbar:        
            for i,(input,msk,target,lens,sources,texts) in enumerate(val_loader):

                input = input.long().to(device)
                target = target.long().to(device)
                msk = msk.float().to(device)

                # teacher forcing logits for lm evaluation
                x_shifted  = input[:, 1:, 0].contiguous().view(-1)
                
                lm_logits, clf_logits = model(input)

                top_values, top_indices = torch.topk(sm(clf_logits),
                                               5,
                                               dim=1,
                                               largest=True,
                                               sorted=True)
                
                # resize to BxLMxVOCAB shape
                # then apply the mask to ignore the meaningless positions
                lm_logits_ = lm_logits.view(clf_logits.size(0),
                                            -1,
                                            lm_logits.size(1))
                
                # lm_logits_ = lm_logits_ * mask[:, 1:].expand_as(expand_as)                
                
                top_lm_values, top_lm_indices = torch.topk(sm_lm(lm_logits_),
                                               5,
                                               dim=2,
                                               largest=True,
                                               sorted=True)                

                _dict = {'dataset':source,
                         'text': list(texts),
                         'target': list(target.cpu().numpy()),
                         'top_probs': list(top_values.cpu().numpy()),
                         'top_indices': list(top_indices.cpu().numpy()),
                         
                         'top_lm_probs': top_lm_values.cpu().numpy().tolist(),
                         'top_lm_indices': top_lm_indices.cpu().numpy().tolist()
                        }

                eval_df = eval_df.append(pd.DataFrame.from_dict(_dict))              

                # clf_loss,lm_loss = compute_loss_fct(input, target, msk, clf_logits, lm_logits,
                #                                     only_return_losses=True,
                #                                     return_both_losses=True)            

                # measure accuracy and record loss
                # lm_losses.update(lm_loss, input.size(0))
                # clf_losses.update(clf_loss, input.size(0))

                prec1, prec5 = accuracy(clf_logits.detach(), target, topk=(1, 5))
                acc1_meter.update(prec1.item(), input.size(0))
                acc5_meter.update(prec5.item(), input.size(0))           

                lm_prec1, lm_prec5, lm_prec100 = accuracy(lm_logits.detach(), x_shifted, topk=(1, 5, 100))
                lm_acc1_meter.update(lm_prec1.item(), input.size(0))
                lm_acc5_meter.update(lm_prec5.item(), input.size(0))
                lm_acc100_meter.update(lm_prec100.item(), input.size(0))           
                
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                minib_counter += 1

                if i % args.print_freq == 0:
                    print(' [{0}/{1}]\t'
                          'Time     {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'ACC1     {acc1_meter.val:.4f} ({acc1_meter.avg:.4f})\t'
                          'ACC5     {acc5_meter.val:.4f} ({acc5_meter.avg:.4f})\t'
                          'LM 1     {lm_acc1_meter.val:.4f} ({lm_acc1_meter.avg:.4f})\t'
                          'LM 5     {lm_acc5_meter.val:.4f} ({lm_acc5_meter.avg:.4f})\t'
                          'LM 100   {lm_acc100_meter.val:.4f} ({lm_acc100_meter.avg:.4f})\t'
                          .format(
                           i, len(val_loader),
                           batch_time=batch_time, 
                           acc1_meter=acc1_meter,acc5_meter=acc5_meter,
                           lm_acc1_meter=lm_acc1_meter,
                           lm_acc5_meter=lm_acc5_meter,lm_acc100_meter=lm_acc100_meter))
                    
                pbar.update(1)

    print(' * Avg Eval ACC1 {acc1_meter.avg:.4f}'.format(acc1_meter=acc1_meter))
    print(' * Avg Eval ACC5 {acc5_meter.avg:.4f}'.format(acc5_meter=acc5_meter))
    print(' * Avg Eval LM ACC1 {lm_acc1_meter.avg:.4f}'.format(lm_acc1_meter=lm_acc1_meter))    
    print(' * Avg Eval LM ACC5 {lm_acc5_meter.avg:.4f}'.format(lm_acc5_meter=lm_acc5_meter))
    print(' * Avg Eval LM ACC100 {lm_acc100_meter.avg:.4f}'.format(lm_acc100_meter=lm_acc100_meter))
    
    return eval_df

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
        
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res        

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')   