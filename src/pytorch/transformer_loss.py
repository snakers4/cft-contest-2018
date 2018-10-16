import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class MultipleChoiceLossCompute:
    "A Loss compute and train function for multiple choice tasks."

    def __init__(self, lm_criterion, clf_criterion, lm_coef,
                 opt=None,
                 calc_lm_hdice=False,
                 hdice_ths=0.25,
                 lm_loss_weight = 1.0,
                 clf_loss_weight = 1.0):
        
        self.lm_loss_weight = lm_loss_weight
        self.clf_loss_weight = clf_loss_weight
        
        self.lm_criterion = lm_criterion
        self.clf_criterion = clf_criterion
        self.calc_lm_hdice = calc_lm_hdice
        self.lm_coef = lm_coef
        self.opt = opt
        
        if self.calc_lm_hdice:
            self.hdice = HardDice(threshold=hdice_ths)

    def __call__(self,
                 X, Y, # input sequence, target classification, 
                 M, X_gt, # the target (!) mask, the target sequence
                 clf_logits,
                 lm_logits=None,
                 only_return_losses=False,
                 return_both_losses=True):
        
        # assert that Y is Bx1
        assert len(Y.size()) == 1
        
        # all B dimensions are the same
        assert Y.size(0) == M.size(0)
        assert X.size(0) == M.size(0)
        
        # X and X_gt
        assert X.size() == X_gt.size()
        
        # conform X tensor to BxSEQx(TOKEN+SEQUENCE)format
        assert len(X.size())==3
        assert len(X_gt.size())==3
        
        # X = X.view(-1, X.size(-2), X.size(-1))
        
        # Language modeling loss
        if lm_logits is not None:
            # conform M tensor to BxSEQ format
            assert len(M.size())==2
            # M = M.view(-1, M.size(2))
            
            # use teacher forcing and skip the first token
            # BxSEQx2
            # x_shifted  = X[:, 1:, 0].contiguous().view(-1)
            
            # the difference between this and language modelling is that we have a set gt
            # also our gt is not shifted
            
            x_shifted  = X_gt[:, :, 0].contiguous().view(-1)
            
            if self.calc_lm_hdice:
                # BxSEQ-1
                x_shifted_ = X[:, :, 0].contiguous()
                assert len(x_shifted_.size())==2                
                # create a ohe version of x tensor
                # BxSEQ-1 => BxSEQ-1xVOCAB => ALLxVOCAB
                x_shifted_ohe = np.zeros((lm_logits.size(0), lm_logits.size(1)))
                x_shifted_ = x_shifted_.contiguous().view(-1)
                x_shifted_ohe[np.arange(lm_logits.size(0)), x_shifted_.cpu().contiguous().numpy()] = 1   
                x_shifted_ohe = torch.from_numpy(x_shifted_ohe).to(lm_logits.device)
                
                hdice = self.hdice(lm_logits,x_shifted_ohe)
            
            lm_losses = self.lm_criterion(lm_logits, x_shifted)
            # lm_losses = lm_losses.view(X.size(0), X.size(1) - 1)
            # lm_losses = lm_losses * M[:, 1:]
            # lm_losses = lm_losses.sum(1) / torch.sum(M[:, 1:], 1)
            
            # the difference between this and language modelling is that we have a set gt
            # also our gt is not shifted
            
            lm_losses = lm_losses.view(X.size(0), X.size(1))
            # M is gt mask, in case the gt is longer
            lm_losses = lm_losses * M[:,:]
            lm_losses = lm_losses.sum(1) / torch.sum(M[:,:], 1)            
            
        # сlassification loss
        clf_losses = self.clf_criterion(clf_logits, Y)
        
        
        # do not do backward pass
        if only_return_losses:
            clf_loss = clf_losses.sum() * self.clf_loss_weight
            if lm_logits is not None:
                lm_loss = lm_losses.sum() * self.lm_loss_weight
            else:
                lm_loss = 0
            
            train_loss = clf_loss + self.lm_coef * lm_loss
            
            if return_both_losses:
                if self.calc_lm_hdice:
                    if type(lm_loss)==int:
                        return clf_loss.item(),0,hdice.item()
                    else:
                        return clf_loss.item(),lm_loss.item(),hdice.item()
                else:
                    return clf_loss.item(),lm_loss.item()
            else:
                return train_loss.item()                

        # do backward pass
        if self.lm_coef > 0 and lm_logits is not None:
            clf_loss = clf_losses.sum() * self.clf_loss_weight
            lm_loss = lm_losses.sum() * self.lm_loss_weight
            train_loss = clf_loss + self.lm_coef * lm_loss
        else:
            clf_loss = clf_losses.sum()
            lm_loss = 0
            train_loss = clf_loss
            
        train_loss.backward()
        
        if self.opt is not None:
            self.opt.step()
            self.opt.zero_grad()
            
        if return_both_losses:
            if self.calc_lm_hdice:
                if type(lm_loss)==int:
                    return clf_loss.item(),0,0
                else:
                    return clf_loss.item(),lm_loss.item(),hdice.item()
            else:
                return clf_loss.item(),lm_loss.item()
        else:
            return train_loss.item()

class ClassificationLossCompute:
    "A Loss compute and train function for classification tasks."

    def __init__(self, lm_criterion, clf_criterion, lm_coef, opt=None):
        self.lm_criterion  = lm_criterion
        self.clf_criterion = clf_criterion
        self.lm_coef       = lm_coef
        self.opt           = opt

    def __call__(self, X, Y, M, clf_logits, lm_logits=None, only_return_losses=False):
        # Language modeling loss
        if lm_logits is not None:
            x_shifted = X[:, 1:, 0].contiguous().view(-1)
            M         = M.view(-1, M.size(-1))
            lm_losses = self.lm_criterion(lm_logits, x_shifted)
            lm_losses = lm_losses.view(X.size(0), X.size(-2) - 1)
            lm_losses = lm_losses * M[:, 1:]
            lm_losses = lm_losses.sum(1) / torch.sum(M[:, 1:], 1)
        # Classification loss
        clf_losses = self.clf_criterion(clf_logits, Y)
        if only_return_losses:
            return (clf_losses, lm_losses) if lm_logits is not None else clf_losses

        if self.lm_coef > 0 and lm_logits is not None:
            train_loss = clf_losses.sum() + self.lm_coef * lm_losses.sum()
        else:
            train_loss = clf_losses.sum()
        train_loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.zero_grad()
        return train_loss.item()

# TODO Implement a LossCompute class for similiraty tasks.

class HardDice(nn.Module):
    def __init__(self,
                 threshold=0.5,
                 deduct_intersection=False):
        super().__init__()
        self.threshold = threshold
        self.deduct_intersection = deduct_intersection
        
    def forward(self, outputs, targets):
        eps = 1e-10
        
        dice_target = (targets == 1).float()
        dice_output = F.softmax(outputs,dim=1)    
        hard_output = (dice_output > self.threshold).float()
        
        intersection = (hard_output * dice_target).sum()
        if self.deduct_intersection:
            union = hard_output.sum() + dice_target.sum() - intersection + eps
        else:
            union = hard_output.sum() + dice_target.sum() + eps
        
        hard_dice = (1+torch.log(2 * intersection / union))
        hard_dice = torch.clamp(hard_dice,0,1)
        return hard_dice