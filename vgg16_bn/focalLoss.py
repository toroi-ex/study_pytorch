import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float,int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=1, gamma=2, reduction: str = 'mean'):
#         super().__init__()
#         if reduction not in ['mean', 'none', 'sum']:
#             raise NotImplementedError('Reduction {} not implemented.'.format(reduction))
#         self.reduction = reduction
#         self.alpha = alpha
#         self.gamma = gamma
#
#     def forward(self, x, target):
#         p_t = torch.where(target == 1, x, 1-x)
#         fl = - 1 * (1 - p_t) ** self.gamma * torch.log(p_t)
#         fl = torch.where(target == 1, fl * self.alpha, fl)
#         return self._reduce(fl)
#
#     def _reduce(self, x):
#         if self.reduction == 'mean':
#             return x.mean()
#         elif self.reduction == 'sum':
#             return x.sum()
#         else:
#             return x

# class FocalLoss(nn.modules.loss._WeightedLoss):
#     def __init__(self, weight=None, gamma=2,reduction='mean'):
#         super(FocalLoss, self).__init__(weight,reduction=reduction)
#         self.gamma = gamma
#         self.weight = weight #weight parameter will act as the alpha parameter to balance class weights
#
#     def forward(self, input, target):
#
#         ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
#         pt = torch.exp(-ce_loss)
#         focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
#         return focal_loss
