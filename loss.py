
import torch
import torch.nn as nn


class cross_entropy_weighted(nn.Module):
    def __init__(self,size_average=True,thresh=0.75):
        super(cross_entropy_weighted, self).__init__()
        self.lossf=nn.CrossEntropyLoss(reduction='none')
        self.size_average = size_average
        self.thresh=thresh

    def forward(self, input, target):
        logpt=self.lossf(input,target)
        pt=torch.exp(-logpt)
        thresh_mask=torch.lt(pt,self.thresh).to(torch.float)
        loss=logpt*thresh_mask
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum() 
