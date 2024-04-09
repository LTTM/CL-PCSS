import torch, gc
import numpy as np
from torch import Tensor
from typing import Optional
import torch.nn

################## UTILITY FUNCTIONS #################

#function to map logits of micro classes to macro ones
def micro2macro(logits, map_macro, idx, classes):
    mat = torch.zeros((classes, classes), device='cuda')
    for key in map_macro:
        mat[idx[map_macro[key]]][key-1] = 1
    macro_tens = torch.mm(logits, mat)
    return macro_tens.detach()

########## FAIRNESS MEASURES ###########

#fairness measure to properly allocate resources for each class
def fairness(labels, logits, c):
    p_vec = []
    labels = labels.reshape(labels.shape[-1]*labels.shape[-2]*labels.shape[-3])
    logits = logits.squeeze(dim=0)
    logits = logits.reshape((c, logits.shape[-1])).T
    for i in range(0, c):
      #gather logits per that class
      cl = (labels == i)
      w = torch.where(cl)
      if len(w[0])==0:
          pp = torch.tensor(0, device='cuda')
      else:
          w = w[0]
          #print(logits.shape)
          ll = logits[w] #torch.gather(logits, 0, index=w)
          p = torch.mean(ll, axis=0)
          pp = p[i]
      p_vec.append(pp)
    xx = torch.stack(p_vec)
    numerator = torch.pow(torch.sum(xx), 2)
    denominator = torch.multiply(torch.tensor(c), torch.sum(torch.pow(xx, 2)))
    fair = torch.divide(numerator, denominator)
    return fair.detach()

def fair(labels, logits, map, c):
    p_vec = []
    labels = labels.reshape(labels.shape[-1]*labels.shape[-2])
    logits = logits.swapaxes(0, 1)
    logits = logits.reshape((c, logits.shape[-1] * logits.shape[-2])).T
    #u, count = torch.unique(labels, return_counts=True)

    for i in range(0, c):
      #gather logits per that class
      cl = (labels == i)
      w = torch.where(cl)
      if len(w[0])==0:
          pp = torch.tensor(0, device='cuda')
      else:
          w = w[0]
          ll = logits[w]
          p = torch.nanmean(ll, axis=0)
          pp = p[i]
      p_vec.append(pp)
    xx = torch.stack(p_vec)
    idx = torch.from_numpy(np.array([v for v in map.values()]))[1:]
    ii = torch.nn.functional.one_hot(idx.to(torch.int64)).type(torch.cuda.FloatTensor).detach()
    numerator = torch.pow(torch.matmul(ii[xx!=0].T, xx[xx!=0]), 2)
    cc = torch.sum(ii[xx!=0], axis=0)
    denominator = torch.multiply(cc, torch.torch.matmul(ii[xx!=0].T, (torch.pow(xx[xx!=0], 2))))
    fair = torch.divide(numerator, denominator)
    return torch.nanmean(fair).detach()

"""
#fairness measure to properly allocate resources
def fair(logits, map, idx, classes):
    logits = torch.swapaxes(logits, 0, 1)
    depth, B, a, b, c = logits.shape
    logits = torch.reshape(logits, (depth, B * a * b * c))
    logits = torch.swapaxes(logits, 0, 1)
    n = torch.zeros(classes, dtype=torch.int64, device='cuda')
    u, counts = torch.unique(torch.from_numpy(np.array(list(map.values()))), return_counts=True)
    n[idx] = counts.to('cuda')
    numerator = torch.pow(micro2macro(logits, map, idx, classes), 2)
    denominator = torch.mm(micro2macro(torch.pow(logits, 2), map, idx, classes), torch.reshape(n, shape=(classes,1)).float())
    fair = torch.divide(torch.sum(numerator, axis=1), torch.sum(denominator, axis=1))
    return fair.nanmean().detach()
"""

########### PROTOTYPES BUILDING ##################

def build_prototypes(features, labels, prots, K, macro_labels, macro_prots, M):
    # function to build the prototypes for the classes
    # feats: hidden features
    # labels: labels for the whole batch

    B = features.shape[0]  # batch size
    depth = features.shape[1]  # length of feature vectors
    a, b, c = features.shape[2:]
    features = torch.swapaxes(features,0,1)

    mxx = torch.reshape(macro_labels, (B * a * b * c,)).long()
    xx = torch.reshape(labels, (B * a * b * c,)).long()
    yy = torch.reshape(features, (depth, B * a * b * c))
    yy = torch.swapaxes(yy, 0, 1)
    u, idx, c = torch.unique(xx, return_counts=True, return_inverse=True)
    #"""
    new_K = torch.zeros_like(K)
    new_K[u] = c.float().cuda()
    new_prots = torch.zeros_like(prots)
    for cl in u:
        new_prots[cl] = torch.mean(yy[xx == cl], dim=0)
        prots[cl] = ((prots[cl] * K[cl]) + (new_prots[cl] * new_K[cl]))/(K[cl]+new_K[cl])
    K = K + new_K
    """
    ii = torch.nn.functional.one_hot(idx).type(torch.cuda.FloatTensor).detach()
    K_old = K.clone().detach()
    K[u] += c.cuda() #update K
    # compute protos
    p = torch.mm(torch.diag(K_old), prots)
    p[u] = p[u] + torch.mm(torch.diag(c.type(torch.cuda.FloatTensor)), torch.mm(ii.T, yy).detach())
    new_prots = torch.mm(torch.diag(1/K[u]), p[u])
    prots[u] = new_prots #update protos
    """
    #compute macro protos (map labels and repreat procedure)
    u, idx, c = torch.unique(mxx, return_counts=True, return_inverse=True)
    """
    ii = torch.nn.functional.one_hot(idx).type(torch.cuda.FloatTensor).detach()
    M_old = M.clone().detach()
    M[u] += c.cuda()  # update M
    p = torch.mm(torch.diag(M_old), macro_prots)
    p[u] = p[u] + torch.mm(torch.diag(c.type(torch.cuda.FloatTensor)), torch.mm(ii.T, yy).detach())
    new_macro_prots = torch.mm(torch.diag(1/M[u]), p[u])
    macro_prots[u] = new_macro_prots #update macro protos
    """
    new_M = torch.zeros_like(M)
    new_M[u] = c.float().cuda()
    new_macro_prots = torch.zeros_like(macro_prots)
    for cl in u:
        new_macro_prots[cl] = torch.mean(yy[mxx == cl], dim=0)
        macro_prots[cl] = ((macro_prots[cl] * M[cl]) + (new_macro_prots[cl] * new_M[cl]))/(M[cl]+new_M[cl])
    M = M + new_M
    """
    for a in aa:
        n = torch.where(xx == a)  # find which vectors belong to my class
        K[a] = K[a] + len(n)
        prots[a] = ((prots[a] * K[a]) + torch.sum(yy[n], axis=0)) / K[a]
        b = lookup_table[int(a.detach().cpu().numpy())]
        M[b] = M[b] + len(n)
        macro_prots[b] = ((macro_prots[b] * M[b]) + torch.sum(yy[n], axis=0)) / M[b]
    """
    return prots.detach(), K.detach(), macro_prots.detach(), M.detach()
        
########################### LOSS FUNCTIONS #####################################

class FairnessLoss(torch.nn.modules.loss._WeightedLoss):

    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(self, weight: Optional[Tensor] = None, alpha=10., classes=20, size_average=None, ignore_index: int = -1,
                 reduce=None, reduction: str = 'mean') -> None:
        super(FairnessLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.classes = classes

        def forward(self, input: Tensor, target: Tensor) -> Tensor:
            # input: logits
            # target: labels
            # alpha: param fairness
            # my_loss = F.nll_loss(input, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)
            my_loss = self.alpha * (1 - fairness(target, input, self.classes))
            return my_loss

class FairLoss(torch.nn.modules.loss._WeightedLoss):
    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(self, weight: Optional[Tensor] = None, alpha=10., classes=20, size_average=None,
                 ignore_index: int = -1, mapping = None, idx = None,
                 reduce=None, reduction: str = 'mean') -> None:
        super(FairLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.classes = classes
        self.mapping = mapping
        self.idx = idx

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        #input: logits
        #alpha: param fairness
        #my_loss = F.nll_loss(input, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)
        my_loss = self.alpha * (1 - fair(target, input, self.mapping, self.classes))
        return my_loss

class ProtoLoss(torch.nn.modules.loss._WeightedLoss):

    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(self, weight: Optional[Tensor] = None, gamma=1e-4, classes=20, size_average=None, ignore_index: int = -1,
                 reduce=None, reduction: str = 'mean') -> None:
        super(ProtoLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.gamma = gamma
        self.classes = classes

    def forward(self, features: Tensor, labels: Tensor, prototypes: Tensor) -> Tensor:
        #feats
        B = features.shape[0]  # batch size
        depth = features.shape[1]  # length of feature vectors
        a, b, c = features.shape[2:]
        features = torch.swapaxes(features, 0, 1)
        xx = torch.reshape(labels, (B * a * b * c,)).long()
        yy = torch.reshape(features, (depth, B * a * b * c))
        yy = torch.swapaxes(yy, 0, 1)
        mask = xx!=0
        xxx = xx[mask]
        yyy = yy[mask]
        zz = prototypes[xxx] #.detach()
        # compute loss
        #tau = 1
        #ex = -torch.log((((yyy * zz)/tau)/torch.sum((yyy * zz)/tau)))
        #ex[ex == torch.inf] = 0
        #my_loss = self.gamma * torch.nanmean(ex)
        #my_loss = self.gamma * torch.nanmean(torch.norm(yyy - zz, dim=0,))
        my_loss = self.gamma * torch.nanmean(torch.linalg.vector_norm(yyy - zz, dim=0, ord=2))
        return my_loss

class PrototypeLoss(torch.nn.modules.loss._WeightedLoss):

    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(self, weight: Optional[Tensor] = None, gamma=1e-4, classes=20, size_average=None, ignore_index: int = -1,
                 reduce=None, reduction: str = 'mean') -> None:
        super(ProtoLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.gamma = gamma
        self.classes = classes

    def forward(self, features: Tensor, labels: Tensor, prototypes: Tensor) -> Tensor:
        #feats
        B = features.shape[0]  # batch size
        depth = features.shape[1]  # length of feature vectors
        a, b, c = features.shape[2:]
        features = torch.swapaxes(features, 0, 1)
        xx = torch.reshape(labels, (B * a * b * c,)).long()
        yy = torch.reshape(features, (depth, B * a * b * c))
        yy = torch.swapaxes(yy, 0, 1)
        zz = prototypes[xx] #.detach()
        # compute loss
        my_loss = self.gamma * torch.norm(yy/torch.max(yy) - zz/torch.max(yy))
        return my_loss