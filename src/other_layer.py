import torch
import torch.nn as nn


class ConditionalLayerNorm(nn.Module):
    def __init__(self,
                 normalized_shape,
                 cond_shape,
                 eps=1e-12):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.Tensor(normalized_shape))
        self.bias = nn.Parameter(torch.Tensor(normalized_shape))
        self.weight_dense = nn.Linear(cond_shape, normalized_shape, bias=False)
        self.bias_dense = nn.Linear(cond_shape, normalized_shape, bias=False)
        self.reset_weight_and_bias()

    def reset_weight_and_bias(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        # 此处全0初始化的作用是在训练开始阶段不让 conditional layer norm 起作用
        nn.init.zeros_(self.weight_dense.weight)
        nn.init.zeros_(self.bias_dense.weight)

    def forward(self, inputs, cond=None):
        '''
            inputs: [b, s, h]
            cond: [b, h]
        '''
        assert cond is not None, 'Conditional tensor need to input when use conditional layer norm'

        cond = torch.unsqueeze(cond, 1)  # (b, 1, h)
        weight = self.weight_dense(cond) + self.weight  # (b, 1, h)
        bias = self.bias_dense(cond) + self.bias  # (b, 1, h)

        mu = torch.mean(inputs, dim=-1, keepdim=True)  # （b, s, 1）
        outputs = inputs - mu  # (b, s, h)
        var = torch.mean(outputs ** 2, dim=-1, keepdim=True)
        std = torch.sqrt(var + self.eps)  # (b, s, 1)
        outputs = weight * (inputs - mu) / (std + self.eps) + bias
        return outputs


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output, target):
        c = output.size()[-1]
        log_pred = torch.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_pred.sum()
        else:
            loss = -log_pred.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()

        return loss * self.eps / c + (1 - self.eps) * torch.nn.functional.nll_loss(log_pred, target,
                                                                                   reduction=self.reduction,

                                                                                   ignore_index=self.ignore_index)
class FocalLoss(nn.Module):
    """Multi-class Focal loss implementation"""
    def __init__(self, gamma=2, weight=None, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        log_pt = torch.log_softmax(input, dim=1)
        pt = torch.exp(log_pt)
        log_pt = (1 - pt) ** self.gamma * log_pt
        loss = torch.nn.functional.nll_loss(log_pt, target, self.weight, reduction=self.reduction, ignore_index=self.ignore_index)
        return loss
