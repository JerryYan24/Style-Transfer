import torch
import torch.nn as nn


class Content_Loss(nn.Module):
    """
    计算内容损失
    """
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion(input, self.target)
        out = input.clone()

        return out

    def backward(self, retain_graph = True):
        self.loss.backward(retain_graph = retain_graph)
        
        return self.loss


class Style_Loss(nn.Module):
    """
    计算风格损失
    """
    def __init__(self, target):
        super(Style_Loss, self).__init__()
        self.target = target.detach()
        self.criterion = nn.MSELoss()
        
    def forward(self, input):
        G1 = self.Gram(input)
        G2 = self.Gram(self.target) 
        self.loss = self.criterion(G1, G2)
        out = input.clone()

        return out
        
    def backward(self):
        self.loss.backward(retain_graph=True)

        return self.loss   

    def Gram(self, input):
        """
        计算Gram矩阵
        """
        B, C, H, W = input.size()
        feature = input.view(B * C, H * W)
        gram = torch.mm(feature, feature.t())
        gram /= (B * C * H * W)

        return gram

