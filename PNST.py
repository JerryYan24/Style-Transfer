import torch.nn as nn
import torch
import torchvision.models as models

import Loss, utills


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VGG(nn.Module):
    """
    VGG网络
    """
    def __init__(self, style_img, content_img):
        super(VGG, self).__init__()
        self.model = models.vgg19(pretrained=True).to(device)
        self.style_img = style_img
        self.content_img = content_img

    def forward(self, x):
        model = nn.Sequential()
        stylelosses = []
        contentlosses = []

        """layer1"""
        model.add_module("conv_1", self.model.features[0])
        style_target = model(self.style_img)
        style_loss_1 = Loss.Style_Loss(style_target)
        model.add_module('style_loss_1', style_loss_1)
        stylelosses.append(style_loss_1)
        model.add_module('relu1', self.model.features[1])

        """layer2"""
        model.add_module('conv2', self.model.features[2])
        model.to(device)
        style_target = model(self.style_img).detach()
        style_loss_2= Loss.Style_Loss(style_target)
        model.add_module('style_loss_2', style_loss_2)
        stylelosses.append(style_loss_2)
        model.add_module('relu2', self.model.features[3])
        model.add_module('pool_2', self.model.features[4])
        
        """layer3"""
        model.add_module("conv_3", self.model.features[5])
        model.to(device)
        style_target = model(self.style_img).detach()
        style_loss_3 = Loss.Style_Loss(style_target)
        model.add_module('style_loss_3', style_loss_3)
        stylelosses.append(style_loss_3)
        model.add_module('relu3', self.model.features[6])
        
        """layer4"""
        model.add_module('conv4', self.model.features[7])
        model.to(device)
        content_target = model(self.content_img).detach()
        content_loss = Loss.Content_Loss(content_target)
        contentlosses.append(content_loss)
        model.add_module('content_loss_4', content_loss)
        model.to(device)
        style_target = model(self.style_img).detach()
        style_loss_4 = Loss.Style_Loss(style_target)
        model.add_module('style_loss_4', style_loss_4)
        stylelosses.append(style_loss_4)
        model.add_module('relu4', self.model.features[8])
        model.add_module('pool_4', self.model.features[9])
        
        """layer5"""
        model.add_module("conv_5", self.model.features[10])
        model.to(device)
        style_target = model(self.style_img).detach()
        style_loss_5 = Loss.Style_Loss(style_target)
        model.add_module('style_loss_5', style_loss_5)
        stylelosses.append(style_loss_5)

        x = model(x)

        return x, stylelosses, contentlosses


def main():
    img_size = 512
    style_img_path = r"C:\Users\X\Downloads\pst\ref\1.jpg"
    content_img_path = r"C:\Users\X\Downloads\pst\ref\2.jpg"
    
    style_img = utills.load_img(style_img_path, img_size)
    content_img = utills.load_img(content_img_path, img_size)

    style_weight = 100
    content_weight = 1
    cnn = VGG(style_img, content_img)


if __name__=="__main__":
    main()
