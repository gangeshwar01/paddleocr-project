import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class DBHead(nn.Module):
    def __init__(self, in_channels, k=50):
        super().__init__()
        self.k = k
        self.binarize = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, 2),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, 2, 2),
            nn.Sigmoid()
        )
        self.thresh = self.binarize # Shared weights for threshold map

    def forward(self, x):
        prob_map = self.binarize(x)
        thresh_map = self.thresh(x)
        
        # Differentiable Binarization
        binary_map = 1 / (1 + torch.exp(-self.k * (prob_map - thresh_map)))
        
        return prob_map, thresh_map, binary_map

class DBNET_FPN(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        # Simplified FPN
        self.out_channels = out_channels
        self.in2_conv = nn.Conv2d(in_channels[0], out_channels, 1)
        self.in3_conv = nn.Conv2d(in_channels[1], out_channels, 1)
        self.in4_conv = nn.Conv2d(in_channels[2], out_channels, 1)
        self.in5_conv = nn.Conv2d(in_channels[3], out_channels, 1)

        self.p5_conv = nn.Conv2d(out_channels, out_channels // 4, 3, padding=1)
        self.p4_conv = nn.Conv2d(out_channels, out_channels // 4, 3, padding=1)
        self.p3_conv = nn.Conv2d(out_channels, out_channels // 4, 3, padding=1)
        self.p2_conv = nn.Conv2d(out_channels, out_channels // 4, 3, padding=1)
        
    def forward(self, features):
        c2, c3, c4, c5 = features
        
        # Latéral connections
        p5 = self.in5_conv(c5)
        p4 = self.in4_conv(c4)
        p3 = self.in3_conv(c3)
        p2 = self.in2_conv(c2)
        
        # Top-down pathway
        p4 = F.interpolate(p5, scale_factor=2) + p4
        p3 = F.interpolate(p4, scale_factor=2) + p3
        p2 = F.interpolate(p3, scale_factor=2) + p2
        
        # Fuse and upsample
        p5 = self.p5_conv(p5)
        p4 = self.p4_conv(p4)
        p3 = self.p3_conv(p3)
        p2 = self.p2_conv(p2)

        p5 = F.interpolate(p5, scale_factor=8)
        p4 = F.interpolate(p4, scale_factor=4)
        p3 = F.interpolate(p3, scale_factor=2)
        
        return torch.cat([p2, p3, p4, p5], dim=1)


class DBNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = resnet50(weights=weights)
        
        # Extract features from intermediate layers
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Define feature channels for ResNet50
        backbone_channels = [256, 512, 1024, 2048]
        
        self.fpn = DBNET_FPN(backbone_channels, out_channels=256)
        self.head = DBHead(in_channels=256)

    def forward(self, x):
        # Backbone features
        c2 = self.backbone.layer1(self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x))))
        c3 = self.backbone.layer2(c2)
        c4 = self.backbone.layer3(c3)
        c5 = self.backbone.layer4(c4)
        
        features = self.fpn([c2, c3, c4, c5])
        prob_map, thresh_map, binary_map = self.head(features)
        
        # Squeeze to remove channel dim of 1
        return prob_map.squeeze(1), thresh_map.squeeze(1), binary_map.squeeze(1)
