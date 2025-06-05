from collections import OrderedDict
from arc import AdaptiveRotatedConv2d, RountingFunction    
import torch
from torch import nn
from torch.nn import functional as F


class Transformer_Fusion(nn.Module):
    def __init__(self, dim=768, nhead=8, num_layers=1):
        super(Transformer_Fusion, self).__init__()
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=nhead)
        self.transformer_model = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

    def forward(self, v, lan):
        W, H = v.shape[2], v.shape[3]
        v = v.view(v.shape[0], v.shape[1], -1)
        v = v.permute(2, 0, 1)
        l = lan.permute(2, 0, 1)
        v = self.transformer_model(v, l)
        v = v.permute(1, 2, 0)
        v = v.view(v.shape[0], v.shape[1], W, H)
        return v


class TCMD(nn.Module):
    def __init__(self, c4_dims, factor=2):
        super(TCMD, self).__init__()

        lan_size = 768
        hidden_size = lan_size
        c4_size = c4_dims
        c3_size = c4_dims//(factor**1)
        c2_size = c4_dims//(factor**2)
        c1_size = c4_dims//(factor**3)

        self.adpool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv1_4 = nn.Conv2d(c4_size+c3_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_4 = nn.BatchNorm2d(hidden_size)
        self.relu1_4 = nn.ReLU()
        routing_function1 = RountingFunction(in_channels=hidden_size, kernel_number=1)
        self.conv2_4 = AdaptiveRotatedConv2d(in_channels=hidden_size, out_channels=hidden_size,
                                                     kernel_size=3, padding=1, rounting_func=routing_function1, bias=False, kernel_number=1)
        self.bn2_4 = nn.BatchNorm2d(hidden_size)
        self.relu2_4 = nn.ReLU()

        self.transformer_fusion1 = Transformer_Fusion(dim=768, nhead=8, num_layers=1)

        self.conv1_3 = nn.Conv2d(hidden_size + c2_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_3 = nn.BatchNorm2d(hidden_size)
        self.relu1_3 = nn.ReLU()
        routing_function2 = RountingFunction(in_channels=hidden_size, kernel_number=1)
        self.conv2_3 = AdaptiveRotatedConv2d(in_channels=hidden_size, out_channels=hidden_size,
                                             kernel_size=3, padding=1, rounting_func=routing_function2, bias=False, kernel_number=1)
        self.bn2_3 = nn.BatchNorm2d(hidden_size)
        self.relu2_3 = nn.ReLU()
        self.transformer_fusion2 = Transformer_Fusion(dim=768, nhead=8, num_layers=1)


        self.conv1_2 = nn.Conv2d(hidden_size + c1_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(hidden_size)
        self.relu1_2 = nn.ReLU()
        routing_function3 = RountingFunction(in_channels=hidden_size, kernel_number=1)
        self.conv2_2 = AdaptiveRotatedConv2d(in_channels=hidden_size, out_channels=hidden_size,
                                             kernel_size=3, padding=1, rounting_func=routing_function3, bias=False, kernel_number=1)
        self.bn2_2 = nn.BatchNorm2d(hidden_size)
        self.relu2_2 = nn.ReLU()

        self.conv1_1 = nn.Conv2d(hidden_size, 2, 1)


    def forward(self, lan, x_c4, x_c3, x_c2, x_c1):
        # fuse Y4 and Y3
        if x_c4.size(-2) < x_c3.size(-2) or x_c4.size(-1) < x_c3.size(-1):
            x_c4 = F.interpolate(input=x_c4, size=(x_c3.size(-2), x_c3.size(-1)), mode='bilinear', align_corners=True)
        x = torch.cat([x_c4, x_c3], dim=1)
        x = self.conv1_4(x)
        x = self.bn1_4(x)
        x = self.relu1_4(x)
        x = self.conv2_4(x)
        x = self.bn2_4(x)
        x = self.relu2_4(x) 


        x = self.transformer_fusion1(x, lan)

        # fuse top-down features and Y2 features and pre1
        if x.size(-2) < x_c2.size(-2) or x.size(-1) < x_c2.size(-1):
            x = F.interpolate(input=x, size=(x_c2.size(-2), x_c2.size(-1)), mode='bilinear', align_corners=True)
        x = torch.cat([x, x_c2], dim=1)
        x = self.conv1_3(x)
        x = self.bn1_3(x)
        x = self.relu1_3(x)
        x = self.conv2_3(x)
        x = self.bn2_3(x)
        x = self.relu2_3(x) 



        x = self.transformer_fusion2(x, lan)

        # fuse top-down features and Y1 features
        if x.size(-2) < x_c1.size(-2) or x.size(-1) < x_c1.size(-1):
            x = F.interpolate(input=x, size=(x_c1.size(-2), x_c1.size(-1)), mode='bilinear', align_corners=True)
        x = torch.cat([x, x_c1], dim=1)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu1_2(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.relu2_2(x) 

        return self.conv1_1(x)
