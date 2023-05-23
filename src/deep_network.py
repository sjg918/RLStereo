
import torch
import torch.nn as nn
import torch.nn.functional as F


# refer : https://github.com/Tianxiaomo/pytorch-YOLOv4
class Conv_Bn_Activation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, depthwise=False, dilation=1, bn=True, bias=False):
        super().__init__()
        pad = (kernel_size - 1) // 2

        self.conv = nn.ModuleList()
        if depthwise and dilation == 1:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, dilation=dilation, groups=in_channels, bias=bias))
        elif dilation == 1:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, dilation=dilation, bias=bias))
        else:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=dilation, dilation=dilation, bias=bias))
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels))
        if activation == "mish":
            self.conv.append(Mish())
        elif activation == "relu":
            self.conv.append(nn.ReLU(inplace=True))
        elif activation == "leaky":
            self.conv.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == "linear":
            pass
        else:
            print("activate error !!! {} {} {}".format(sys._getframe().f_code.co_filename,
                                                       sys._getframe().f_code.co_name, sys._getframe().f_lineno))

    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x


class TransPosedConv_Bn_Activation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, activation, depthwise=False, bn=True, bias=False):
        super().__init__()

        self.conv = nn.ModuleList()
        if depthwise:
            self.conv.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups=in_channels, bias=bias))
        else:
            self.conv.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=bias))

        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels))
        if activation == "mish":
            self.conv.append(Mish())
        elif activation == "relu":
            self.conv.append(nn.ReLU(inplace=True))
        elif activation == "leaky":
            self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == "linear":
            pass
        else:
            print("activate error !!! {} {} {}".format(sys._getframe().f_code.co_filename,
                                                       sys._getframe().f_code.co_name, sys._getframe().f_lineno))

    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x


class DCASPP_FENet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = Conv_Bn_Activation(3, 32, 3, 1, 'relu')
        self.conv2 = Conv_Bn_Activation(32, 32, 3, 2, 'relu') #1/2
        self.conv3 = Conv_Bn_Activation(32, 32, 3, 1, 'relu')
        self.conv4 = Conv_Bn_Activation(32, 32, 3, 2, 'relu') #1/4
        self.conv5 = Conv_Bn_Activation(32, 32, 3, 1, 'relu')

        self.conv6 = Conv_Bn_Activation(32, 64, 3, 2, 'relu') #1/8
        self.conv7 = Conv_Bn_Activation(64, 32, 3, 1, 'relu')
        self.conv8 = Conv_Bn_Activation(64, 32, 3, 1, 'relu', dilation=2)
        # conv9 = concatenate(6+7+8)
        self.conv10 = Conv_Bn_Activation(128, 64, 3, 1, 'relu')

        self.conv11 = Conv_Bn_Activation(64, 128, 3, 2, 'relu') #1/16
        self.conv12 = Conv_Bn_Activation(128, 32, 3, 1, 'relu')
        self.conv13 = Conv_Bn_Activation(128, 32, 3, 1, 'relu', dilation=2)
        # conv14 = concatenate(11+12+13)
        self.conv15 = Conv_Bn_Activation(192, 32, 3, 1, 'relu')
        self.conv16 = Conv_Bn_Activation(192, 32, 3, 1, 'relu', dilation=2)
        # conv17 = concatenate(11+12+13+15+16)
        self.conv18 = Conv_Bn_Activation(256, 128, 3, 1, 'relu')

        self.conv19 = Conv_Bn_Activation(128, 256, 3, 2, 'relu') #1/32
        self.conv20 = Conv_Bn_Activation(256, 32, 3, 1, 'relu')
        self.conv21 = Conv_Bn_Activation(256, 32, 3, 1, 'relu', dilation=2)
        # conv22 = concatenate(19+20+21)
        self.conv23 = Conv_Bn_Activation(320, 32, 3, 1, 'relu')
        self.conv24 = Conv_Bn_Activation(320, 32, 3, 1, 'relu', dilation=2)
        # conv25 = concatenate(19+20+21+23+24)
        self.conv26 = Conv_Bn_Activation(384, 32, 3, 1, 'relu')
        self.conv27 = Conv_Bn_Activation(384, 32, 3, 1, 'relu', dilation=2)
        # conv28 = concatenate(19+20+21+23+24+26+27)
        self.conv29 = Conv_Bn_Activation(448, 32, 3, 1, 'relu')
        self.conv30 = Conv_Bn_Activation(448, 32, 3, 1, 'relu', dilation=2)
        # conv31 = concatenate(19+20+21+23+24+26+27+29+30)
        self.conv32 = Conv_Bn_Activation(512, 256, 3, 1, 'relu')

        self.conv33 = TransPosedConv_Bn_Activation(256, 128, 3, 2, 1, 1, 'relu') # 1/16
        self.conv34 = Conv_Bn_Activation(128, 128, 3, 1, 'relu')

        # conv35 = concatenate(18+34)
        self.conv36 = TransPosedConv_Bn_Activation(256, 64, 3, 2, 1, 1, 'relu') # 1/8
        self.conv37 = Conv_Bn_Activation(64, 64, 3, 1, 'relu')

        # conv38 = concatenate(10+37)
        self.conv39 = TransPosedConv_Bn_Activation(128, 32, 3, 2, 1, 1, 'relu') # 1/4
        self.conv40 = Conv_Bn_Activation(32, 32, 3, 1, 'relu')
        self.conv41 = Conv_Bn_Activation(32, 32, 3, 1, 'linear', bn=False)

    def forward(self, x0):
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x6)
        x9 = torch.cat((x6, x7, x8), dim=1)
        x10 = self.conv10(x9)

        x11 = self.conv11(x10)
        x12 = self.conv12(x11)
        x13 = self.conv13(x11)
        x14 = torch.cat((x11, x12, x13), dim=1)
        x15 = self.conv15(x14)
        x16 = self.conv16(x14)
        x17 = torch.cat((x14, x15, x16), dim=1)
        x18 = self.conv18(x17)

        x19 = self.conv19(x18)
        x20 = self.conv20(x19)
        x21 = self.conv21(x19)
        x22 = torch.cat((x19, x20, x21), dim=1)
        x23 = self.conv23(x22)
        x24 = self.conv24(x22)
        x25 = torch.cat((x22, x23, x24), dim=1)
        x26 = self.conv26(x25)
        x27 = self.conv27(x25)
        x28 = torch.cat((x25, x26, x27), dim=1)
        x29 = self.conv29(x28)
        x30 = self.conv30(x28)
        x31 = torch.cat((x28, x29, x30), dim=1)
        x32 = self.conv32(x31)

        x33 = self.conv33(x32)
        x34 = self.conv34(x33)

        x35 = torch.cat((x18, x34), dim=1)
        x36 = self.conv36(x35)
        x37 = self.conv37(x36)

        x38 = torch.cat((x10, x37), dim=1)
        x39 = self.conv39(x38)
        x40 = self.conv40(x39)
        x41 = self.conv41(x40)
        return x41
    
# The size of the actor's state output is 1/4.
# And the actor's Network body uses 1-40 in Table 1.
# So the sizes of the inputs are not compatible !!!!!
# So I removed up to ~conv4.
class DCASPP_ANet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.initial_disp = cfg.initial_disp
        self.max_action = cfg.max_action
        self.max_iteration = cfg.max_iteration

        self.conv5 = Conv_Bn_Activation(65, 32, 3, 1, 'relu')

        self.conv6 = Conv_Bn_Activation(32, 64, 3, 2, 'relu') #1/8
        self.conv7 = Conv_Bn_Activation(64, 32, 3, 1, 'relu')
        self.conv8 = Conv_Bn_Activation(64, 32, 3, 1, 'relu', dilation=2)
        # conv9 = concatenate(6+7+8)
        self.conv10 = Conv_Bn_Activation(128, 64, 3, 1, 'relu')

        self.conv11 = Conv_Bn_Activation(64, 128, 3, 2, 'relu') #1/16
        self.conv12 = Conv_Bn_Activation(128, 32, 3, 1, 'relu')
        self.conv13 = Conv_Bn_Activation(128, 32, 3, 1, 'relu', dilation=2)
        # conv14 = concatenate(11+12+13)
        self.conv15 = Conv_Bn_Activation(192, 32, 3, 1, 'relu')
        self.conv16 = Conv_Bn_Activation(192, 32, 3, 1, 'relu', dilation=2)
        # conv17 = concatenate(11+12+13+15+16)
        self.conv18 = Conv_Bn_Activation(256, 128, 3, 1, 'relu')

        self.conv19 = Conv_Bn_Activation(128, 256, 3, 2, 'relu') #1/32
        self.conv20 = Conv_Bn_Activation(256, 32, 3, 1, 'relu')
        self.conv21 = Conv_Bn_Activation(256, 32, 3, 1, 'relu', dilation=2)
        # conv22 = concatenate(19+20+21)
        self.conv23 = Conv_Bn_Activation(320, 32, 3, 1, 'relu')
        self.conv24 = Conv_Bn_Activation(320, 32, 3, 1, 'relu', dilation=2)
        # conv25 = concatenate(19+20+21+23+24)
        self.conv26 = Conv_Bn_Activation(384, 32, 3, 1, 'relu')
        self.conv27 = Conv_Bn_Activation(384, 32, 3, 1, 'relu', dilation=2)
        # conv28 = concatenate(19+20+21+23+24+26+27)
        self.conv29 = Conv_Bn_Activation(448, 32, 3, 1, 'relu')
        self.conv30 = Conv_Bn_Activation(448, 32, 3, 1, 'relu', dilation=2)
        # conv31 = concatenate(19+20+21+23+24+26+27+29+30)
        self.conv32 = Conv_Bn_Activation(512, 256, 3, 1, 'relu')

        self.conv33 = TransPosedConv_Bn_Activation(256, 128, 3, 2, 1, 1, 'relu') # 1/16
        self.conv34 = Conv_Bn_Activation(128, 128, 3, 1, 'relu')

        # conv35 = concatenate(18+34)
        self.conv36 = TransPosedConv_Bn_Activation(256, 64, 3, 2, 1, 1, 'relu') # 1/8
        self.conv37 = Conv_Bn_Activation(64, 64, 3, 1, 'relu')

        # conv38 = concatenate(10+37)
        self.conv39 = TransPosedConv_Bn_Activation(128, 32, 3, 2, 1, 1, 'relu') # 1/4
        self.conv40 = Conv_Bn_Activation(32, 32, 3, 1, 'relu')

        self.conv41 = Conv_Bn_Activation(32, 1, 1, 1, 'linear', bn=False)

    # refer : https://github.com/xy-guo/GwcNet
    def action_regression(self, x):
        assert len(x.shape) == 4
        disp_values = torch.arange(0, self.max_action + 1, dtype=x.dtype, device=x.device)
        disp_values = disp_values.view(1, self.max_action + 1, 1, 1)
        return torch.sum(x * disp_values, 1, keepdim=False)

    def forward(self, left_fea, warp_fea, cur_disp_map):
        B, _, H, W = left_fea.shape
        # normalize dispmap
        cur_disp_map = cur_disp_map / (self.initial_disp + self.max_action * self.max_iteration)
        x0 = torch.cat((left_fea, warp_fea, cur_disp_map.unsqueeze(1)), dim=1)

        x5 = self.conv5(x0)

        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x6)
        x9 = torch.cat((x6, x7, x8), dim=1)
        x10 = self.conv10(x9)

        x11 = self.conv11(x10)
        x12 = self.conv12(x11)
        x13 = self.conv13(x11)
        x14 = torch.cat((x11, x12, x13), dim=1)
        x15 = self.conv15(x14)
        x16 = self.conv16(x14)
        x17 = torch.cat((x14, x15, x16), dim=1)
        x18 = self.conv18(x17)

        x19 = self.conv19(x18)
        x20 = self.conv20(x19)
        x21 = self.conv21(x19)
        x22 = torch.cat((x19, x20, x21), dim=1)
        x23 = self.conv23(x22)
        x24 = self.conv24(x22)
        x25 = torch.cat((x22, x23, x24), dim=1)
        x26 = self.conv26(x25)
        x27 = self.conv27(x25)
        x28 = torch.cat((x25, x26, x27), dim=1)
        x29 = self.conv29(x28)
        x30 = self.conv30(x28)
        x31 = torch.cat((x28, x29, x30), dim=1)
        x32 = self.conv32(x31)

        x33 = self.conv33(x32)
        x34 = self.conv34(x33)

        x35 = torch.cat((x18, x34), dim=1)
        x36 = self.conv36(x35)
        x37 = self.conv37(x36)

        x38 = torch.cat((x10, x37), dim=1)
        x39 = self.conv39(x38)
        x40 = self.conv40(x39)

        x41 = self.conv41(x40).squeeze(1)
        x41 = torch.clamp(x41, -self.max_action, self.max_action)
        return x41
    

# An edge-aware refinement network is not implemented in the original paper.
# I implement the refinement network described in the paper below.
# StereoNet: Guided Hierarchical Refinement for Real-Time Edge-Aware Depth Prediction
# refer : https://arxiv.org/pdf/1807.08865.pdf
class GHRNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.initial_disp = cfg.initial_disp
        self.max_action = cfg.max_action
        self.max_iteration = cfg.max_iteration
        self.max_disp = cfg.max_disp

        self.conv1 = Conv_Bn_Activation(4, 32, 3, 1, 'leaky')

        self.conv2 = Conv_Bn_Activation(32, 32, 3, 1, 'leaky')
        self.conv3 = Conv_Bn_Activation(32, 32, 3, 1, 'leaky', dilation=2)
        self.conv4 = Conv_Bn_Activation(32, 32, 3, 1, 'leaky', dilation=4)
        self.conv5 = Conv_Bn_Activation(32, 32, 3, 1, 'leaky', dilation=8)
        self.conv6 = Conv_Bn_Activation(32, 32, 3, 1, 'leaky')
        self.conv7 = Conv_Bn_Activation(32, 32, 3, 1, 'leaky')
        
        self.conv8 = Conv_Bn_Activation(32, 1, 1, 1, 'linear', bn=False)

    # refer : https://github.com/xy-guo/GwcNet
    def action_regression(self, x):
        assert len(x.shape) == 4
        disp_values = torch.arange(0, self.max_disp + 1, dtype=x.dtype, device=x.device)
        disp_values = disp_values.view(1, self.max_disp + 1, 1, 1)
        return torch.sum(x * disp_values, 1, keepdim=False)

    def forward(self, x0, pred_disp_map__):
        B, H, W = pred_disp_map__.shape 
        # resize
        pred_disp_map = F.interpolate(pred_disp_map__.unsqueeze(1), (H*4, W*4), mode='bilinear', align_corners=True)
        # normalize
        pred_disp_map = pred_disp_map / self.max_disp * 2 - 1

        x1 = self.conv1(torch.cat((x0, pred_disp_map), dim=1))
        x2 = self.conv2(x1) + x1
        x3 = self.conv3(x2) + x2
        x4 = self.conv4(x3) + x3
        x5 = self.conv5(x4) + x4
        x6 = self.conv6(x5) + x5
        x7 = self.conv7(x6) + x6

        x8 = self.conv8(x7)
        pred_disp_map = pred_disp_map.squeeze(1) + x8.squeeze(1)
        pred_disp_map = torch.clamp(pred_disp_map, 0, self.max_disp)
        return pred_disp_map
    
