import torch
from torch import nn
import timm
import math
from torch.nn import functional as F
import pdb


class Conv2d_Hori_Veri_Cross(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_Hori_Veri_Cross, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):

        [C_out, C_in, H_k, W_k] = self.conv.weight.shape
        tensor_zeros = torch.FloatTensor(C_out, C_in, 1).fill_(0).cuda()
        conv_weight = torch.cat((tensor_zeros, self.conv.weight[:, :, :, 0], tensor_zeros, self.conv.weight[:, :, :, 1],
                                 self.conv.weight[:, :, :, 2], self.conv.weight[:, :, :, 3], tensor_zeros,
                                 self.conv.weight[:, :, :, 4], tensor_zeros), 2)
        conv_weight = conv_weight.contiguous().view(C_out, C_in, 3, 3)

        out_normal = F.conv2d(input=x, weight=conv_weight, bias=self.conv.bias, stride=self.conv.stride,
                              padding=self.conv.padding)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # pdb.set_trace()
            [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0,
                                groups=self.conv.groups)

            return out_normal - self.theta * out_diff


class Conv2d_Diag_Cross(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_Diag_Cross, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):

        [C_out, C_in, H_k, W_k] = self.conv.weight.shape
        tensor_zeros = torch.FloatTensor(C_out, C_in, 1).fill_(0).cuda()
        conv_weight = torch.cat((self.conv.weight[:, :, :, 0], tensor_zeros, self.conv.weight[:, :, :, 1], tensor_zeros,
                                 self.conv.weight[:, :, :, 2], tensor_zeros, self.conv.weight[:, :, :, 3], tensor_zeros,
                                 self.conv.weight[:, :, :, 4]), 2)
        conv_weight = conv_weight.contiguous().view(C_out, C_in, 3, 3)

        out_normal = F.conv2d(input=x, weight=conv_weight, bias=self.conv.bias, stride=self.conv.stride,
                              padding=self.conv.padding)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # pdb.set_trace()
            [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0,
                                groups=self.conv.groups)

            return out_normal - self.theta * out_diff


class DC_Conv(nn.Module):
    def __init__(self, conv1, conv2):
        self.conv1 = conv1
        self.conv2 = conv2




    def forward(self, x):
        return (self.conv1(x) + self.conv2(x))/2




class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff
def forward_block(self, x):
    x = x + self.drop_path1(self.attn(self.norm1(x))) + self.drop_path1(self.adapter_attn(self.norm1(x))) * self.s
    x = x + self.drop_path2(self.mlp(self.norm2(x))) + self.drop_path2(self.adapter_mlp(self.norm2(x))) * self.s
    return x

def forward_block_attn(self, x):
    x = x + self.drop_path1(self.attn(self.norm1(x))) + self.drop_path1(self.adapter_attn(self.norm1(x))) * self.s
    x = x + self.drop_path2(self.mlp(self.norm2(x)))
    return x

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Convpass(nn.Module):
    def __init__(self, dim=8, xavier_init=False, conv_type='conv'):
        super().__init__()


        if conv_type=='conv':
            self.adapter_conv = nn.Conv2d(dim, dim, 3, 1, 1)
            
            if xavier_init:
                nn.init.xavier_uniform_(self.adapter_conv.weight)
                
            else:
                nn.init.zeros_(self.adapter_conv.weight)
                self.adapter_conv.conv.data[:, :, 1, 1] += torch.eye(8, dtype=torch.float)
    
            nn.init.zeros_(self.adapter_conv.bias)

            self.adapter_down = nn.Linear(768, dim)  # equivalent to 1 * 1 Conv
            self.adapter_up = nn.Linear(dim, 768)  # equivalent to 1 * 1 Conv
            nn.init.xavier_uniform_(self.adapter_down.weight)
            nn.init.zeros_(self.adapter_down.bias)
            nn.init.zeros_(self.adapter_up.weight)
            nn.init.zeros_(self.adapter_up.bias)

        elif conv_type=='cdc':
            self.adapter_conv = Conv2d_cd(dim, dim, 3, 1, 1)
            if xavier_init:
                nn.init.xavier_uniform_(self.adapter_conv.conv.weight)
            else:
                nn.init.zeros_(self.adapter_conv.conv.weight)
                self.adapter_conv.conv.weight.data[:, :, 1, 1] += torch.eye(8, dtype=torch.float)
            #nn.init.zeros_(self.adapter_conv.conv.bias)

            self.adapter_down = nn.Linear(768, dim)  # equivalent to 1 * 1 Conv
            self.adapter_up = nn.Linear(dim, 768)  # equivalent to 1 * 1 Conv
            nn.init.xavier_uniform_(self.adapter_down.weight)
            nn.init.zeros_(self.adapter_down.bias)
            nn.init.zeros_(self.adapter_up.weight)
            nn.init.zeros_(self.adapter_up.bias)

        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        B, N, C = x.shape

        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)

        x_patch = x_down[:, 1:].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch = self.adapter_conv(x_patch)
        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)

        x_cls = x_down[:, :1].reshape(B, 1, 1, self.dim).permute(0, 3, 1, 2)
        x_cls = self.adapter_conv(x_cls)
        x_cls = x_cls.permute(0, 2, 3, 1).reshape(B, 1, self.dim)

        x_down = torch.cat([x_cls, x_patch], dim=1)

        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv

        return x_up

class Convpass_2_adapter(nn.Module):
    def __init__(self, dim=8, xavier_init=False, conv_type='conv'):
        super().__init__()

        if conv_type=='conv':
            self.adapter_conv_1 = nn.Conv2d(dim, dim, 3, 1, 1)
            self.adapter_conv_2 = nn.Conv2d(dim, dim, 3, 1, 1)

            if xavier_init:
                nn.init.xavier_uniform_(self.adapter_conv_1.weight)
                nn.init.xavier_uniform_(self.adapter_conv_2.weight)
            else:
                nn.init.zeros_(self.adapter_conv_1.weight)
                nn.init.zeros_(self.adapter_conv_2.weight)
                self.adapter_conv_1.conv.data[:, :, 1, 1] += torch.eye(8, dtype=torch.float)
                self.adapter_conv_2.conv.data[:, :, 1, 1] += torch.eye(8, dtype=torch.float)
            nn.init.zeros_(self.adapter_conv_1.bias)
            nn.init.zeros_(self.adapter_conv_2.bias)

            self.adapter_down = nn.Linear(768, dim)  # equivalent to 1 * 1 Conv
            self.adapter_up = nn.Linear(dim, 768)  # equivalent to 1 * 1 Conv
            nn.init.xavier_uniform_(self.adapter_down.weight)
            nn.init.zeros_(self.adapter_down.bias)
            nn.init.zeros_(self.adapter_up.weight)
            nn.init.zeros_(self.adapter_up.bias)

        elif conv_type=='cdc':
            self.adapter_conv = Conv2d_cd(dim, dim, 3, 1, 1)
            if xavier_init:
                nn.init.xavier_uniform_(self.adapter_conv.conv.weight)
            else:
                nn.init.zeros_(self.adapter_conv.conv.weight)
                self.adapter_conv.conv.weight.data[:, :, 1, 1] += torch.eye(8, dtype=torch.float)
            #nn.init.zeros_(self.adapter_conv.conv.bias)

            self.adapter_down = nn.Linear(768, dim)  # equivalent to 1 * 1 Conv
            self.adapter_up = nn.Linear(dim, 768)  # equivalent to 1 * 1 Conv
            nn.init.xavier_uniform_(self.adapter_down.weight)
            nn.init.zeros_(self.adapter_down.bias)
            nn.init.zeros_(self.adapter_up.weight)
            nn.init.zeros_(self.adapter_up.bias)

        self.CAconv = nn.Conv2d(dim, 64, kernel_size=1, padding=0, bias=False)
        self.CAconv2 = nn.Conv2d(64, 2, kernel_size=1, padding=0, bias=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):

        B, N, C = x.shape

        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)

        x_patch = x_down[:, 1:].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch_1 = self.adapter_conv_1(x_patch)
        x_patch_1 = x_patch_1.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)

        x_patch_2 = self.adapter_conv_2(x_patch)
        x_patch_2 = x_patch_2.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)
        
        ChannelAtten = self.softmax(self.CAconv2(self.act(self.CAconv(self.avgpool(x_patch)))))

        x_patch = ChannelAtten[:,0] * x_patch_1 + ChannelAtten[:,1] * x_patch_2
        
        self.feature_patch = [x_patch_1, x_patch_2]

        x_cls = x_down[:, :1].reshape(B, 1, 1, self.dim).permute(0, 3, 1, 2)
        x_cls_1 = self.adapter_conv_1(x_cls)
        x_cls_1 = x_cls_1.permute(0, 2, 3, 1).reshape(B, 1, self.dim)

        x_cls_2 = self.adapter_conv_2(x_cls)
        x_cls_2 = x_cls_2.permute(0, 2, 3, 1).reshape(B, 1, self.dim)

        x_cls = (x_cls_1 + x_cls_2) / 2

        x_down = torch.cat([x_cls, x_patch], dim=1)

        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv

        return x_up

class Convpass_3_adapter(nn.Module):
    def __init__(self, dim=8, xavier_init=False, conv_type='conv'):
        super().__init__()

        if conv_type=='conv':
            self.adapter_conv_1 = nn.Conv2d(dim, dim, 3, 1, 1)
            self.adapter_conv_2 = nn.Conv2d(dim, dim, 3, 1, 1)
            self.adapter_conv_3 = nn.Conv2d(dim, dim, 3, 1, 1)

            if xavier_init:
                nn.init.xavier_uniform_(self.adapter_conv_1.weight)
                nn.init.xavier_uniform_(self.adapter_conv_2.weight)
                nn.init.xavier_uniform_(self.adapter_conv_3.weight)
            else:
                nn.init.zeros_(self.adapter_conv_1.weight)
                nn.init.zeros_(self.adapter_conv_2.weight)
                nn.init.zeros_(self.adapter_conv_3.weight)
                self.adapter_conv_1.conv.data[:, :, 1, 1] += torch.eye(8, dtype=torch.float)
                self.adapter_conv_2.conv.data[:, :, 1, 1] += torch.eye(8, dtype=torch.float)
                self.adapter_conv_3.conv.data[:, :, 1, 1] += torch.eye(8, dtype=torch.float)
            nn.init.zeros_(self.adapter_conv_1.bias)
            nn.init.zeros_(self.adapter_conv_2.bias)
            nn.init.zeros_(self.adapter_conv_3.bias)

            self.adapter_down = nn.Linear(768, dim)  # equivalent to 1 * 1 Conv
            self.adapter_up = nn.Linear(dim, 768)  # equivalent to 1 * 1 Conv
            nn.init.xavier_uniform_(self.adapter_down.weight)
            nn.init.zeros_(self.adapter_down.bias)
            nn.init.zeros_(self.adapter_up.weight)
            nn.init.zeros_(self.adapter_up.bias)

        elif conv_type=='cdc':
            self.adapter_conv = Conv2d_cd(dim, dim, 3, 1, 1)
            if xavier_init:
                nn.init.xavier_uniform_(self.adapter_conv.conv.weight)
            else:
                nn.init.zeros_(self.adapter_conv.conv.weight)
                self.adapter_conv.conv.weight.data[:, :, 1, 1] += torch.eye(8, dtype=torch.float)
            #nn.init.zeros_(self.adapter_conv.conv.bias)

            self.adapter_down = nn.Linear(768, dim)  # equivalent to 1 * 1 Conv
            self.adapter_up = nn.Linear(dim, 768)  # equivalent to 1 * 1 Conv
            nn.init.xavier_uniform_(self.adapter_down.weight)
            nn.init.zeros_(self.adapter_down.bias)
            nn.init.zeros_(self.adapter_up.weight)
            nn.init.zeros_(self.adapter_up.bias)

        self.CAconv = nn.Conv2d(dim, 64, kernel_size=1, padding=0, bias=False)
        self.CAconv2 = nn.Conv2d(64, 3, kernel_size=1, padding=0, bias=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        B, N, C = x.shape

        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)

        x_patch = x_down[:, 1:].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch_1 = self.adapter_conv_1(x_patch)
        x_patch_1 = x_patch_1.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)

        x_patch_2 = self.adapter_conv_2(x_patch)
        x_patch_2 = x_patch_2.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)

        x_patch_3 = self.adapter_conv_3(x_patch)
        x_patch_3 = x_patch_3.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)

        ChannelAtten = self.softmax(self.CAconv2(self.act(self.CAconv(self.avgpool(x_patch)))))

        x_patch = ChannelAtten[:,0] * x_patch_1 + ChannelAtten[:,1] * x_patch_2 + ChannelAtten[:,2] * x_patch_3
        
        self.feature_patch = [x_patch_1, x_patch_2, x_patch_3]

        x_cls = x_down[:, :1].reshape(B, 1, 1, self.dim).permute(0, 3, 1, 2)
        x_cls_1 = self.adapter_conv_1(x_cls)
        x_cls_1 = x_cls_1.permute(0, 2, 3, 1).reshape(B, 1, self.dim)

        x_cls_2 = self.adapter_conv_2(x_cls)
        x_cls_2 = x_cls_2.permute(0, 2, 3, 1).reshape(B, 1, self.dim)

        x_cls_3 = self.adapter_conv_3(x_cls)
        x_cls_3 = x_cls_3.permute(0, 2, 3, 1).reshape(B, 1, self.dim)

        x_cls = (x_cls_1 + x_cls_2 + x_cls_3) / 3

        x_down = torch.cat([x_cls, x_patch], dim=1)

        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv

        return x_up

class Convpass_4_adapter(nn.Module):
    def __init__(self, dim=8, xavier_init=False, conv_type='conv'):
        super().__init__()

        if conv_type=='conv':
            self.adapter_conv_1 = nn.Conv2d(dim, dim, 3, 1, 1)
            self.adapter_conv_2 = nn.Conv2d(dim, dim, 3, 1, 1)
            self.adapter_conv_3 = nn.Conv2d(dim, dim, 3, 1, 1)
            self.adapter_conv_4 = nn.Conv2d(dim, dim, 3, 1, 1)

            if xavier_init:
                nn.init.xavier_uniform_(self.adapter_conv_1.weight)
                nn.init.xavier_uniform_(self.adapter_conv_2.weight)
                nn.init.xavier_uniform_(self.adapter_conv_3.weight)
                nn.init.xavier_uniform_(self.adapter_conv_4.weight)
            else:
                nn.init.zeros_(self.adapter_conv_1.weight)
                nn.init.zeros_(self.adapter_conv_2.weight)
                nn.init.zeros_(self.adapter_conv_3.weight)
                nn.init.zeros_(self.adapter_conv_4.weight)
                self.adapter_conv_1.conv.data[:, :, 1, 1] += torch.eye(8, dtype=torch.float)
                self.adapter_conv_2.conv.data[:, :, 1, 1] += torch.eye(8, dtype=torch.float)
                self.adapter_conv_3.conv.data[:, :, 1, 1] += torch.eye(8, dtype=torch.float)
                self.adapter_conv_4.conv.data[:, :, 1, 1] += torch.eye(8, dtype=torch.float)
            nn.init.zeros_(self.adapter_conv_1.bias)
            nn.init.zeros_(self.adapter_conv_2.bias)
            nn.init.zeros_(self.adapter_conv_3.bias)
            nn.init.zeros_(self.adapter_conv_4.bias)

            self.adapter_down = nn.Linear(768, dim)  # equivalent to 1 * 1 Conv
            self.adapter_up = nn.Linear(dim, 768)  # equivalent to 1 * 1 Conv
            nn.init.xavier_uniform_(self.adapter_down.weight)
            nn.init.zeros_(self.adapter_down.bias)
            nn.init.zeros_(self.adapter_up.weight)
            nn.init.zeros_(self.adapter_up.bias)

        elif conv_type=='cdc':
            self.adapter_conv = Conv2d_cd(dim, dim, 3, 1, 1)
            if xavier_init:
                nn.init.xavier_uniform_(self.adapter_conv.conv.weight)
            else:
                nn.init.zeros_(self.adapter_conv.conv.weight)
                self.adapter_conv.conv.weight.data[:, :, 1, 1] += torch.eye(8, dtype=torch.float)
            #nn.init.zeros_(self.adapter_conv.conv.bias)

            self.adapter_down = nn.Linear(768, dim)  # equivalent to 1 * 1 Conv
            self.adapter_up = nn.Linear(dim, 768)  # equivalent to 1 * 1 Conv
            nn.init.xavier_uniform_(self.adapter_down.weight)
            nn.init.zeros_(self.adapter_down.bias)
            nn.init.zeros_(self.adapter_up.weight)
            nn.init.zeros_(self.adapter_up.bias)

        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        B, N, C = x.shape

        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)

        x_patch = x_down[:, 1:].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch_1 = self.adapter_conv_1(x_patch)
        x_patch_1 = x_patch_1.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)

        x_patch_2 = self.adapter_conv_2(x_patch)
        x_patch_2 = x_patch_2.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)

        x_patch_3 = self.adapter_conv_3(x_patch)
        x_patch_3 = x_patch_3.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)

        x_patch_4 = self.adapter_conv_4(x_patch)
        x_patch_4 = x_patch_4.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)

        self.feature_patch = [x_patch_1, x_patch_2, x_patch_3, x_patch_4]

        x_patch = (x_patch_1 + x_patch_2 + x_patch_3 + x_patch_4) / 4

        x_cls = x_down[:, :1].reshape(B, 1, 1, self.dim).permute(0, 3, 1, 2)
        x_cls_1 = self.adapter_conv_1(x_cls)
        x_cls_1 = x_cls_1.permute(0, 2, 3, 1).reshape(B, 1, self.dim)

        x_cls_2 = self.adapter_conv_2(x_cls)
        x_cls_2 = x_cls_2.permute(0, 2, 3, 1).reshape(B, 1, self.dim)

        x_cls_3 = self.adapter_conv_3(x_cls)
        x_cls_3 = x_cls_3.permute(0, 2, 3, 1).reshape(B, 1, self.dim)

        x_cls_4 = self.adapter_conv_4(x_cls)
        x_cls_4 = x_cls_4.permute(0, 2, 3, 1).reshape(B, 1, self.dim)

        x_cls = (x_cls_1 + x_cls_2 + x_cls_3 + x_cls_4) / 4

        x_down = torch.cat([x_cls, x_patch], dim=1)

        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv

        return x_up


def set_Convpass(model, method, dim, adapter_num, s=1, xavier_init=False, conv_type='conv'):

    print('dim:{}'.format(dim))
    print('adapter_num:{}'.format(adapter_num))

    if method == 'convpass':

        for _ in model.children():
            if type(_) == timm.models.vision_transformer.Block:
                if adapter_num == 1:
                    _.adapter_attn = Convpass(dim, xavier_init, conv_type=conv_type)
                    _.adapter_mlp = Convpass(dim, xavier_init,conv_type=conv_type)
                elif adapter_num == 2:
                    _.adapter_attn = Convpass_2_adapter(dim, xavier_init, conv_type=conv_type)
                    _.adapter_mlp = Convpass_2_adapter(dim, xavier_init,conv_type=conv_type)
                elif adapter_num == 3:
                    _.adapter_attn = Convpass_3_adapter(dim, xavier_init, conv_type=conv_type)
                    _.adapter_mlp = Convpass_3_adapter(dim, xavier_init,conv_type=conv_type)
                else:
                    _.adapter_attn = Convpass_4_adapter(dim, xavier_init, conv_type=conv_type)
                    _.adapter_mlp = Convpass_4_adapter(dim, xavier_init,conv_type=conv_type)
                _.s = s
                bound_method = forward_block.__get__(_, _.__class__)
                setattr(_, 'forward', bound_method)
            elif len(list(_.children())) != 0:
                set_Convpass(_, method, dim, adapter_num, s, xavier_init, conv_type=conv_type)
    else:
        for _ in model.children():
            if type(_) == timm.models.vision_transformer.Block:
                if adapter_num == 1: 
                    _.adapter_attn = Convpass(dim, xavier_init, conv_type=conv_type)
                else:
                    _.adapter_attn = Convpass_2_adapter(dim, xavier_init, conv_type=conv_type)
                _.s = s
                bound_method = forward_block_attn.__get__(_, _.__class__)
                setattr(_, 'forward', bound_method)
            elif len(list(_.children())) != 0:
                set_Convpass(_, method, dim, adapter_num, s, xavier_init, conv_type=conv_type)
