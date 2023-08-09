from timm.models.layers import DropPath
from torch import nn
import torch
import math

class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x

class Separate_conv3(nn.Module):
    def __init__(self, dim, dimout,n_div, forward):
        super().__init__()

        self.dim_conv3 = dim // n_div
        if self.dim_conv3 == 0:
            self.dim_conv3 = 1
        self.dim_conv3out = dimout // n_div

        self.dim_untouched = dim - self.dim_conv3
        self.dim_untouchedout = dimout - self.dim_conv3out
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3out, 3, 1, 1, bias=False)
        self.partial_conv1 = nn.Conv2d(self.dim_untouched, self.dim_untouchedout, 1, 1, 0, bias=False)
        self.forward = self.forward_split_cat


    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)

        x1 = self.partial_conv3(x1)
        x2 = self.partial_conv1(x2)
        x = torch.cat((x1, x2), 1)
        return x


#
# def Conv(dim, mlp_hidden_dim, param):
#      nn.Conv2d(dim, mlp_hidden_dim, param, padding=0)
#      nn.BatchNorm2d(mlp_hidden_dim)
#      nn.ReLU(inplace=True)
class Conv(nn.Module):
    def __init__(self, dim, mlp_hidden_dim, param, pading = 0):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(dim, mlp_hidden_dim, param, padding=pading)
        self.bn = nn.BatchNorm2d(mlp_hidden_dim)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class GSC_Block(nn.Module):
    def __init__(self,
                 inc,
                 dim,
                 n_div=4,
                 mlp_ratio=2,
                 drop_path=0.1,
                 layer_scale_init_value=0.0,
                 pconv_fw_type='split_cat'
                 ):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div
        self.sig = nn.Sigmoid()

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer = [
            Conv(dim, mlp_hidden_dim, 1),
            Conv(mlp_hidden_dim, dim, 1),
            # nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        # self.spatial_mixing = Partial_conv3(
        #     dim,
        #     n_div,
        #     pconv_fw_type
        # )
        self.spatial_mixing = Separate_conv3(
            inc,
            dim,
            n_div,
            pconv_fw_type
        )

        self.adjust_channel = None
        if inc != dim:
            self.adjust_channel = GhostModule(inc, dim)

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x):
        Ghostshortcut = x
        Ghostshortcut = self.adjust_channel(Ghostshortcut)
        x = self.spatial_mixing(x)
        x = self.mlp(x)
        x = self.drop_path(x) + Ghostshortcut
        return x

    def forward_layer_scale(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


# class C3_Faster(C3):
#     # C3 module with cross-convolutions
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         super().__init__(c1, c2, n, shortcut, g, e)
#         c_ = int(c2 * e)
#         self.m = nn.Sequential(*(Faster_Block(c_, c_) for _ in range(n)))


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        # 第一次卷积：得到通道数为init_channels，是输出的 1/ratio
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential())

        # 第二次卷积：注意有个参数groups，为分组卷积
        # 每个feature map被卷积成 raito-1 个新的 feature map
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        # 第一次卷积得到的 feature map，被作为 identity
        # 和第二次卷积的结果拼接在一起
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]