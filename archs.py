
import torch
from torch import nn
from torchstat import stat
import GSCblock

__all__ = ['UNet', 'NestedUNet']



class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out



class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [4,8,16,32, 64]
        # nb_filter = [64, 128, 256, 512, 1024]
        # nb_filter = [32, 64, 128, 256, 512]
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # scale_factor:放大的倍数  插值
        # 进行了下采样卷积
        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        # 进行了上采样卷积
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        # 5次VGG卷积
        x0_0 = self.conv0_0(input)
        # 4次下采样
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # 4次上采样后与上一层cat融合(先上采样了后在融合  卷积了在上采样恢复，在与没有卷积的融合)
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output

# AttentionUnet代码
class AttentionUnet(nn.Module):
    def __init__(self, num_classes, input_channels=3,):
        super(AttentionUnet, self).__init__()

        # 常规 5层
        self.stage_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.stage_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.stage_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.stage_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.stage_5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )

        #  上采样4次
        self.upsample_4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1)
        )
        self.upsample_3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        )
        self.upsample_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        )
        self.upsample_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        )

        self.stage_up_4 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.stage_up_3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.stage_up_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.stage_up_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.Attentiongate1 = AttentionBlock(512, 512, 512)
        self.Attentiongate2 = AttentionBlock(256, 256, 256)
        self.Attentiongate3 = AttentionBlock(128, 128, 128)

        self.final = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = x.float()
        # 下采样过程 5次卷积 4次下采样
        stage_1 = self.stage_1(x)
        stage_2 = self.stage_2(stage_1)
        stage_3 = self.stage_3(stage_2)
        stage_4 = self.stage_4(stage_3)
        stage_5 = self.stage_5(stage_4)

        # 上采样成为第四层大小  + 先注意力  + 然后常规unet结合
        up_4 = self.upsample_4(stage_5)
        stage_4 = self.Attentiongate1(up_4, stage_4)
        up_4_conv = self.stage_up_4(torch.cat([up_4, stage_4], dim=1))

        up_3 = self.upsample_3(up_4_conv)
        stage_3 = self.Attentiongate2(up_3, stage_3)
        up_3_conv = self.stage_up_3(torch.cat([up_3, stage_3], dim=1))

        up_2 = self.upsample_2(up_3_conv)
        stage_2 = self.Attentiongate3(up_2, stage_2)
        up_2_conv = self.stage_up_2(torch.cat([up_2, stage_2], dim=1))

        up_1 = self.upsample_1(up_2_conv)
        up_1_conv = self.stage_up_1(torch.cat([up_1, stage_1], dim=1))

        output = self.final(up_1_conv)

        return output


# unet++
class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        # nb_filter = [32, 64, 128, 256, 512]
        nb_filter = [64, 128, 256, 512, 1024]
        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):

        # 卷积第一次
        # print('input:',input.shape)
        x0_0 = self.conv0_0(input)
        # print('x0_0:',x0_0.shape)
        # 下采样第一次 后卷积  unet常规层  属于第二层了
        x1_0 = self.conv1_0(self.pool(x0_0))
        # print('x1_0:',x1_0.shape)
        # 下采样第一次后 上采样 与 上一次融合   属于第一层的第二个
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        # print('x0_1:',x0_1.shape)

        # 卷积下采样第二次  得到第三层
        x2_0 = self.conv2_0(self.pool(x1_0))
        # print('x2_0:',x2_0.shape)
        # 恢复成第二层 融合卷积  第二层第二个
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        # print('x1_1:',x1_1.shape)
        # 恢复成第一层 融合卷积  第一层第三个
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        # print('x0_2:',x0_2.shape)

        # 卷积下采样第三次  得到第四层
        x3_0 = self.conv3_0(self.pool(x2_0))
        # print('x3_0:',x3_0.shape)
        # 第三层第二个
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        # print('x2_1:',x2_1.shape)
        # 第二层第三个
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        # print('x1_2:',x1_2.shape)
        # 第一层第四个
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        # print('x0_3:',x0_3.shape)

        # 卷积下采样第四次  得到第五层 最终层
        x4_0 = self.conv4_0(self.pool(x3_0))
        # print('x4_0:',x4_0.shape)
        # 第四层第二个
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        # print('x3_1:',x3_1.shape)
        # 第三层第三个
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        # print('x2_2:',x2_2.shape)
        # 第二层第四个
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        # print('x1_3:',x1_3.shape)
        # 第一层第五个 结束  最终结果返回
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        # print('x0_4:',x0_4.shape)

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output




class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g = self.W_g(g)
        x = self.W_x(x)
        psi = self.relu(g + x)
        psi = self.psi(psi)

        return x * psi





class ChannelAttention(nn.Module):
    def __init__(self, in_planes, out_planes, ratio=2):
        super(ChannelAttention, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, 1, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc11 = nn.Conv2d(out_planes, out_planes // ratio, 1, bias=False)
        self.fc12 = nn.Conv2d(out_planes // ratio, out_planes, 1, bias=False)

        self.fc21 = nn.Conv2d(out_planes, out_planes // ratio, 1, bias=False)
        self.fc22 = nn.Conv2d(out_planes // ratio, out_planes, 1, bias=False)
        self.relu1 = nn.ReLU(True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        avg_out = self.fc12(self.relu1(self.fc11(self.avg_pool(x))))
        max_out = self.fc22(self.relu1(self.fc21(self.max_pool(x))))
        out = avg_out + max_out
        del avg_out, max_out
        return x * self.sigmoid(out)





class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class GSCEU_Net(nn.Module):
    def __init__(self, num_classes, input_channels=3, groups=1, kernel_size=56, **kwargs):
        super().__init__()
        # nb_filter = [64,128,256,512,1024]
        nb_filter = [4, 8,16, 32, 64]
        # nb_filter = [ 64, 128,256,512,1024]
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # scale_factor:放大的倍数  插值
        # 进行了下采样卷积
        self.faster1 = GSCblock.GSC_Block(input_channels, nb_filter[0])
        self.faster2 = GSCblock.GSC_Block(nb_filter[0], nb_filter[1])
        self.faster3 = GSCblock.GSC_Block(nb_filter[1], nb_filter[2])
        self.faster4 = GSCblock.GSC_Block(nb_filter[2], nb_filter[3])
        self.faster5 = GSCblock.GSC_Block(nb_filter[3], nb_filter[4])
        # 进行了上采样卷积
        self.conv3_1 = GSCblock.GSC_Block(nb_filter[3] + nb_filter[4], nb_filter[3])
        self.conv2_2 = GSCblock.GSC_Block(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv1_3 = GSCblock.GSC_Block(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0_4 = GSCblock.GSC_Block(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.ECA1 = eca_layer(nb_filter[4])
        self.ECA2 = eca_layer(nb_filter[3])
        self.ECA3 = eca_layer(nb_filter[2])
        self.ECA4 = eca_layer(nb_filter[1])
        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.faster1(input)
        x1_0 = self.faster2(self.pool(x0_0))
        x2_0 = self.faster3(self.pool(x1_0))
        x3_0 = self.faster4(self.pool(x2_0))
        x4_0 = self.faster5(self.pool(x3_0))
        #
        x3_1 = self.conv3_1(torch.cat([x3_0, self.ECA1(self.up(x4_0))], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.ECA2(self.up(x3_1))], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.ECA3(self.up(x2_2))], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.ECA4(self.up(x1_3))], 1))

        # x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        # x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        # x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        # x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

if __name__ == '__main__':
    model=UNet(1,3)
    # model=NestedUNet(1,3)
    # model = AttentionUnet(1)
    # model = AttentionUnet(1)
    # model = fasterUnet(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stat(model, (3, 256, 256))
