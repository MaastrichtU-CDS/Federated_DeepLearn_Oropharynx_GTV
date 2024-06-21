import torch
from torch import nn
from torch.nn import functional as F


class FastSmoothSENorm(nn.Module):
    class SEWeights(nn.Module):
        def __init__(self, in_channels, reduction=2):
            super().__init__()
            self.conv1 = nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0, bias=True)
            self.conv2 = nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0, bias=True)

        def forward(self, x):
            b, c, d, h, w = x.size()
            out = torch.mean(x.view(b, c, -1), dim=-1).view(b, c, 1, 1, 1)
            out = F.relu(self.conv1(out))
            out = self.conv2(out)
            return out

    def __init__(self, in_channels, reduction=2):
        super(FastSmoothSENorm, self).__init__()
        self.norm = nn.InstanceNorm3d(in_channels, affine=False)
        self.gamma = self.SEWeights(in_channels, reduction)
        self.beta = self.SEWeights(in_channels, reduction)

    def forward(self, x):
        gamma = torch.sigmoid(self.gamma(x))
        beta = torch.tanh(self.beta(x))
        x = self.norm(x)
        return gamma * x + beta


class FastSmoothSeNormConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=2, **kwargs):
        super(FastSmoothSeNormConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=True, **kwargs)
        self.norm = FastSmoothSENorm(out_channels, reduction)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        x = self.norm(x)
        return x


class RESseNormConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=2, **kwargs):
        super().__init__()
        self.conv1 = FastSmoothSeNormConv3d(in_channels, out_channels, reduction, **kwargs)

        if in_channels != out_channels:
            self.res_conv = FastSmoothSeNormConv3d(in_channels, out_channels, reduction, kernel_size=1, stride=1, padding=0)
        else:
            self.res_conv = None

    def forward(self, x):
        residual = self.res_conv(x) if self.res_conv else x
        x = self.conv1(x)
        x += residual
        return x


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=2, scale=2):
        super().__init__()
        self.scale = scale
        self.conv = FastSmoothSeNormConv3d(in_channels, out_channels, reduction, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv(x)
        x = F.interpolate(x, scale_factor=self.scale, mode='trilinear', align_corners=False)
        return x


class AttentionGate(nn.Module):
    def __init__(self, in_channels_g, in_channels_l, out_channels, **kwargs):
        super(AttentionGate, self).__init__()
        self.wg = nn.Conv3d(in_channels_g, out_channels, bias=False, padding=0,
                            kernel_size=1, stride=1)
        self.wx = nn.Conv3d(in_channels_l, out_channels, bias=False, padding=0,
                            kernel_size=1, stride=1)

        self.psi = nn.Conv3d(in_channels=out_channels, out_channels=1,
                             bias=False, padding=0, kernel_size=1, stride=1)

    def forward(self, g, x):
        g1 = self.wg(g)
        x1 = self.wx(x)
        psi = torch.sigmoid(self.psi(F.relu(g1 + x1, inplace=True)))
        return x * psi


class FastSmoothSENormDeepUNet_supervision_skip_no_drop(nn.Module):
    """Model copied from: https://github.com/iantsen/hecktor"""

    def __init__(self, in_channels, n_cls, n_filters, reduction=2, return_logits=False):
        super(FastSmoothSENormDeepUNet_supervision_skip_no_drop, self).__init__()
        self.in_channels = in_channels
        self.n_cls = 1 if n_cls == 2 else n_cls
        self.n_filters = n_filters
        self.return_logits = return_logits

        self.block_1_1_left = RESseNormConv3d(in_channels, n_filters, reduction, kernel_size=7, stride=1, padding=3)
        self.block_1_2_left = RESseNormConv3d(n_filters, n_filters, reduction, kernel_size=3, stride=1, padding=1)

        self.pool_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block_2_1_left = RESseNormConv3d(n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_2_2_left = RESseNormConv3d(2 * n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_2_3_left = RESseNormConv3d(2 * n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1, padding=1)

        self.pool_2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block_3_1_left = RESseNormConv3d(2 * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_3_2_left = RESseNormConv3d(4 * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_3_3_left = RESseNormConv3d(4 * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1, padding=1)

        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block_4_1_left = RESseNormConv3d(4 * n_filters, 8 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_4_2_left = RESseNormConv3d(8 * n_filters, 8 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_4_3_left = RESseNormConv3d(8 * n_filters, 8 * n_filters, reduction, kernel_size=3, stride=1, padding=1)

        self.pool_4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block_5_1_left = RESseNormConv3d(8 * n_filters, 16 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_5_2_left = RESseNormConv3d(16 * n_filters, 16 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_5_3_left = RESseNormConv3d(16 * n_filters, 16 * n_filters, reduction, kernel_size=3, stride=1, padding=1)

        self.upconv_4 = nn.ConvTranspose3d(16 * n_filters, 8 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.attention_4 = AttentionGate(in_channels_g=8 * n_filters,
                                         in_channels_l=8 * n_filters,
                                         out_channels=8 * n_filters)
        self.block_4_1_right = FastSmoothSeNormConv3d((8 + 8) * n_filters, 8 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_4_2_right = FastSmoothSeNormConv3d(8 * n_filters, 8 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.vision_4 = UpConv(8 * n_filters, n_filters, reduction, scale=8)

        self.upconv_3 = nn.ConvTranspose3d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.attention_3 = AttentionGate(in_channels_g=4 * n_filters,
                                         in_channels_l=4 * n_filters,
                                         out_channels=4 * n_filters)
        self.block_3_1_right = FastSmoothSeNormConv3d((4 + 4) * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_3_2_right = FastSmoothSeNormConv3d(4 * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.vision_3 = UpConv(4 * n_filters, n_filters, reduction, scale=4)

        self.upconv_2 = nn.ConvTranspose3d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.attention_2 = AttentionGate(in_channels_g=2 * n_filters,
                                         in_channels_l=2 * n_filters,
                                         out_channels=2 * n_filters)
        self.block_2_1_right = FastSmoothSeNormConv3d((2 + 2) * n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_2_2_right = FastSmoothSeNormConv3d(2 * n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.vision_2 = UpConv(2 * n_filters, n_filters, reduction, scale=2)

        self.upconv_1 = nn.ConvTranspose3d(2 * n_filters, 1 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.attention_1 = AttentionGate(in_channels_g=1 * n_filters,
                                         in_channels_l=1 * n_filters,
                                         out_channels=1 * n_filters)
        self.block_1_1_right = FastSmoothSeNormConv3d((1 + 1) * n_filters, n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_1_2_right = FastSmoothSeNormConv3d(n_filters, n_filters, reduction, kernel_size=3, stride=1, padding=1)

        self.conv1x1 = nn.Conv3d(1 * n_filters, self.n_cls, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        ds0 = self.block_1_2_left(self.block_1_1_left(x))
        ds1 = self.block_2_3_left(self.block_2_2_left(self.block_2_1_left(self.pool_1(ds0))))
        ds2 = self.block_3_3_left(self.block_3_2_left(self.block_3_1_left(self.pool_2(ds1))))
        ds3 = self.block_4_3_left(self.block_4_2_left(self.block_4_1_left(self.pool_3(ds2))))
        x = self.block_5_3_left(self.block_5_2_left(self.block_5_1_left(self.pool_4(ds3))))

        up4 = self.upconv_4(x)
        ag4 = self.attention_4(up4, ds3)
        x = self.block_4_2_right(self.block_4_1_right(torch.cat([up4, ag4], 1)))
        sv4 = self.vision_4(x)

        up3 = self.upconv_3(x)
        ag3 = self.attention_3(up3, ds2)
        x = self.block_3_2_right(self.block_3_1_right(torch.cat([up3, ag3], 1)))
        sv3 = self.vision_3(x)

        up2 = self.upconv_2(x)
        ag2 = self.attention_2(up2, ds1)
        x = self.block_2_2_right(self.block_2_1_right(torch.cat([up2, ag2], 1)))
        sv2 = self.vision_2(x)

        up1 = self.upconv_1(x)
        ag1 = self.attention_1(up1, ds0)
        x = self.block_1_1_right(torch.cat([up1, ag1], 1))
        x = x + sv4 + sv3 + sv2
        x = self.block_1_2_right(x)

        x = self.conv1x1(x)

        if self.return_logits:
            return x
        else:
            if self.n_cls == 1:
                return torch.sigmoid(x)
            else:
                return F.softmax(x, dim=1)
