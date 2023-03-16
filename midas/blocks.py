import torch
import torch.nn as nn

from .vit import (
    _make_pretrained_vitb_rn50_384,
    _make_pretrained_vitl16_384,
    _make_pretrained_vitb16_384,
    forward_vit,
)

from midas.DeformConv2d_sphe import DeformConv2d_sphe, DeformConv2d_sphe_SameExport, mySequential


def _make_encoder(backbone, features, use_pretrained, groups=1, expand=False, exportable=True, hooks=None, use_vit_only=False, use_readout="ignore",):
    if backbone == "vitl16_384":
        pretrained = _make_pretrained_vitl16_384(
            use_pretrained, hooks=hooks, use_readout=use_readout
        )
        scratch = _make_scratch(
            [256, 512, 1024, 1024], features, groups=groups, expand=expand
        )  # ViT-L/16 - 85.0% Top1 (backbone)
    elif backbone == "vitb_rn50_384":
        pretrained = _make_pretrained_vitb_rn50_384(
            use_pretrained,
            hooks=hooks,
            use_vit_only=use_vit_only,
            use_readout=use_readout,
        )
        scratch = _make_scratch(
            [256, 512, 768, 768], features, groups=groups, expand=expand
        )  # ViT-H/16 - 85.0% Top1 (backbone)
    elif backbone == "vitb16_384":
        pretrained = _make_pretrained_vitb16_384(
            use_pretrained, hooks=hooks, use_readout=use_readout
        )
        scratch = _make_scratch(
            [96, 192, 384, 768], features, groups=groups, expand=expand
        )  # ViT-B/16 - 84.6% Top1 (backbone)
    elif backbone == "resnext101_wsl":
        pretrained = _make_pretrained_resnext101_wsl(use_pretrained)
        scratch = _make_scratch([256, 512, 1024, 2048], features, groups=groups, expand=expand)     # efficientnet_lite3
    elif backbone == "efficientnet_lite3":
        pretrained = _make_pretrained_efficientnet_lite3(use_pretrained, exportable=exportable)
        scratch = _make_scratch([32, 48, 136, 384], features, groups=groups, expand=expand)  # efficientnet_lite3
    elif backbone == "efficientnet_lite3_sphe1":
        pretrained = _make_pretrained_efficientnet_lite3_sphe(use_pretrained, exportable=exportable)
        scratch = _make_scratch([32, 48, 136, 384], features, groups=groups, expand=expand)  # efficientnet_lite3
    elif backbone == "efficientnet_lite3_sphe12":
        pretrained = _make_pretrained_efficientnet_lite3_sphe2(use_pretrained, exportable=exportable)
        scratch = _make_scratch([32, 48, 136, 384], features, groups=groups, expand=expand)  # efficientnet_lite3
    elif backbone == "efficientnet_lite3_sphe101":
        pretrained = _make_pretrained_efficientnet_lite3(use_pretrained, exportable=exportable)
        scratch = _make_scratch_sphe([32, 48, 136, 384], features, groups=groups, expand=expand)  # efficientnet_lite3
    elif backbone == "efficientnet_lite3_sphe2":
        pretrained = _make_pretrained_efficientnet_lite3_sphe(use_pretrained, exportable=exportable)
        scratch = _make_scratch_sphe([32, 48, 136, 384], features, groups=groups, expand=expand)  # efficientnet_lite3
    elif backbone == "efficientnet_lite3_spheFULL":
        pretrained = _make_pretrained_efficientnet_lite3_spheFULL(use_pretrained, exportable=exportable)
        scratch = _make_scratch_spheFULL([32, 48, 136, 384], features, groups=groups, expand=expand)  # efficientnet_lite3
    elif backbone == "efficientnet_lite3_sphe_FL":
        pretrained = _make_pretrained_efficientnet_lite3_sphe(use_pretrained, exportable=exportable)
        scratch = _make_scratch([32, 48, 136, 384], features, groups=groups, expand=expand)  # efficientnet_lite3
    elif backbone == "efficientnet_lite3_sphe_FL2":
        pretrained = _make_pretrained_efficientnet_lite3_sphe2(use_pretrained, exportable=exportable)
        scratch = _make_scratch([32, 48, 136, 384], features, groups=groups, expand=expand)  # efficientnet_lite3
    elif backbone == "efficientnet_lite3_sphe_FULL":
        pretrained = _make_pretrained_efficientnet_lite3_spheFULL(use_pretrained, exportable=exportable)
        scratch = _make_scratch_spheFULL([32, 48, 136, 384], features, groups=groups, expand=expand)  # efficientnet_lite3
    else:
        print(f"Backbone '{backbone}' not implemented")
        assert False

    return pretrained, scratch


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape
    if expand == True:
        out_shape1 = out_shape
        out_shape2 = out_shape*2
        out_shape3 = out_shape*4
        out_shape4 = out_shape*8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    # scratch.layer4_rn = DeformConv2d_sphe(
    #     in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    # )

    return scratch


def _make_scratch_sphe(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape
    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape*2
        out_shape3 = out_shape*4
        out_shape4 = out_shape*8

    scratch.layer1_rn = DeformConv2d_sphe(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = DeformConv2d_sphe(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = DeformConv2d_sphe(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer4_rn = DeformConv2d_sphe(
        in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )

    return scratch


def _make_scratch_spheFULL(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape
    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape*2
        out_shape3 = out_shape*4
        out_shape4 = out_shape*8

    scratch.layer1_rn = DeformConv2d_sphe(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = DeformConv2d_sphe(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = DeformConv2d_sphe(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer4_rn = DeformConv2d_sphe(
        in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )

    return scratch


def _make_pretrained_efficientnet_lite3(use_pretrained, exportable=False):
    efficientnet = torch.hub.load(
        "weights/rwightman_gen-efficientnet-pytorch_master",
        "tf_efficientnet_lite3",
        pretrained=use_pretrained,
        exportable=exportable,
        source='local'
    )
    # efficientnet = torch.hub.load(
    #     "rwightman/gen-efficientnet-pytorch",
    #     "tf_efficientnet_lite3",
    #     pretrained=use_pretrained,
    #     exportable=exportable
    # )
    return _make_efficientnet_backbone(efficientnet)


def _make_efficientnet_backbone(effnet):
    pretrained = nn.Module()

    pretrained.layer1 = nn.Sequential(
        effnet.conv_stem, effnet.bn1, effnet.act1, *effnet.blocks[0:2]
    )

    pretrained.layer2 = nn.Sequential(*effnet.blocks[2:3])
    pretrained.layer3 = nn.Sequential(*effnet.blocks[3:5])
    pretrained.layer4 = nn.Sequential(*effnet.blocks[5:9])

    return pretrained


def _make_pretrained_efficientnet_lite3_sphe(use_pretrained, exportable=False):
    efficientnet = torch.hub.load(
        "weights/rwightman_gen-efficientnet-pytorch_master",
        "tf_efficientnet_lite3",
        pretrained=use_pretrained,
        exportable=exportable,
        source='local'
    )
    return _make_efficientnet_backbone_sphe(efficientnet)


def _make_pretrained_efficientnet_lite3_sphe2(use_pretrained, exportable=False):
    efficientnet = torch.hub.load(
        "weights/rwightman_gen-efficientnet-pytorch_master",
        "tf_efficientnet_lite3",
        pretrained=use_pretrained,
        exportable=exportable,
        source='local'
    )
    return _make_efficientnet_backbone_sphe2(efficientnet)


def _make_pretrained_efficientnet_lite3_spheFULL(use_pretrained, exportable=False):
    efficientnet = torch.hub.load(
        "weights/rwightman_gen-efficientnet-pytorch_master",
        "tf_efficientnet_lite3",
        pretrained=use_pretrained,
        exportable=exportable,
        source='local'
    )
    return _make_efficientnet_backbone_spheFULL(efficientnet)


def _make_efficientnet_backbone_sphe(effnet):
    pretrained = nn.Module()

    # Conv2dSameExport(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
    effnet.conv_stem = DeformConv2d_sphe(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)

    # print(effnet.blocks[0][0])
    # import sys
    # sys.exit()

    pretrained.layer1 = mySequential(
        effnet.conv_stem, effnet.bn1, effnet.act1, *effnet.blocks[0:2]
    )
    # print(pretrained.layer1)
    # print(effnet.conv_stem)
    # import sys
    # sys.exit()
    pretrained.layer2 = nn.Sequential(*effnet.blocks[2:3])
    pretrained.layer3 = nn.Sequential(*effnet.blocks[3:5])
    pretrained.layer4 = nn.Sequential(*effnet.blocks[5:9])

    return pretrained


def _make_efficientnet_backbone_sphe2(effnet):
    pretrained = nn.Module()

    # Conv2dSameExport(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
    # effnet.conv_stem = DeformConv2d_sphe(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
    # effnet.conv_stem = Conv2dSameExport_sphe(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
    effnet.conv_stem = DeformConv2d_sphe_SameExport(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)

    # print(effnet.blocks[-2][1].conv_dw)
    # import sys
    # sys.exit()
    # effnet.blocks[-2][5].conv_dw = DeformConv2d_sphe(1392, 1392, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1392, bias=False)

    print("LAST LAYER ACTIVE")
    effnet.blocks[-1][0].conv_dw = DeformConv2d_sphe(1392, 1392, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1392, bias=False)

    pretrained.layer1 = mySequential(
        effnet.conv_stem, effnet.bn1, effnet.act1, *effnet.blocks[0:2]
    )
    pretrained.layer2 = nn.Sequential(*effnet.blocks[2:3])
    pretrained.layer3 = nn.Sequential(*effnet.blocks[3:5])
    pretrained.layer4 = nn.Sequential(*effnet.blocks[5:9])

    return pretrained


def _make_efficientnet_backbone_spheFULL(effnet):
    pretrained = nn.Module()

    # Conv2dSameExport(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
    # effnet.conv_stem = DeformConv2d_sphe(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
    # effnet.conv_stem = Conv2dSameExport_sphe(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
    effnet.conv_stem = DeformConv2d_sphe_SameExport(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)

    pretrained.layer1 = mySequential(
        effnet.conv_stem, effnet.bn1, effnet.act1, *effnet.blocks[0:2]
    )
    # print(pretrained.layer1)
    # print(effnet.conv_stem)
    # import sys
    # import pdb
    # pdb.set_trace()
    pretrained.layer2 = nn.Sequential(*effnet.blocks[2:3])
    pretrained.layer3 = nn.Sequential(*effnet.blocks[3:5])
    pretrained.layer4 = nn.Sequential(*effnet.blocks[5:9])

    # print(pretrained)

    pretrained_copy = pretrained

    CCOUNT = 0
    for name, p in pretrained.named_parameters():
        lnames = name.split('.')
        # print(lnames)
        if "conv" in name:
            cnv = pretrained._modules[lnames[0]]._modules[lnames[1]]._modules[lnames[2]]._modules[lnames[3]]
            # if cnv.kernel_size != (1, 1) and CCOUNT <= 5:
            if cnv.kernel_size != (1, 1):
                CCOUNT += 1
                # print(type(cnv).__name__)
                if type(cnv).__name__ == "Conv2dSameExport":
                    custom_cnv = DeformConv2d_sphe_SameExport(cnv.in_channels, cnv.out_channels, cnv.kernel_size,  cnv.stride, cnv.padding,
                                                              cnv.dilation, groups=cnv.groups, bias=False)  # weight=cnv.weight.data
                    pretrained_copy._modules[lnames[0]]._modules[lnames[1]]._modules[lnames[2]]._modules[lnames[3]] = custom_cnv
                else:
                    custom_cnv = DeformConv2d_sphe(cnv.in_channels, cnv.out_channels, cnv.kernel_size,  cnv.stride, cnv.padding, cnv.dilation, groups=cnv.groups, bias=False)  # weight=cnv.weight.data
                    pretrained_copy._modules[lnames[0]]._modules[lnames[1]]._modules[lnames[2]]._modules[lnames[3]] = custom_cnv

    pretrained = pretrained_copy
    # import sys
    # sys.exit()
    return pretrained


def _make_resnet_backbone(resnet):
    pretrained = nn.Module()
    pretrained.layer1 = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
    )

    pretrained.layer2 = resnet.layer2
    pretrained.layer3 = resnet.layer3
    pretrained.layer4 = resnet.layer4

    return pretrained


def _make_pretrained_resnext101_wsl(use_pretrained):
    resnet = torch.hub.load("facebookresearch/WSL-Images", "resnext101_32x8d_wsl")
    return _make_resnet_backbone(resnet)


class Interpolate(nn.Module):
    """Interpolation module.
    """

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners
        )

        return x


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x


class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=True
        )

        return output


class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        # self.conv1 = DeformConv2d_sphe(
        #     features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        # )

        # self.conv2 = DeformConv2d_sphe(
        #     features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        # )

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)

        # return out + x


class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features//2

        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
            # output += res

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output


class ResidualConvUnit_custom_sphe(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        # self.conv1 = nn.Conv2d(
        #     features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        # )

        # self.conv2 = nn.Conv2d(
        #     features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        # )

        self.conv1 = DeformConv2d_sphe(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        self.conv2 = DeformConv2d_sphe(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)

        # return out + x


class ResidualConvUnit_custom_sphe_half(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        # self.conv2 = nn.Conv2d(
        #     features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        # )

        # self.conv1 = DeformConv2d_sphe(
        #     features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        # )

        self.conv2 = DeformConv2d_sphe(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)

        # return out + x


class FeatureFusionBlock_custom_sphe(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom_sphe, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features//2

        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        # self.resConfUnit1 = ResidualConvUnit_custom_sphe(features, activation, bn)
        # self.resConfUnit2 = ResidualConvUnit_custom_sphe(features, activation, bn)

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom_sphe(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            # print(res.shape)
            # print(output.shape)
            output = self.skip_add.add(output, res)
            # output += res

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output
