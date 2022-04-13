#!/usr/bin/env python

import os.path as osp

import torch
import torch.nn as nn

from . import resnet, resnext, mobilenet, hrnet
from .basnet import BASNet
from .unet import UNet, NestedUNet
from .hrnet import HRNetV2
from .models import C1, CNHead, PPM, UPerNet, MobileNetV2Dilated, Resnet, ResnetDilated, CAModule
from .segnet import SegNet

from tasks.utils.logger import configure_logger
from tasks.utils.tensor import to_mask
from tasks.utils.io import save_model


logger = configure_logger(modname=__name__)


def build_model(algorithm, use_dims=3, **kwargs):
    """Buiild model for 2D segmentation

    Args:
        algorithm (dict[str, any]): configuration of algorithm
        use_dims (int, optional): number of dimensions for input image

    Returns:
        model (torch.nn.Module)
    """
    # if name is defined
    name = algorithm.get("name", None)
    # if encoder and decoder are defined
    encoder = algorithm.get("encoder", None)
    decoder = algorithm.get("decoder", None)
    # if two-stage models are defined
    # stage1/2 is dict()
    stage1 = algorithm.get("stage1", None)
    stage2 = algorithm.get("stage2", None)

    if not (name or (encoder and decoder)) is None:
        num_classes = algorithm["num_classes"]

    if kwargs.get("num_classes") is not None:
        num_classes = kwargs["num_classes"]

    in_channels = use_dims

    # in case of in_channels is specified
    in_channels = in_channels if algorithm.get(
        "in_channels") is None else algorithm["in_channels"]

    # parameters for deep supervision
    deep_sup = algorithm.get("deep_sup", False)
    deep_sup_scale = algorithm.get("deep_sup_scale", 1.0)

    if name is not None:
        # Build by `name`
        name = name.lower()
        if name == "unet":
            model = UNet(in_channels, num_classes)
        elif name == "nested_unet":
            model = NestedUNet(in_channels, num_classes, deep_sup)
        elif name == "hrnetv2":
            model = HRNetV2(in_channels, num_classes)
        elif name == "segnet":
            model = SegNet(in_channels, num_classes)
        elif name == "basnet":
            deep_sup = True
            model = BASNet(in_channels, num_classes)
        else:
            raise ValueError("unsupported model `{}`".format(name))
    elif not (encoder and decoder) is None:
        # Build by `encoder` and `decoder`
        encoder = encoder.lower()
        decoder = decoder.lower()
        # ------ encoder -------
        pretrained = algorithm.get("pretrained", True)
        if pretrained and in_channels != 3:
            logger.warn(
                "if use pretrained encoder, input channels must be 3, CAModule() will be used")
            in_channels_ = 3
        else:
            in_channels_ = in_channels

        if encoder == "mobilenetv2dilated":
            orig_mobilenet = mobilenet.__dict__[
                "mobilenetv2"](in_channels=in_channels_, pretrained=pretrained)
            net_encoder = MobileNetV2Dilated(
                orig_mobilenet, dilate_scale=8, is_encoder=True)
        elif encoder == "resnet18":
            orig_resnet = resnet.__dict__["resnet18"](
                in_channels=in_channels_, pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif encoder == "resnet18dilated":
            orig_resnet = resnet.__dict__["resnet18"](
                in_channels=in_channels_, pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif encoder == "resnet50":
            orig_resnet = resnet.__dict__["resnet50"](
                in_channels=in_channels_, pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif encoder == "resnet50dilated":
            orig_resnet = resnet.__dict__["resnet50"](
                in_channels=in_channels_, pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif encoder == "resnet101":
            orig_resnet = resnet.__dict__["resnet101"](
                in_channels=in_channels_, pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif encoder == "resnet101dilated":
            orig_resnet = resnet.__dict__["resnet101"](
                in_channels=in_channels_, pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif encoder == "resnext101":
            orig_resnext = resnext.__dict__[
                "resnext101"](in_channels=in_channels_, pretrained=pretrained)
            net_encoder = Resnet(orig_resnext)  # we can still use class Resnet
        elif encoder == "hrnetv2":
            net_encoder = hrnet.__dict__["hrnetv2"](
                in_channels=in_channels_, pretrained=pretrained, is_encoder=True)
        elif name == "nested_unet":
            model = NestedUNet(in_channels, num_classes, deep_sup)
        else:
            raise Exception("unsupported encoder `{}`".format(encoder))

        if pretrained and in_channels != 3:
            logger.warn("Encoder is loaded with CAModule()")
            net_encoder = CAModule(model=net_encoder, in_channels=in_channels)

        # ----- Decoder ------
        fc_dim = algorithm.get("fc_dim", 512)
        use_softmax = algorithm.get("use_softmax", False)
        if decoder == "c1":
            net_decoder = C1(
                num_class=num_classes,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                deep_sup=deep_sup)
        elif decoder == "cnhead":
            num_layers = algorithm.get("num_layers", 1)
            net_decoder = CNHead(
                num_layers=num_layers,
                num_class=num_classes,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                deep_sup=deep_sup)
        elif decoder == "ppm":
            net_decoder = PPM(num_class=num_classes, fc_dim=fc_dim,
                              use_softmax=use_softmax, deep_sup=deep_sup)
        elif decoder == "upernet_lite":
            net_decoder = UPerNet(
                num_class=num_classes, fc_dim=fc_dim, use_softmax=use_softmax, fpn_dim=256)
        elif decoder == "upernet":
            net_decoder = UPerNet(
                num_class=num_classes, fc_dim=fc_dim, use_softmax=use_softmax, fpn_dim=512)
        else:
            raise Exception("unsupported decoder `{}`".format(decoder))

        model = EDModule(net_encoder, net_decoder)
    elif not (stage1 and stage2) is None:
        # Build by `stage1` and `stage2`
        multiply = algorithm.get("multiply", True)
        if not multiply:
            stage2.update({"in_channels": in_channels + stage1["num_classes"]})
        net_stage1 = build_model(stage1, use_dims=use_dims)
        net_stage2 = build_model(stage2, use_dims=use_dims)

        stage1_weight = stage1.get("pretrained_path")
        stage2_weight = stage2.get("pretrained_path")
        model = TwoStageModule(
            net_stage1,
            net_stage2,
            stage1_weight=stage1_weight,
            stage2_weight=stage2_weight,
            multiply=multiply
        )
    else:
        raise Exception("name: {}, encoder: {}, decoder: {}, stage1: {}, stage2: {}".format(
            name, encoder, decoder, stage1, stage2))

    # set attribute
    if not hasattr(model, "name"):
        setattr(model, "name", model.__class__.__name__)

    if not hasattr(model, "deep_sup"):
        setattr(model, "deep_sup", deep_sup)

    if not hasattr(model, "deep_sup_scale"):
        setattr(model, "deep_sup_scale", deep_sup_scale)

    logger.info("Building Segmentation2d model: {}".format(model.name))

    return model


class EDModule(nn.Module):
    """Encoder and Decoder module

    Args:
        encoder (torch.nn.Module)
        decoder (torch.nn.Module)
    """

    def __init__(self, encoder, decoder):
        super(EDModule, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.enc_name = encoder.__class__.__name__
        self.dec_name = decoder.__class__.__name__

        self._name = self.enc_name + "+" + self.dec_name

    @property
    def name(self):
        return self._name

    def save_models_separately(self, save_dir, prefix=None):
        """Save models' weight separately
        Args:
            save_dir (str)
            prefix (str, optional)
        """
        enc_filename = self.enc_name
        dec_filename = self.dec_name

        if prefix is not None:
            enc_filename += ("_" + str(prefix))
            dec_filename += ("_" + str(prefix))

        enc_filename = osp.join(save_dir, enc_filename + ".pth")
        dec_filename = osp.join(save_dir, dec_filename + ".pth")

        save_model(self.encoder, enc_filename)
        save_model(self.decoder, dec_filename)

    def forward(self, inputs):
        segSize = inputs.shape[2:]

        x = self.encoder(inputs)
        x = self.decoder(x, segSize)

        return x


class TwoStageModule(nn.Module):
    """2-stage module

    Args:
        stage1 (torch.nn.Module)
        stage2 (torch.nn.Module)
        use_mask (bool, optional): whether convert BxNxHxW stage1 output to Bx1xHxW mask
        multiply (bool, optional): whether multiply original input and
            stage1 output being used as stage2 input
    """

    def __init__(self, stage1, stage2, use_mask=True, multiply=True, freeze=False, **kwargs):
        super(TwoStageModule, self).__init__()
        self.stage1 = stage1
        self.stage2 = stage2

        # Load state dict
        stage1_weight = kwargs.get("stage1_weight")
        if stage1_weight is not None:
            device1 = next(self.stage1.parameters()).device
            self.stage1.load_state_dict(torch.load(
                stage1_weight, map_location=device1), strict=True)
            logger.info("stage1 is loaded with: {}".format(stage1_weight))
        stage2_weight = kwargs.get("stage2_weight")
        if stage2_weight is not None:
            device2 = next(self.stage2.parameters()).device
            self.stage2.load_state_dict(torch.load(
                stage2_weight, map_location=device2), strict=True)
            logger.info("stage2 is loaded with: {}".format(stage2_weight))

        if use_mask is False and multiply:
            raise ValueError(
                "in case of 'use_mask' is False, 'multiply' cannot be specified")
        self.use_mask = use_mask
        self.multiply = multiply
        self.freeze = freeze

        if self.freeze:
            for param in self.stage1.parameters():
                param.required_grad = False

        # Register name
        self._name = self.stage1.name + "+" + self.stage2.name

    @property
    def name(self):
        return self._name

    def forward(self, x):
        x_stage1 = self.stage1(x)

        mask = to_mask(x_stage1) if self.use_mask else x_stage1
        x_masked = mask * x if self.multiply else torch.cat((x, mask), dim=1)

        x_stage2 = self.stage2(x_masked)

        return x_stage1, x_stage2
