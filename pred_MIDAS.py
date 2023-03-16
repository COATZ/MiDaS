"""Compute depth maps for images in the input folder.
"""
# import sys
# MIDAS_PATH = '/media/cartizzu/DATA/LIN/2_CODE/3_DEPTH/MiDaS/'
# sys.path.append(MIDAS_PATH)

import numpy as np
from midas.transforms import Resize, NormalizeImage, PrepareForNet
from midas.midas_net_custom import MidasNet_small
from midas.midas_net_custom_sphe import MidasNet_small_sphe
from midas.midas_net import MidasNet
from midas.dpt_depth import DPTDepthModel
from torchvision.transforms import Compose
import argparse
import cv2
import midas.utils as utils
import torch
import glob
import os


def init(model_type="midas_v21_small"):
    print("initialize")

    # MIDAS_PATH = './midas/'
    MIDAS_PATH = './'

    default_models = {
        "midas_v21_small": os.path.join(MIDAS_PATH, "weights/midas_v21_small-70d6b9c8.pt"),
        "midas_v21_small_sphe01": os.path.join(MIDAS_PATH, "weights/midas_v21_small-70d6b9c8.pt"),
        "midas_v21_small_sphe10": os.path.join(MIDAS_PATH, "weights/midas_v21_small-70d6b9c8.pt"),
        "midas_v21_small_sphe11": os.path.join(MIDAS_PATH, "weights/midas_v21_small-70d6b9c8.pt"),
        "midas_v21_small_sphe12": os.path.join(MIDAS_PATH, "weights/midas_v21_small-70d6b9c8.pt"),
        "midas_v21_small_sphe101": os.path.join(MIDAS_PATH, "weights/midas_v21_small-70d6b9c8.pt"),
        "midas_v21_small_sphe2": os.path.join(MIDAS_PATH, "weights/midas_v21_small-70d6b9c8.pt"),
        "midas_v21_small_spheFULL": os.path.join(MIDAS_PATH, "weights/midas_v21_small-70d6b9c8.pt"),
        "midas_v21": os.path.join(MIDAS_PATH, "weights/midas_v21-f6b98070.pt"),
        "dpt_large": os.path.join(MIDAS_PATH, "weights/dpt_large-midas-2f21e586.pt"),
        "dpt_hybrid": os.path.join(MIDAS_PATH, "weights/dpt_hybrid-midas-501f0c75.pt"),

        "midas_v21_small_sphe_FL": os.path.join(MIDAS_PATH, "weights/midas_v21_small-70d6b9c8.pt"),
        "midas_v21_small_sphe_FL2": os.path.join(MIDAS_PATH, "weights/midas_v21_small-70d6b9c8.pt"),
        "midas_v21_small_sphe_encoder": os.path.join(MIDAS_PATH, "weights/midas_v21_small-70d6b9c8.pt"),
        "midas_v21_small_sphe_decoder": os.path.join(MIDAS_PATH, "weights/midas_v21_small-70d6b9c8.pt"),
        "midas_v21_small_sphe_FULL": os.path.join(MIDAS_PATH, "weights/midas_v21_small-70d6b9c8.pt"),
        "midas_v21_small_sphe_FL+decoder": os.path.join(MIDAS_PATH, "weights/midas_v21_small-70d6b9c8.pt"),
        "midas_v21_small_sphe_FL2+decoder": os.path.join(MIDAS_PATH, "weights/midas_v21_small-70d6b9c8.pt"),
        "midas_v21_small_sphe_E1LL+decoder": os.path.join(MIDAS_PATH, "weights/midas_v21_small-70d6b9c8.pt"),
        "midas_v21_small_sphe_E1LL+LLres+decoder": os.path.join(MIDAS_PATH, "weights/midas_v21_small-70d6b9c8.pt"),
    }

    model_path = default_models[model_type]

    # load network
    if model_type == "dpt_large":  # DPT-Large
        model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
        )
    #     net_w, net_h = 384, 384
    #     resize_mode = "minimal"
    #     normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # elif model_type == "dpt_hybrid":  # DPT-Hybrid
    #     model = DPTDepthModel(
    #         path=model_path,
    #         backbone="vitb_rn50_384",
    #         non_negative=True,
    #     )
    #     net_w, net_h = 384, 384
    #     resize_mode = "minimal"
    #     normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # elif model_type == "midas_v21":
    #     model = MidasNet(model_path, non_negative=True)
    #     net_w, net_h = 384, 384
    #     resize_mode = "upper_bound"
    #     normalization = NormalizeImage(
    #         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    #     )
    elif model_type == "midas_v21_small":
        model = MidasNet_small(model_path, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})
        net_w, net_h = 256, 256
        resize_mode = "upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    elif model_type == "midas_v21_small_sphe10":
        model = MidasNet_small_sphe(model_path, features=64, decoder="sphe", backbone="efficientnet_lite3", FFBc="persp", exportable=True, non_negative=True, blocks={'expand': True})
        net_w, net_h = 256, 256
        resize_mode = "upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    # elif model_type == "midas_v21_small_sphe11":
    #     model = MidasNet_small_sphe(model_path, features=64, decoder="sphe", backbone="efficientnet_lite3_sphe1", FFBc="persp", exportable=True, non_negative=True, blocks={'expand': True})
    #     net_w, net_h = 256, 256
    #     resize_mode = "upper_bound"
    #     normalization = NormalizeImage(
    #         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    #     )
    # elif model_type == "midas_v21_small_sphe12":
    #     model = MidasNet_small_sphe(model_path, features=64, decoder="sphe", backbone="efficientnet_lite3_sphe12", FFBc="persp", exportable=True, non_negative=True, blocks={'expand': True})
    #     net_w, net_h = 256, 256
    #     resize_mode = "upper_bound"
    #     normalization = NormalizeImage(
    #         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    #     )
    # elif model_type == "midas_v21_small_sphe101":
    #     model = MidasNet_small_sphe(model_path, features=64, decoder="sphe", backbone="efficientnet_lite3_sphe101", FFBc="sphe", exportable=True, non_negative=True, blocks={'expand': True})
    #     net_w, net_h = 256, 256
    #     resize_mode = "upper_bound"
    #     normalization = NormalizeImage(
    #         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    #     )
    elif model_type == "midas_v21_small_sphe01":
        model = MidasNet_small_sphe(model_path, features=64, decoder="persp", backbone="efficientnet_lite3_sphe1", FFBc="persp", exportable=True, non_negative=True, blocks={'expand': True})
        net_w, net_h = 256, 256
        resize_mode = "upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    # elif model_type == "midas_v21_small_sphe2":
    #     model = MidasNet_small_sphe(model_path, features=64, decoder="sphe", backbone="efficientnet_lite3_sphe2", FFBc="sphe", exportable=True, non_negative=True, blocks={'expand': True})
    #     net_w, net_h = 256, 256
    #     resize_mode = "upper_bound"
    #     normalization = NormalizeImage(
    #         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    #     )
    # elif model_type == "midas_v21_small_spheFULL":
    #     model = MidasNet_small_sphe(model_path, features=64, decoder="sphe", backbone="efficientnet_lite3_spheFULL", FFBc="sphe", exportable=True, non_negative=True, blocks={'expand': True})
    #     net_w, net_h = 256, 256
    #     resize_mode = "upper_bound"
    #     normalization = NormalizeImage(
    #         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    #     )

    elif model_type == "midas_v21_small_sphe_FL":
        model = MidasNet_small_sphe(model_path, features=64, decoder="persp", backbone="efficientnet_lite3_sphe_FL", FFBc="persp", exportable=True, non_negative=True, blocks={'expand': True})
        net_w, net_h = 256, 256
        resize_mode = "upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    elif model_type == "midas_v21_small_sphe_FL2":
        model = MidasNet_small_sphe(model_path, features=64, decoder="persp", backbone="efficientnet_lite3_sphe_FL2", FFBc="persp", exportable=True, non_negative=True, blocks={'expand': True})
        net_w, net_h = 256, 256
        resize_mode = "upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    elif model_type == "midas_v21_small_sphe_encoder":
        model = MidasNet_small_sphe(model_path, features=64, decoder="persp", backbone="efficientnet_lite3_spheFULL", FFBc="sphe", exportable=True, non_negative=True, blocks={'expand': True})
        net_w, net_h = 256, 256
        resize_mode = "upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    elif model_type == "midas_v21_small_sphe_decoder":
        model = MidasNet_small_sphe(model_path, features=64, decoder="sphe", backbone="efficientnet_lite3", FFBc="persp", exportable=True, non_negative=True, blocks={'expand': True})
        net_w, net_h = 256, 256
        resize_mode = "upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    elif model_type == "midas_v21_small_sphe_FULL":
        model = MidasNet_small_sphe(model_path, features=64, decoder="sphe", backbone="efficientnet_lite3_sphe_FULL", FFBc="sphe", exportable=True, non_negative=True, blocks={'expand': True})
        net_w, net_h = 256, 256
        resize_mode = "upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    elif model_type == "midas_v21_small_sphe_FL+decoder":
        model = MidasNet_small_sphe(model_path, features=64, decoder="sphe", backbone="efficientnet_lite3_sphe_FL", FFBc="persp", exportable=True, non_negative=True, blocks={'expand': True})
        net_w, net_h = 256, 256
        resize_mode = "upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    elif model_type == "midas_v21_small_sphe_FL2+decoder":
        model = MidasNet_small_sphe(model_path, features=64, decoder="sphe", backbone="efficientnet_lite3_sphe_FL2", FFBc="persp", exportable=True, non_negative=True, blocks={'expand': True})
        net_w, net_h = 256, 256
        resize_mode = "upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    elif model_type == "midas_v21_small_sphe_E1LL+decoder":
        model = MidasNet_small_sphe(model_path, features=64, decoder="sphe", backbone="efficientnet_lite3_sphe_FL2", FFBc="persp", exportable=True, non_negative=True, blocks={'expand': True})
        net_w, net_h = 256, 256
        resize_mode = "upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    elif model_type == "midas_v21_small_sphe_E1LL+LLres+decoder":
        model = MidasNet_small_sphe(model_path, features=64, decoder="sphe", backbone="efficientnet_lite3_sphe_FL2", FFBc="persp", exportable=True, non_negative=True, blocks={'expand': True})
        net_w, net_h = 256, 256
        resize_mode = "upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    else:
        print(f"model_type '{model_type}' not implemented, use: --model_type large")
        assert False

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    return model, transform


def pred(image_in, model, transform, optimize=True):
    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("device: %s" % device)

    model.eval()

    if optimize:
        # rand_example = torch.rand(1, 3, net_h, net_w)
        # model(rand_example)
        # traced_script_module = torch.jit.trace(model, rand_example)
        # model = traced_script_module

        if device == torch.device("cuda"):
            model = model.to(memory_format=torch.channels_last)
            model = model.half()

    model.to(device)

    # print("start processing")

    # print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))

    # input

    # img = utils.read_image(img_name)
    img = image_in
    img_input = transform({"image": img})["image"]

    # compute
    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
        if optimize and device == torch.device("cuda"):
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()
        # print(model)
        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

        # # output
        # filename = os.path.join(
        #     output_path, os.path.splitext(os.path.basename(img_name))[0]
        # )
        # utils.write_depth(filename, prediction, bits=2)

    # print("finished")
    pred = write_depth(prediction, bits=2)

    return pred


def write_depth(depth, bits=1):

    # print(depth)

    depth_min = depth.min()
    depth_max = depth.max()
    # print(depth_min)
    # print(depth_max)

    # max_val = (2**(8*bits))-1
    # print(max_val)

    if depth_max - depth_min > np.finfo("float").eps:
        out = 1 - (depth - depth_min) / (depth_max - depth_min)
        # print(out)
        # print(out.shape)
        # out = (np.clip(image_out, 0, self.SIM_CONF.maxdep) / self.SIM_CONF.maxdep)  # 0-255  0-20m 255-0m
    else:
        out = np.zeros(depth.shape, dtype=depth.type)

    # print(out)

    return out
