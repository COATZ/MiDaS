import OMNI_DRL.envs.pred_MIDAS as pred_MIDAS

import os
import numpy as np
import airsim
# import PIL.Image
import cv2
import time


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred)**2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return silog, log10, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3


# rgb_path = "./OUTPUT/DEPTH_TEST/"
rgb_path = "./OUTPUT/DEPTH_TEST_256_2/"
# rgb_path = "./OUTPUT/DEPTH_TEST_1000/"
depth_net = "midas_v21_small"
# depth_net = "midas_v21_small_sphe_E1LL+decoder"
out_path = os.path.join(rgb_path, depth_net)
os.makedirs(out_path, exist_ok=True)

midas_model, midas_transform = pred_MIDAS.init(depth_net)

print(midas_model)

Titer = np.array([])

for idy, file in enumerate([f for f in sorted(os.listdir(rgb_path)) if (os.path.isfile(os.path.join(rgb_path, f)) and f.endswith('.png'))]):
    print("PREDICTION FOR ", file)
    # tmp_image = PIL.Image.open(os.path.join(rgb_path, file))
    tmp_image = cv2.imread(os.path.join(rgb_path, file))
    # tmp_image = cv2.cvtColor(tmp_image, cv2.COLOR_BGR2RGB)
    tmp_image = np.array(tmp_image)
    T0 = time.time()
    tmp_depth = pred_MIDAS.pred(tmp_image / 255.0, midas_model, midas_transform)
    T1 = time.time()
    Titer = np.append(Titer, T1 - T0)
    depth_img = (tmp_depth[np.newaxis, :, :]*255).astype(np.uint8)
    depth_img = (depth_img).astype(np.uint8).swapaxes(0, 2).swapaxes(0, 1)

    # print(depth_img.shape)
    # print(tmp_image.shape)

    # airsim.write_png(os.path.normpath(os.path.join(out_path, file)), depth_img)

    depth_img = np.concatenate((depth_img, depth_img, depth_img), axis=-1)
    # print(depth_img.shape)
    numpy_vertical = np.vstack((tmp_image, depth_img))
    numpy_vertical_concat = np.concatenate((tmp_image, depth_img), axis=0)

    # cv2.imwrite(os.path.normpath(os.path.join(out_path, file + '_3')), tmp_image)

    airsim.write_png(os.path.normpath(os.path.join(out_path, file + '_3')), depth_img)

    airsim.write_png(os.path.normpath(os.path.join(out_path, file)), numpy_vertical)

print(Titer)
print("TIME FOR PRED ", Titer[1:].mean())
