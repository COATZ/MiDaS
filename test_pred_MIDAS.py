import os
# import sys
import numpy as np
# import airsim
# import PIL.Image
import cv2
import time
import torch
import pred_MIDAS

# folder_path = "./OUTPUT/DEPTH_TEST/"
# folder_path = "./OUTPUT/DEPTH_TEST_256/"
# folder_path = "./OUTPUT/DEPTH_TEST_RICOH/APPLE1/"
# folder_path = "./OUTPUT/DEPTH_TEST_RICOH/BALL1/"
# folder_path = "./OUTPUT/DEPTH_TEST_RICOH/CAR1/"
# folder_path = "./OUTPUT/DEPTH_TEST_RICOH/CAR2/"
# folder_path = "./OUTPUT/DEPTH_TEST_RICOH/CAR3/"
# folder_path = "./OUTPUT/DEPTH_TEST_RICOH/CAR4/"
# folder_path = "./OUTPUT/DEPTH_TEST_RICOH/CAR5/"
folder_path = "./OUTPUT/DEPTH_TEST_256/"
# folder_path = "./OUTPUT/DEPTH_TEST_512x256/"
# folder_path = "./OUTPUT/DEPTH_TEST_1024x512/"
# folder_path = "./OUTPUT/DEPTH_TEST_1/"
# depth_net = "midas_v21_small"
# depth_net = "midas_v21_small_sphe_decoder"
# depth_net = "midas_v21_small_sphe_encoder"
# depth_net = "midas_v21_small_sphe_FULL"
# depth_net = "midas_v21_small_sphe_FL"
# depth_net = "midas_v21_small_sphe_FL2"
# depth_net = "midas_v21_small_sphe_FL+decoder"
depth_net = "midas_v21_small_sphe_FL2+decoder"
# depth_net = "midas_v21_small_sphe_E1LL+decoder"
# depth_net = "midas_v21_small_sphe_E1LL+LLres+decoder"

depth_net_list = [depth_net]

for depth_net in depth_net_list:
    out_path = os.path.join(folder_path, depth_net)
    # out_path = os.path.join(folder_path, depth_net + "_classic_pad0")
    rgb_path = folder_path + '/INPUT/'
    # rgb_path = folder_path + '/images/'
    os.makedirs(out_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mem_0 = torch.cuda.memory_allocated(device)
    print(device)
    print("Beginning mem: ", mem_0, "Note - this may be higher than 0, which is due to PyTorch caching. Don't worry too much about this number")
    midas_model, midas_transform = pred_MIDAS.init(depth_net)
    print(midas_model)
    mem_1 = torch.cuda.memory_allocated(device)
    print("After model to device: {} Difference: {}".format(mem_1, mem_1-mem_0))

    Titer = np.array([])
    list_elt = np.array([])
    sum_time = 0

    with open(os.path.join(out_path, "net.txt"), 'w') as save_file:
        save_file.write(repr(midas_model))

    for idy, file in enumerate([f for f in sorted(os.listdir(rgb_path)) if (os.path.isfile(os.path.join(rgb_path, f)) and f.endswith('_rgb.png'))]):
        # for idy, file in enumerate([f for f in sorted(os.listdir(rgb_path)) if (os.path.isfile(os.path.join(rgb_path, f)) and f.endswith('.png'))]):
        print("PREDICTION FOR ", file)
        tmp_image = cv2.imread(os.path.join(rgb_path, file))
        # tmp_image = cv2.cvtColor(tmp_image, cv2.COLOR_BGR2RGB)
        tmp_image = np.array(tmp_image)
        T0 = time.time()

        mem_2 = torch.cuda.memory_allocated(device)
        print("Before Execution: {} Diff: {}".format(mem_2, mem_2-mem_1))
        start_time = time.time()
        tmp_depth = pred_MIDAS.pred(tmp_image / 255.0, midas_model, midas_transform)
        loc_time = 100*(time.time() - start_time)
        if idy != 0:
            sum_time = sum_time + float(loc_time)
        print("Total execution time: {0:.3f} ms".format(sum_time))
        print("Total per item: {0:.3f} ms".format(loc_time))
        mem_3 = torch.cuda.memory_allocated(device)
        print("After Execution: {} Diff: {}".format(mem_3, mem_3-mem_2))

        T1 = time.time()
        Titer = np.append(Titer, T1 - T0)
        depth_img = (tmp_depth[np.newaxis, :, :]*255).astype(np.uint8)
        depth_img = (depth_img).astype(np.uint8).swapaxes(0, 2).swapaxes(0, 1)
        depth_img = np.concatenate((depth_img, depth_img, depth_img), axis=-1)
        numpy_vertical = np.vstack((tmp_image, depth_img))
        numpy_vertical_concat = np.concatenate((tmp_image, depth_img), axis=0)

        # airsim.write_png(os.path.normpath(os.path.join(out_path, file[:-8] + '_dep_pred.png')), depth_img)
        # airsim.write_png(os.path.normpath(os.path.join(out_path, file[:-8] + '_rgbdep.png')), numpy_vertical)

        list_elt = np.append(list_elt, str(os.path.join(rgb_path, file)))

    np.savetxt(os.path.join(out_path, "list_elt.csv"), list_elt, delimiter=",", newline="\n", fmt="%s")

    print(Titer)
    print("TIME FOR PRED ", Titer[1:].mean())
    print("Total execution time: {0:.3f} ms".format(sum_time))
    print("Total per item: {0:.3f} ms".format(sum_time/998))
