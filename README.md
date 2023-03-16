# Citation
This repositeory contains the depth model used in our article "OMNI-CONV: Generalization of the Omnidirectional Distortion-Aware Convolutions".

```
@article{Artizzu2023,
	title        = {{OMNI-CONV: Generalization of the Omnidirectional Distortion-Aware Convolutions}},
	author       = {Artizzu, Charles-Olivier and Allibert, Guillaume and Demonceaux, Cédric},
	year         = 2023,
	journal      = {Journal of Imaging},
	volume       = 9,
	number       = 2,
	article-number = 29,
	url          = {https://www.mdpi.com/2313-433X/9/2/29},
	pubmedid     = 36826948,
	issn         = {2313-433X},
	doi          = {10.3390/jimaging9020029}
}
```

# Installation
Create the python3 env and install packages:
```
python3 -m venv MIDAS_ENV;
source MIDAS_ENV/bin/activate;
pip3 install pytorch, opencv-python, timm, (airsim)
```

# Spherical adaptation
Distortion-aware convolutions for the ENCODER are located in "blocks.py" file from line 258 to 260 and are activated by commenting the appropriate line
```
    # effnet.conv_stem = DeformConv2d_sphe(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
    # effnet.conv_stem = Conv2dSameExport_sphe(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
    effnet.conv_stem = DeformConv2d_sphe_SameExport(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
```

Distortion-aware convolutions for the DECODER are located in "midas_net_custom_sphe.py" file from line 83 to 93 and are activated by commenting the appropriate line:
```
    elif decoder == "sphe":
        self.scratch.output_conv = nn.Sequential(
            # nn.Conv2d(features, features//2, kernel_size=3, stride=1, padding=1, groups=self.groups),
            DeformConv2d_sphe(features, features//2, kernel_size=3, stride=1, padding=1, groups=self.groups),
            Interpolate(scale_factor=2, mode="bilinear"),
            # nn.Conv2d(features//2, 32, kernel_size=3, stride=1, padding=1),
            DeformConv2d_sphe(features//2, 32, kernel_size=3, stride=1, padding=1),
            self.scratch.activation,
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
```

# Evaluation
Run the evaluation of the model:
```
python3 test_pred_MIDAS.py
python3 eval_with_png.py --pred_path OUTPUT/DEPTH_TEST_512x256/midas_v21_small_sphe_E1LL+LLres+decoder/ --gt_path OUTPUT/DEPTH_TEST_512x256/INPUT/
```

