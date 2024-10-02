## XFeat: Accelerated Features for Lightweight Image Matching

## Installation
XFeat has minimal dependencies, only relying on torch. Also, XFeat does not need a GPU for real-time sparse inference (vanilla pytorch w/o any special optimization), unless you run it on high-res images. If you want to run the real-time matching demo, you will also need OpenCV.
We recommend using conda, but you can use any virtualenv of your choice.
If you use conda, just create a new env with:
```bash
git clone https://github.com/verlab/accelerated_features.git
cd accelerated_features

#Create conda env
conda create -n xfeat python=3.8
conda activate xfeat
```

Then, install [pytorch (>=1.10)](https://pytorch.org/get-started/previous-versions/) and then the rest of depencencies in case you want to run the demos:
```bash

conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
#or
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118

#Install dependencies for the demo
pip install opencv-contrib-python tqdm
```


## Real-time Demo
To demonstrate the capabilities of XFeat, we provide a real-time matching demo with Homography registration. Currently, you can experiment with XFeat, ORB and SIFT. You will need a working webcam. To run the demo and show the possible input flags, please run:
```bash
python3 realtime_demo.py -h
```

Don't forget to press 's' to set a desired reference image. Notice that the demo only works correctly for planar scenes and rotation-only motion, because we're using a homography model.

If you want to run the demo with XFeat, please run:
```bash
python3 realtime_demo.py --method XFeat
```


## XFeat with LightGlue
We have trained a lighter version of LightGlue (LighterGlue). It has fewer parameters and is approximately three times faster than the original LightGlue. Special thanks to the developers of the [GlueFactory](https://github.com/cvg/glue-factory) library, which enabled us to train this version of LightGlue with XFeat.
Below, we compare the original SP + LG using the [GlueFactory](https://github.com/cvg/glue-factory) evaluation script on MegaDepth-1500.
Please follow the example to test on your own images:  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/verlab/accelerated_features/blob/main/notebooks/xfeat%2Blg_torch_hub.ipynb)

Metrics (AUC @ 5 / 10 / 20)
| Setup           | Max Dimension | Keypoints | XFeat + LighterGlue           | SuperPoint + LightGlue (Official) |
|-----------------|---------------|-----------|-------------------------------|-----------------------------------|
| **Fast**  | 640           | 1300      | 0.444 / 0.610 / 0.746       | 0.469 / 0.633 / 0.762          |
| **Accurate** | 1024          | 4096      | 0.564 / 0.710 / 0.819       | 0.591 / 0.738 / 0.841            |

## Contributing
Contributions to XFeat are welcome! 
Currently, it would be nice to have an export script to efficient deployment engines such as TensorRT and ONNX. Also, it would be cool to train other lightweight learned matchers on top of XFeat local features.

## Citation
If you find this code useful for your research, please cite the paper:

```bibtex
@INPROCEEDINGS{potje2024cvpr,
  author={Guilherme {Potje} and Felipe {Cadar} and Andre {Araujo} and Renato {Martins} and Erickson R. {Nascimento}},
  booktitle={2024 IEEE / CVF Computer Vision and Pattern Recognition (CVPR)}, 
  title={XFeat: Accelerated Features for Lightweight Image Matching}, 
  year={2024}}
```

## License
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

## Acknowledgements
- We thank the agencies CAPES, CNPq, and Google for funding different parts of this work.
- We thank the developers of Kornia for the [kornia library](https://github.com/kornia/kornia)!

**VeRLab:** Laboratory of Computer Vison and Robotics https://www.verlab.dcc.ufmg.br
<br>
<img align="left" width="auto" height="50" src="./figs/ufmg.png">
<img align="right" width="auto" height="50" src="./figs/verlab.png">
<br/>
