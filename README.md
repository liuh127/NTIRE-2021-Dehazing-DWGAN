# DW-GAN: A discret wavelet transform GAN for NonHomogenous Image Dehazing

This is the official PyTorch implement of DW-GAN.  See more details in  [[report]]() , [[paper]](), [[certificates]]( )

## Environment:

- Ubuntu: 18.04

- CUDA Version: 11.0 
- Python 3.8

## Dependencies:

- torch==1.6.0
- torchvision==0.7.0
- NVIDIA GPU and CUDA

## Pretrained Weights & Dataset

1. Download [ImageNet pretrained weights](https://drive.google.com/file/d/1612XsgoUEx2Q3D7PPiLaEI5qZEEwVlVp/view?usp=sharing) and [Dehaze weights](https://drive.google.com/file/d/1JkeGhdNwKi_9jObamjMtMlQ_b1i8WQ3r/view?usp=sharing) and place into the folder ```./weights```.  
2. Download the [NH-HAZE](https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/) and [NH-HAZE2](https://drive.google.com/drive/folders/1jBoP1d8eSCHcPgxcWQ42RKIA2Fxo_Thw?usp=sharing) (only image pairs 1-25) dataset.

## Test

For model Inference, run following commands. Please check the test hazy image path (test.py line 12) and the output path (test.py line 13) .

```
python test.py
```

## Qualitative Results

Results on NTIRE 2021 NonHomogeneous Dehazing Challenge validation images:  

<div style="text-align: center">
<img alt="" src="/Image/validation.png" style="display: inline-block;" />
</div>

Results on NTIRE 2021 NonHomogeneous Dehazing Challenge testing images:

<div style="text-align: center">
<img alt="" src="/Image/test.png" style="display: inline-block;" />
</div>
## Acknowledgement

We thank the authors of [Res2Net](https://mmcheng.net/res2net/), [MWCNN](https://github.com/lpj0/MWCNN.git), and [KTDN](https://github.com/GlassyWu/KTDN). Part of our code is built upon their modules.

## Citation

If our work helps your research, please consider to cite our paper:

```
@article{
}
```



