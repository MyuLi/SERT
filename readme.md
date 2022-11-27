# Spectral Enhanced Rectangle Transformer for Hyperspectral Image Denoising




<hr />

> **Abstract:**  Denoising is a crucial step for Hyperspectral image (HSI) applications. Though witnessing the great power of deep learning, existing HSI denoising methods suffer from limitations in capturing the non-local self-similarity. Transformers have shown potential in capturing long-range dependencies, but few attempts have been made with specifically designed Transformer to model the spatial and spectral correlation in HSIs. In this paper, we address these issues by proposing a spectral enhanced rectangle Transformer, driving it to explore the non-local spatial similarity and global spectral low-rank property of HSIs. For the former, we exploit the rectangle self-attention horizontally and vertically to capture the non-local similarity in the spatial domain. For the latter, we design a spectral enhancement module that is capable
of extracting global underlying low-rank property of spatial-spectral cubes to suppress noise, while enabling the interactions among non-overlapping spatial rectangles. Extensive experiments have been conducted on both synthetic noisy HSIs and real noisy HSIs, showing the effectiveness of our proposed method in terms of both objective metric and subjective visual quality.
<hr />

## Network Architecture

<img src = "figs/overall.png"> 

## Contents
1. [Models](#Models)
1. [Datasets](#Datasets)
1. [Training and Testing](#Training)
1. [Results](#Results)

<a id="Models"></a> 

## Models


|  Task   | Method  | Params (M)  | Dataset  |        Model Zoo                           |             
| :-----: | :------ | :--------: | :--------: | :----------------------------------------------------------: | 
|   Simulated Gaussian noise    | SERT  |   1.91    | ICVL | [Google Drive](https://drive.google.com/file/d/1Wsqyh66JRwiwn2FH-xxG-fdwmgBACBq9/view?usp=share_link) |
|   Simulated Complex noise     | SERT   |   1.91    | Urban100 |   [Google Drive](https://drive.google.com/file/d/1Pivpngcn5JkNzM1GZWq70M2mSuyX9Q3f/view?usp=share_link) | 
|   Real Noise | SERT |   8.00    | Urban | [Google Drive](https://drive.google.com/file/d/1r7VtOcRUPo9Tfjwy39i3g59Vi4aoI6K_/view?usp=share_link) | 
| Real Noise| SERT     |   1.90    |   Realistic dataset   | [Google Drive](https://drive.google.com/file/d/1zuaphGGw52FUBZ5fsYbHd4la88p7LoD7/view?usp=share_link) | 
<a id="Datasets"></a>

## Datasets

### ICVL 
* The entire ICVL dataset download link: https://icvl.cs.bgu.ac.il/hyperspectral/
1. split the entire dataset into training samples, testing samples and validating samples. The files used in training are listed in utility/icvl_train_list.txt.
2. generate lmdb dataset for training

```
python utility/lmdb_data.py
```

3. download the testing data from BaiduDisk or generate them by yourself through

```
python utility/mat_data.py
```

### Realistic Dataset
* Please refer to [[github-link]](https://github.com/ColinTaoZhang/HSIDwRD) for "Hyperspectral Image Denoising with Realistic Data in ICCV, 2021" to download the dataset

### Urban dataset
* The training dataset are from link: https://apex-esa.org/. The origin Urban dataset are from link:  https://rslab.ut.ac.ir/data.

1. Run the create_big_apex_dataset() funtion in utility/mat_data.py to generate training samples.

2. Run the createDCmall() function in utility/lmdb_data.py to generate training lmdb dataset.


<a id="Training"></a>

## Training and Testing
### ICVL Dataset
```
#for gaussian noise
#----training----
python hside_simu.py -a sert_base -p sert_base_gaussian

#----testing----
python hside_simu_test.py -a sert_base -p sert_base_gaussian_test -r -rp checkpoints/icvl_gaussian.pth --test-dir /icvl_noise_50/512_50
```

```
#for comlpex noise
python hside_simu_test.py -a sert_base -p sert_base_complex_test -r -rp checkpoints/icvl_complex.pth --test-dir  /icvl_noise_50/512_mix
```

### Urban Dataset
```
#----training----
python hside_urban.py -a sert_urban -p sert_urban 

#----testing----
python hside_urban_test.py -a sert_urban -p sert_urban_test -r -rp ./checkpoints/real_urban.pth
```

### Realistic Dataset
```
#----training----
python hside_real.py -a sert_real -p sert_real

#----testing----
python hside_real_test.py -a sert_real -p sert_real_test -r -rp ./checkpoints/real_realistic.pth
```


<a id="Results"></a>

## Results

<details>
<summary><strong>Denoising on Random noise (ICVL)</strong> (click to expand) </summary>
<img src = "figs/table1.png"> 
</details>

<details>
<summary><strong>Denoising on Complex noise (ICVL)</strong> (click to expand) </summary>
<img src = "figs/table2.png"> 
<img src = "figs/icvl.png"> 

</details>

<details>
<summary><strong>Denoising on Realistic noise </strong> (click to expand) </summary>
<img src = "figs/real_table.png"> 
<img src = "figs/real.png"> 
</details>

<details>
<summary><strong>Denoising on Urban dataset</strong> (click to expand) </summary>

<img src = "figs/urban.png"> 
</details>