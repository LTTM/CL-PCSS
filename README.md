# Continual Learning for LiDAR Semantic Segmentation: Class-Incremental and Coarse-to-Fine strategies on Sparse Data
Elena Camuffo and Simone Milani, In Proceedings of IEEE Conference of Computer Vision and Pattern Recognition Workshops (CVPRW), CLVision, 2023.
[[Paper]](https://openaccess.thecvf.com/content/CVPR2023W/CLVision/html/Camuffo_Continual_Learning_for_LiDAR_Semantic_Segmentation_Class-Incremental_and_Coarse-To-Fine_Strategies_CVPRW_2023_paper.html)

![image](https://github.com/LTTM/CL-PCSS/assets/63043735/2b841a62-4c4c-4e9a-8fba-0603e8522cb3)

![image](https://github.com/LTTM/CL-PCSS/assets/63043735/5de6982d-c704-4077-a611-3d4aed84e5d4)


Our codebase is licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).

### News 🔥
<!-- - Pretrained models released [here](https://drive.google.com/drive/folders/1fv7y1XrgEji6WWIiRMMoLcorv96-Yh6t?usp=sharing)!! -->

- **Codebase released!**

### Abstract 
During the last few years, **continual learning (CL)** strategies for image classification and segmentation have been widely investigated designing innovative solutions to tackle catastrophic forgetting, like knowledge distillation and self-inpainting. However, the application of continual learning paradigms to point clouds is still unexplored and investigation is required, especially using architectures that capture the sparsity and uneven distribution of LiDAR data. The current paper analyzes the problem of class incremental learning applied to point cloud semantic segmentation, comparing approaches and state-of-the-art architectures. To the best of our knowledge, this is the first example of class-incremental continual learning for **LiDAR point cloud semantic segmentation**. Different CL strategies were adapted to LiDAR point clouds and tested, tackling both classic fine-tuning scenarios and the Coarse-to-Fine learning paradigm. The framework has been evaluated through two different architectures on **SemanticKITTI**, obtaining results in line with state-of-the-art CL strategies and standard offline learning.

### Citation
If you find our work useful for your research, please consider citing:

```
@InProceedings{Camuffo_2023_CVPR,
    author    = {Camuffo, Elena and Milani, Simone},
    title     = {Continual Learning for LiDAR Semantic Segmentation: Class-Incremental and Coarse-To-Fine Strategies on Sparse Data},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
    pages     = {2447-2456}
}
```

## Codebase Instructions
Here above are the generic instruction to run the codebase.

#### Environment Setup 🌴
```
conda env create -f clpcss.yml
```

#### Training Parameters 🏓​

- `CL`: whether to select offline training or enable class-incremental continual learning

- `CLstep`: which learning step of `CL` you want to train on.

- `CLstrategy`: which strategy to mitigate forgetting you want to use (e.g., output knowledge distillation `okd` or feature knowledge distillation `fkd`).

- `setup`: which setup you are willing to use (e.g., Sequential, Sequential_masked, etc.)

- `pretrained_model`: if you want to use as a pretrained model, i.e., if you are training `CLstep = 1`, you want to use a pretrained on step `CLstep = 0`.

- `ckpt_file`: path of the checkpoint file of the pretrained model if `pretrained_model = True`.

- `test_name`: the name you want to give to your test.

#### Training modalities ​​🏋️​
- **Offline**: 
    ```
    python train.py --CL False [params]
    ```
- **Standard CIL Setups**: 
    ```
    python train.py --CL True --setup "Sequential" --CLstep 0 [params]
    ```
- **Knowledge Distillation (output)** $\mathcal{L}_{KD}$: 
    ```
    python train_envelope.py --CLstrategy "okd" --pretrained_model True --ckpt_file "path\to\pretrained.pt" [params]
    ```
- **Knowledge Distillation ($\ell_2$ feats)** $\mathcal{L}^{*}_{KD}$: 
    ```
    python train_envelope.py --CLstrategy "fkd" --pretrained_model True --ckpt_file "path\to\pretrained.pt"  [params]
    ```
- **Knowledge Self-Inpainting**: 
    ```
    python train_envelope.py --CLstrategy "inpaint" --pretrained_model True --ckpt_file "path\to\pretrained.pt"  [params]
    ```
- **C2F Setups**: 
    ```
    python train_c2f.py --CLstep 0 [params]
    ```

#### Testing results 🚀
```
python test.py --pretrained_model True --ckpt_file "your/path/to/model.pt" [params]
```

| Model        | Step 0       | Step 1       | Step 2       |
|:-------------|:------------:|:------------:|-------------:|
| *Baseline*                     | -            | -            | 47.2          |
| Sequential                  | 49.0            | 48.8           | 42.9        |
| Seq. Masked                | 49.0            | 22.0           | 11.0         |
| $\mathcal{L}_{KD}$ (output)               | 49.0       | 47.9     | 44.4     |
| $\mathcal{L}^{*}_{KD}$ ($\ell_2$ feats) | 49.0      | 46.8     | 41.7        |
| Inpainting                  | 49.0            | 46.5           | 39.0        |
| Coarse-to-Fine              | 86.7            | 74.6          | 47.1         |

The table shows overall results in terms of $\text{mIoU}_{0\rightarrow k}$, with $k$ as the continual learning step, on our CIL partition (above) of the [SemanticKITTI](https://www.semantic-kitti.org/) dataset using [RandLA-Net](https://github.com/QingyongHu/RandLA-Net) point based architecture.

<!-- pretrained models can be found [here](https://drive.google.com/drive/folders/1fv7y1XrgEji6WWIiRMMoLcorv96-Yh6t?usp=sharing). -->