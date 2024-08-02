# Continual Learning for LiDAR Semantic Segmentation: Class-Incremental and Coarse-to-Fine Strategies on Sparse Data
Elena Camuffo and Simone Milani, In Proceedings of IEEE Conference of Computer Vision and Pattern Recognition Workshops (CVPRW), CLVision, 2023.
[[Paper]](https://openaccess.thecvf.com/content/CVPR2023W/CLVision/html/Camuffo_Continual_Learning_for_LiDAR_Semantic_Segmentation_Class-Incremental_and_Coarse-To-Fine_Strategies_CVPRW_2023_paper.html)

![image](https://github.com/LTTM/CL-PCSS/assets/63043735/2b841a62-4c4c-4e9a-8fba-0603e8522cb3)

![image](https://github.com/LTTM/CL-PCSS/assets/63043735/5de6982d-c704-4077-a611-3d4aed84e5d4)


### News 🔥
Aug 2024 - **Codebase released!**

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

