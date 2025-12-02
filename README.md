# CLG Volumetric Imaging & Analysis Framework

**Comprehensive Label–Guided (CLG) Volumetric Imaging Enables Accurate Single-Neuron Mapping and Network Reconstruction and Analysis**

![演示图](docs/Composite.png)   

| Zebragdm LCC Reduction Curve| Other Methods' LCC Reduction Curve|
|------|------|
| ![Zebragdm LCC Reduction Curve](docs/Zebragdm_LCC_Reduction_Curve.png) | ![Other Methods' LCC Reduction Curve](docs/gdmnc.png) |    



| CPU Version of Zebragdm| GPU Version of Zebragdm|
|------|------|
| ![图1](docs/CPU_of_Zebragdm.gif) | ![图2](docs/GPU_of_Zebragdm.gif) |   


**Zebragdm on the Corruption Dataset**
![演示图](docs/Zebragdm_on_the_Corruption_Dataset.gif)   


## 简介 (Introduction)

本仓库包含了论文 **"Comprehensive Label–Guided Volumetric Imaging Enables Accurate Single-Neuron Mapping and Network Reconstruction and Analysis"** 中使用的完整分析流程代码。

CLG 是一个整合了双通道成像（核定位结构像 + 胞质钙信号功能像）与深度学习分析的框架。它旨在解决双光子体积成像中常见的 **轴向过度计数 (Axial Overcounting)** 问题，实现高精度的单神经元提取和无偏倚的全脑网络重建。
If you want to quickly use our models and algorithms, please read the `README.md` file carefully.



## 系统流程概览 (Pipeline Overview)

整个处理流程分为以下四个主要模块：

1.  **图像预处理 (Image Preprocessing):** 结构像与功能像的去噪、解卷积与配准。
2.  **3D 结构分割 (3D Structural Segmentation):** 基于深度学习的细胞核分割。
3.  **单神经元信号提取与校准 (Signal Extraction & 3D Calibration):** 结合结构信息提取功能信号，并修正过度计数。
4.  **网络构建与分析 (Network Construction & Analysis):** 功能连接组学分析、网络拆解 (Dismantling) 与雪崩分析。

---

## 1. 图像预处理 (Image Preprocessing)

### 结构像处理 (Structural Imaging - mRuby3)
为了提升细胞核分割的准确率，我们首先对 mRuby3 通道进行稀疏解卷积和局部对比度归一化。

*   **Sparse Deconvolution (稀疏解卷积):**
    *   我们使用了 **Sparse Deconvolution** 算法来提高图像信噪比和分辨率。
    *   **External Link:** [Sparse Deconvolution MATLAB Package (Windows Source)](https://github.com/WeisongZhao/Sparse-SIM/tree/master/src_win) (Reference: Zhao et al., Nat Biotechnol 2022)
    *   **Usage in CLG:** 主要参数如下：`iterations=120`, `z_axis_continuity=1`, `image_fidelity=150`, `sparsity=6`, `deconv_iterations=8`（详见论文 Methods）。

*   **Local Contrast Normalization (局部对比度归一化):**
    *   为了应对组织深度的光强不均匀，我们实施了滑动窗口归一化。
    *   **Our Code:** `[请填入您的代码路径，例如: preprocessing/local_normalization.py]`

### 功能像处理 (Functional Imaging - GCaMP6s)
*   **Motion Correction:** 使用 **NoRMCorre** 进行刚性或非刚性运动校正。
    *   **External Link:** [NoRMCorre](https://github.com/flatironinstitute/NoRMCorre)
    *   **Usage in CLG:** 针对 512×512 图像，主要参数为 `patch_size=128`, `overlap=32`, `iterations=2`；针对 1024×1024 图像，推荐使用 `patch_size=256`, `overlap=64`, `iterations=2`（详见论文 Methods）。
    *   **Our Code:** `main/registration/run_functional_registration.m` (基于 NoRMCorre 封装的通用配准脚本)
    
*   **Denoising:** 使用自监督深度学习方法 **SUPPORT** 进行去噪。
    *   **External Link:** [SUPPORT](https://github.com/FlorentF9/SUPPORT) (或您使用的具体实现链接)
    *   **Usage:** 针对时间序列功能像进行训练和推理。

---

## 2. 3D 结构分割 (3D Structural Segmentation)

这是 CLG 框架的核心步骤之一，利用细胞核通道提供真实的 3D 神经元位置信息。

*   **Deep Learning Segmentation:** 我们使用了 **Cellpose 2** 算法。
    *   **External Link:** [Cellpose](https://github.com/MouseLand/cellpose)
    *   **Our Implementation:**
        *   我们使用预处理后的图像和人工标注数据重新训练了 Cellpose 模型。
        *   利用 Cellpose 的 3D 模式（拼接 2D 切片结果）重建完整的 3D 细胞核掩膜。
    *   **Training/Inference Script:** 
        我们使用以下命令进行模型训练：
        ```bash
        python -m cellpose --train --use_gpu --dir ./trainset --test_dir ./valset --pretrained_model cyto2 --learning_rate 0.1 --weight_decay 0.0001 --n_epochs 500 --verbose
        ```
    *   **Pre-trained Models:** Our fine-tuned Cellpose model for zebrafish. Model download link: XXX.XXX, please put the model file in the path:`main/models/clg_cellpose_model_for_zebrafish` 

---

## 3. 单神经元信号提取与校准 (Signal Extraction & 3D Calibration)

此步骤将功能信号映射到 3D 结构上，并修正轴向重复计数（即同一个细胞在不同层被多次计算）。

1.  **Registration:** 将功能像配准到结构像模板（见前文功能像处理部分的 NoRMCorre）。
2.  **3D Calibration (The "CLG" Step):**
    *   **原理:** 利用 3D 细胞核 mask 的唯一 ID，识别跨越多个成像层（Z-planes）的同一神经元。
    *   **操作:** 将属于同一个 3D ID 的多个层面的 ROI 信号进行合并（平均），从而消除冗余计数。
    *   **Our Code:** `main/extraction/step2_signal_extraction_calibration.ipynb` (Python notebook for signal extraction and CLG 3D calibration)
3.  **ΔF/F Calculation:**
    *   使用 **AllenSDK** 计算相对荧光变化率。
    *   **External Link:** [AllenSDK](https://github.com/AllenInstitute/AllenSDK)
    *   **Usage:** `allensdk.brain_observatory.dff` module.

---

## 4. 网络构建与分析 (Network Construction & Analysis)

基于校准后的单神经元活动数据，构建功能网络并进行拓扑分析。

### 网络构建
*   **Processing:** 去噪 (PCA) -> 相关性计算 (Pearson Correlation)。

If you want to compute the correlation matrix of your time-series data and choose a threshold based on the distribution of correlations, please run the code file `main/networkdismantling/corrdistributionon.py`. In this script, we provide the distribution lines for four percentiles (95 %, 90 %, 85 %, 80 %) as candidate thresholds, and you will also obtain the corresponding visualization. You can choose an appropriate threshold and correlation-value rule according to the observed distribution to construct your network.

```bash
python main/networkdismantling/corrdistributionon.py
```

Once you have selected an appropriate threshold and correlation-value rule for building your network, proceed by running the code file `main/networkdismantling/bulidyournetwork.py`. In the script, set the input and output paths for your data file, specify your chosen rule for selecting correlation values, and define the threshold you have decided on. Then execute the script. We have also included network-visualization output file in the code. You can use the generated network plot to verify whether your chosen threshold and correlation-value rule meet your expectations.

```bash
python main/networkdismantling/bulidyournetwork.py
```


*   **NetworkX:** 用于计算 Degree, Eigenvector Centrality, Communicability 等指标。
    *   **External Link:** [NetworkX](https://networkx.org/)
    *   **Analysis Script:** `main/analysis/step3_network_construction_analysis.ipynb`



### 高级网络分析
*   **Coarse-Graining:** 为了处理大规模网络，首先进行粗粒化处理。
    *   **Code link:** `(https://www.github.com/Bmunn/ICG)`
*   **Network Dismantling (GDM):**
    *   我们使用并改进了基于机器学习的图拆解算法 (**GDM**)。
    *   **Modification:** 我们扩充了训练集（包含 Watts-Strogatz 和模块化图模型）以适应生物神经网络特性。
    *   **Original Algorithm Reference:** [GDM by Grassia et al.](https://github.com/marcograssia/GDM) (Check reference [44] in paper)
    *   **Our Improved GDM Code:** `If you are satisfied with the network you have built, it’s time to start dismantling it! We provide several code files to help you dismantle your network ( `main/networkdismantling/dismantling_XXX.py `), each runnable on either CPU or GPU. Specifically, we offer dismantling strategies based on degree centrality, betweenness centrality, and a new method—`zebragdm`—that incorporates multiple optimizations on Marco Grassia et al.’s GDM framework (Machine-learning dismantling and early-warning signals of disintegration in complex systems. *Nature Communications*, 2021, 12(1): 5190). If you wish to use a dismantling method that incorporates multiple metrics, please apply the `zebragdm` model to your data, and you will need to adjust the relevant parameters in the code. Conversely, if you opt for a single-metric dismantling method, no parameter adjustments are necessary. Choose the approach that best suits your research question, and set your dismantling target value directly in the code file. The final outputs will include detailed information on the dismantled nodes and a visualization of the dismantling process.

```bash
python main/networkdismantling/dismantling_XXX.py
```

**Tip: GPU acceleration can speed up dismantling, but it incurs additional cost. If your network is small—e.g., only a few hundred nodes—using the CPU implementation is usually the better choice.**`

---

## 环境依赖 (Dependencies)

You can use the following command to quickly set up an environment with CUDA 11.8.

```bash
git clone https://github.com/PKUCHENLAB/CLG-Volumetric-Imaging-Analysis-Framework.git
```

```bash
cd main
```

```bash
micromamba env create -f environment.yml -n zebragdm
```

```bash
micromamba activate zebragdm
```





## 引用 (Citation)

如果您使用了本代码或参考了我们的方法，请引用我们的论文：

> [Authors]. Comprehensive Label–Guided Volumetric Imaging Enables Accurate Single-Neuron Mapping and Network Reconstruction and Analysis.

---

### 联系方式 (Contact)

如有疑问，请联系: [Your Email]




