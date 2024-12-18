# image_super_resolution

A Deep Convolutional Neural Network for performing super resolution

<img src="https://github.com/user-attachments/assets/c1c18e7c-b2e7-46e9-8611-f5bf94257485" alt="Super Resolution Nework Architecture" width="800" height="400"/>

## Super-Resolution Network Architecture
 The network consists of multiple residual blocks. Before the first block, an initial feature extraction module is introduced to shape the feature maps. This module begins with a convolutional layer that generates 512-channel feature maps, followed by two stacked convolutional layers, each producing 128-channel feature maps. All convolutional layers in this module use a kernel size of 3 × 3.

The network is composed of 12 blocks, with each block generating an upsampled image at one of three possible resolutions (8x, 4x, or 2x). Additionally, each block produces a downsampled residual, which contains information not fully recovered by the preceding block. The design is inspired by the paper  [Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution](https://arxiv.org/abs/1704.03915),  where the authors demonstrate how adopting multi-scale resolutions can significantly improve performance. Drawing from this idea, I experimented with a similar approach for my machine learning exam.

Each residual block is structured with two main flows. The first flow aims to minimize the residual between the low-resolution input image and the downsampled version. This helps recover information that may not have been fully captured by the earlier layers, enabling deeper layers to better learn the mapping. The second flow consists of upsampled maps that are progressively concatenated and then compressed through a 3x3 kernel convolution to match the original number of feature maps. This process forces the network to learn useful channels, progressively recovering finer details.

After each compression, the feature maps are further compressed to propagate the information deeper into the network.



<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         image_super_resolution and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── image_super_resolution   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes image_super_resolution a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

