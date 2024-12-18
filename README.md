# image_super_resolution

A Deep Convolutional Neural Network for performing super resolution

<img src="https://github.com/user-attachments/assets/c1c18e7c-b2e7-46e9-8611-f5bf94257485" alt="Super Resolution Nework Architecture" width="800" height="400"/>

## Super-Resolution Network Architecture
 The following Network consists of multiple residual blocks. Before
 the first block, an initial feature extraction module is introduced to shape the feature maps. In this module, the first
 convolutional layer generates 512-channel feature maps, followed by two stacked convolutional layers, each producing
 128-channel feature maps. All convolutional layers in this module use a kernel size of 3 × 3.
 The network comprises 12 blocks, with each block producing an upsampled image at one of three possible resolutions
 (8x, 4x, 2x). Additionally, each block generates a downsampled residual, which contains information not fully recovered
 by the preceding block. I've taken inspiration from the following paper [Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution](https://arxiv.org/abs/1704.03915)

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

