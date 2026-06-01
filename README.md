This repository contains the Python code developed for the Master’s Thesis titled **Exploring the use of Deep Learning to identify Early Alzheimer’s Disease patterns using MRI**.

## Repository structure

```text
TFM_Irene_Cuellar/
│
├── scripts_ADNI/
│   ├── registration_T1_adni.py
│   ├── registration_pve_vfinal.py
│   ├── extract_axial_slices.py
│   ├── extract_20_central_coronal_slices.py
│   ├── extract_patches_inferiors.py
│   ├── extract_coronal_pve_registered.py
│   ├── image_resize_def.py
│   ├── gaussian_normalization_def.py
│   ├── SVM_ADNI_ttest.py
│   ├── ResNet50_CUvsAD.py
│   ├── ResNet50_all_inputs.py
│   └── ResNet50_grid_tuning.py
│
└── scripts_A4/
    ├── registration_T1_A4.py
    ├── extract_20_coronal.py
    ├── image_resize_coronal_slices.py
    ├── gaussian_normalization_coronal.py
    └── ResNet50_A4_def.py
```

## General workflow

The image preprocessing pipeline follows these main steps:

1. **T1 registration**  
   The original T1-weighted MRI volumes are skull-stripped and registered to a common 1 mm template.

2. **Image extraction**  
   Different 2D image inputs are extracted from the registered volumes, including axial slices, coronal slices, coronal patches and tissue maps.

3. **Resize**  
   The extracted PNG images are resized to 224 × 224 pixels to match the ResNet50 input size.

4. **Gaussian normalization**  
   A Gaussian smoothing-based intensity normalization is applied to reduce intensity differences between images.

5. **Model training and evaluation**  
   The final images or volumetric features are used to train and evaluate the SVM and ResNet50 models.

## ADNI scripts

The ADNI workflow includes both classical machine learning and deep learning experiments.

| Script | Purpose |
|---|---|
| `registration_T1_adni.py` | Performs skull stripping and registration of ADNI T1-weighted MRI scans to the 1 mm template. |
| `registration_pve_vfinal.py` | Registers PVE tissue maps to the same template using the transforms already obtained from the T1 registration. |
| `extract_axial_slices.py` | Extracts the three central axial slices from the registered T1 images. |
| `extract_20_central_coronal_slices.py` | Extracts the 20 central coronal slices from each registered T1 image. |
| `extract_patches_inferiors.py` | Extracts 64 × 64 inferior coronal patches from the three central coronal slices. |
| `extract_coronal_pve_registered.py` | Extracts the central coronal slice from the registered tissue maps. |
| `image_resize_def.py` | Resizes ADNI tissue maps, coronal slices and axial images to 224 × 224 pixels. |
| `gaussian_normalization_def.py` | Applies Gaussian-based intensity normalization to the ADNI PNG images. |
| `SVM_ADNI_ttest.py` | Trains the linear SVM baseline with volumetric features, performs 5-fold cross-validation and runs feature-level statistical tests. |
| `ResNet50_CUvsAD.py` | Trains a ResNet50 model for the CU vs AD classification task using three coronal slices as input channels. |
| `ResNet50_all_inputs.py` | Trains ResNet50 models for the selected ADNI task using different image input configurations. |
| `ResNet50_grid_tuning.py` | Performs hyperparameter tuning for the ResNet50 model using the coronal three-slice input configuration. |

### ADNI preprocessing order

```text
registration_T1_adni.py
        ↓
extract_axial_slices.py / extract_20_central_coronal_slices.py
        ↓
image_resize_def.py
        ↓
gaussian_normalization_def.py
        ↓
ResNet50_CUvsAD.py / ResNet50_all_inputs.py / ResNet50_grid_tuning.py
```

For the tissue-map input, the additional pipeline is:

```text
registration_T1_adni.py
        ↓
registration_pve_vfinal.py
        ↓
extract_coronal_pve_registered.py
        ↓
image_resize_def.py
        ↓
ResNet50_all_inputs.py
```

For the patch input, the additional patch extraction step is:

```text
registered T1 images
        ↓
central coronal slices
        ↓
extract_patches_inferiors.py
        ↓
gaussian_normalization_def.py
        ↓
ResNet50_all_inputs.py
```

## A4/LEARN scripts

The A4/LEARN workflow uses the same general preprocessing idea but is stored separately because the folder structure and metadata are different.

| Script | Purpose |
|---|---|
| `registration_T1_A4.py` | Performs skull stripping and registration of A4/LEARN T1-weighted MRI scans to the 1 mm template. |
| `extract_20_coronal.py` | Extracts the 20 central coronal slices from each registered A4/LEARN subject. |
| `image_resize_coronal_slices.py` | Resizes the A4/LEARN coronal slices to 224 × 224 pixels using resize and center crop. |
| `gaussian_normalization_coronal.py` | Applies Gaussian-based normalization to the resized A4/LEARN coronal slices. |
| `ResNet50_A4_def.py` | Trains and evaluates a ResNet50 model using three central coronal slices as input channels for the selected A4/LEARN classification task. |

### A4/LEARN preprocessing order

```text
registration_T1_A4.py
        ↓
extract_20_coronal.py
        ↓
image_resize_coronal_slices.py
        ↓
gaussian_normalization_coronal.py
        ↓
ResNet50_A4_def.py
```

## Main dependencies

The scripts were developed in Python and use the following main libraries and tools:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `tensorflow`
- `nibabel`
- `Pillow`
- `scipy`
- `PyYAML`
- `tqdm`
- `ANTsPy` / ANTs tools
- HD-BET

Some preprocessing functions are based on local scripts available in the working environment used for the project. Therefore, the registration scripts may require adapting the paths to the local installation of these tools.

## Data availability

The MRI data used in this project come from ADNI and A4/LEARN. These datasets are not included in this repository because they are subject to data access agreements and cannot be redistributed.

Only the code used for preprocessing, model training, evaluation and result generation is provided here.

## Important notes before running the scripts

The scripts contain absolute paths from the original computing environment. Before running them in another system, update the path variables at the beginning of each script, especially:

- input dataset folders
- output folders
- CSV metadata files
- template image path
- configuration files
- HD-BET, ANTs or FSL paths if needed

Some scripts include `START_IDX` and `END_IDX` variables. These were used to process subjects in blocks when running several jobs in parallel. Modify these values depending on the number of subjects that should be processed.


## Author

Irene Cuéllar Vázquez  
Master in Health Data Science (MHEDAS)
February - May 2026
