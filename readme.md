# CAPE Tests for External Validation of Trained Models

This repository provides a minimal working example to evaluate the performance of **CAPE** on external validation datasets. MLP models are trained to predict age and sex using features extracted from the **BIDMC** dataset, and are evaluated on features from the **PTB-XL** and **CODE15** datasets. 

Additionally, a notebook is included for training a model directly on **PTB-XL** features, demonstrating the performance of CAPE in a supervised setting.


---

## 📁 Folder Structure

```bash
.
├── data
│   ├── ECGdata
│   │   ├── ptb
│   │   │   └── ecg_labels.csv
│   │   │   └── sub_labels.csv
│   │   │   └── super_labels.csv
│   │   └── code15
│   │       └── ecg_labels.csv
│   └── DataSplits
│       └── ptb
│           └── ptb_*_split.csv
├── feats
│   ├── ptb
│   │   └── CAPE_feats_BTCSV_PTBXL.h5
│   │   └── CAPE_feats_BTCSV_IDB_PTBXL.h5
│   └── code15
│       └── CAPE_feats_BTCSV_CODE15.h5
│       └── CAPE_feats_BTCSV_IDB_CODE15.h5
├── indexes
│   └── mimic
│       └── indexes_mimic
│           └── indexes_x.h5
├── models
│   └── CAPE_mimic
├── utils
├── results
├── CAPE_training.ipynb
├── MLP_training_PTB_XL.ipynb
└── CAPE_get_feats.ipynb
```

---

## 📦 Directory and File Descriptions

- `data/`  
  Contains the ECG data for pretraining:

  - `ECGdata/`  
    Each dataset folder contains label files in `.csv` format.

  - `DataSplits/`  
    Contains `.csv` files with filenames for Train/Validation/Test splits.

- `MLP_models/`  
  Contains trained MLP models for CAPE features.

- `results/`  
  Output folder for storing results.

- `MLP_training_PTB_XL.ipynb`  
  Notebook for supervised training using PTB-XL labels and CAPE features.

- `MLP_CAPE_age_sex_test.ipynb`  
  Notebook for testing the MLP model trained for BIDMC dataset on CODE15 and PTB-XL.

- `utils.py`  
  Utility scripts and helper functions used across notebooks.

---

## 📝 Notes

- Unzip `feats.zip` and place in the `feats/` folder.
- Unzip `data.zip` and place in the `data/` folder.
- Ensure all files follow the directory structure shown above.

---

## 🔧 Dependencies

To run the notebooks, install the required Python packages:

```bash
pip install -r requirements.txt
```

> 💡 It's recommended to use a virtual environment to avoid dependency conflicts.

