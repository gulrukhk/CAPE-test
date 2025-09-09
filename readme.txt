# CAPE Pretraining on MIMIC ECG Dataset

This repository contains a small working example demonstrating **CAPE** performance with a supervised training using the **PTB-XL** dataset. AN MLP model is trained to predict age amd sex for the BIDMC dataset. The MLP is tested on features from CODE15 and PTB-XL datasets

---

## 📁 Folder Structure

```bash
.
├── data
│   ├── ECGdata
│   │   ├── ptb
│   │   |    └── ecg_labels.csv
│   │   |    └── sub_labels.csv
│   │   |    └── super_labels.csv
│   │   └── code15
│   │       └── ecg_labels.csv
│   └── DataSplits
│       └── ptb
│           └── ptb_*_split.csv
├── feats
│   └── ptb
│       └── CAPE_feats_BTCSV_PTBXL.h5
│       └── CAPE_feats_BTCSV_IDB_PTBXL.h5
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

📦 Directory and File Descriptions

    data/
    Contains the ECG data for pretraining:
        ECGdata/            
        The folder for each dataset contains:
            Labels as .csv files.
        DataSplits/
        The folder for each dataset contains:
            csv files with filenames for Train/validation/test splits 

    MLP_models/
    Contains trained MLP models for CAPE features.
    
    results/
    Output folder for storing results.

    MLP_training_PTB_XL.ipynb
    Notebook for supervised training using PTB-XL labels and features extracted from the pretrained CAPE model.
    
    MLP_CAPE_age_sex_test.ipynb
    Notebook for testing the MLP model trained for BIDMC dataset on the external cohorts of CODE15 and PTB-XL.
    
    utils.py
    Utility scripts and helper functions used across notebooks.

📝 Notes
    Unzip the feats.zip and save in feats folder
    Unzip the data.zip and save in data folder
    Make sure all required data and files are placed correctly in the folder structure shown above.

🔧 Dependencies

To run the notebooks, install the required Python packages:

pip install -r requirements.txt

    It's recommended to use a virtual environment to avoid dependency conflicts.
