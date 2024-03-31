# EEG Data Analysis Project

Ensure that all the file are contained in the same local directory

## Installation

### Prerequisites

- Python 3.8 or newer

### Required Libraries

To install the required libraries, run the following command in your terminal:

```bash
pip install numpy mne torch

## Downloading the Dataset

Ensure that the dataset is downloaded in the same local directory of the project files.
You can download the dataset from OpenNeuro using one of the following methods:

### Browser Download

Select a local directory to save the dataset.
Grant permission to OpenNeuro to read and write into this directory.
Download will run in the background. Please leave the site open while downloading.
A notification will appear when the download is complete.

### Download from S3 with AWS CLI
This method is suitable for larger datasets or unstable connections:

```bash
aws s3 sync --no-sign-request s3://openneuro.org/ds004809 ds004809-download/

### Download with Node.js
Using @openneuro/cli, download the dataset from the command line:

```bash
openneuro download --snapshot 1.0.0 ds004809 ds004809-download/

This will download to ds004809-download/ in the current directory. If your download is interrupted and you need to retry, rerun the command to resume the download.

Download with DataLad
Public datasets can also be downloaded with DataLad from GitHub:

```bash
datalad install https://github.com/OpenNeuroDatasets/ds004809.git

### Download with a Shell Script
For environments where installing additional software is not feasible, a curl-based script is provided. Replace the placeholder with the actual script content or link to the script.

## Preprocessing
Before running the preprocessing script, ensure that your dataset is downloaded in the same directory of the project files.

Run the preprocessing script with the following command:

```bash
python EEGPreprocessing.py

Training
After preprocessing, you can train the model using the following command:

```bash
python EEGFormer.py

Ensure that the preprocessing script has been executed and the preprocessed data is available for the training script to use.