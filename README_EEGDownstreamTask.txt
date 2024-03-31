# EEG Data Analysis Project

Ensure that all the file are contained in the same local directory

## Installation

### Prerequisites

- Python 3.8 or newer

### Required Libraries

To install the required libraries, run the following command in your terminal:

```bash
pip install numpy mne torch tqdm

## Preprocessing
Before running the preprocessing script, ensure that your dataset is downloaded in the same directory of the project files.

Run the preprocessing script with the following command:

```bash
python EEGLabelProcessing.py

And

```bash
python EEGDownstreamPreprocessing.py



## Fine-Tuning
After preprocessing, you can finetune the model on the downstream task using the following command:

```bash
python EEGDownstreamTask.py

Ensure that the training of the model  is done before of this, and the model is saved in the used path.