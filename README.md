# Dissertation Codebase: Kinematic Analysis of Bicycle IMU Data

This repository contains the complete end-to-end data engineering and machine learning pipeline for my final year dissertation. The project aims to classify bicycle incidents using kinematic IMU sensor data (derived from the SimRa dataset) via a novel 3-stage cascaded machine learning architecture.

## Using the Project

To start, ensure you have Python installed. It is highly recommended to create a virtual environment to manage dependencies. This can be done using the following commands in the root directory:

```bash
# Create the virtual environment
python3 -m venv venv/

# Activate the virtual environment (Mac/Linux)
source venv/bin/activate

# Activate the virtual environment (Windows)
venv\Scripts\activate

# Install project dependencies
pip install -r requirements.txt
```
From here, the entire pipeline from start to finish can be executed by running:

```bash
python auto_pipeline.py
```

Some steps will be skipped due to functional data already being included.
If you wish to run a specific section you can do this by running a command like:
```bash
python3 scripts/*.py
```
where * is the file you want to run out of the following (within the script diretory):
- run_prepare.py
- run_preprocess.py
- run_segment.py 
- run_features.py
- run_stage1.py
- run_stage2.py
- run_stage3.py