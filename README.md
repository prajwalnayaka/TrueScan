
# TrueScan: Medical Image Authenticity & Analysis System 🏥



TrueScan is a computer vision system designed to verify the authenticity of medical scans (specifically knee X-rays as of now) and detect anomalies. It utilizes an ensemble of state-of-the-art Deep Learning models to differentiate between valid medical imaging and images that have been potentially manipulated using AI methods, serving the results via a user-friendly web dashboard. 

The ensemble consists of **ResNet50, VGG19_BN (Batch Normalization) and YOLOv8m-cls**. 



## Features

- **Ensemble Architecture:** Aggregates predictions from **ResNet50**, **VGG19_BN**, and **YOLOv8m-cls** using a voting mechanism to achieve high-confidence classification.
- **Automated Reporting:** Generates downloadable PDF reports with prediction confidence and patient details.
- **Web Dashboard:** A full-stack Flask application with user (doctor) authentication, user access management, scan image analysis and report generation.



## Run it locally

If you want this project on your local machine

```bash
 git clone https://github.com/prajwalnayaka/TrueScan.git
 pip install -r requirements.txt
 cd Python_Scripts
 python app.py OR flask run
```

## Authors

- Prajwal Nayaka T ([GitHub](https://github.com/prajwalnayaka))

    - Trained the Core ML models (ResNet50, VGG19_BN, YOLOv8m-cls).
    - Engineered the Ensemble Voting Mechanism and Inference Pipeline.
    - Developed the Report Generation module.
    - Integrated above mentioned features into the Flask API.
    Note: Training scripts located in /Training.

- Pragya MV ([GitHub](https://github.com/pragyamv))

    - Designed and developed frontend files along with styling.
    - Initialized the database structure.
    - Built the baseline Flask API.
