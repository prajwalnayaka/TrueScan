# TrueScan: Medical Image Authenticity & Analysis System 🏥

TrueScan is a deep learning-powered system that detects AI-generated or manipulated knee X-rays
and classifies them for anomalies. It uses an ensemble of three state-of-the-art CV models
served through a full-stack clinical web dashboard.


---

## The Problem

AI-generated medical images are increasingly indistinguishable from real scans. A radiologist or
clinician relying on a manipulated X-ray for diagnosis could cause serious harm. TrueScan acts as
a verification layer by flagging images that are likely synthetic or tampered with before they
enter the diagnostic pipeline.

---

## How It Works

Rather than relying on a single model, TrueScan uses an ensemble of three architecturally
distinct networks. Each model votes on whether the image is authentic or AI-generated, and the
majority vote determines the final prediction. This approach reduces the risk of any single
model's blind spots affecting the outcome.

| Model | Architecture Type | Role |
|---|---|---|
| ResNet50 | Residual CNN | Deep feature extraction |
| VGG19_BN | Classical deep CNN (Batch Norm) | Texture & pattern detection |
| YOLOv8m-cls | YOLO classification head | Fast, generalized classification |

**Ensemble voting:** A prediction is only marked as confident when at least 2 of 3 models agree.

---

## Datasets

1. Real Knee Scans: [Knee Osteoarthritis Severity Grading Dataset](https://data.mendeley.com/datasets/56rmx5bjcr/1)
2. Fake Knee Scans: [Synthetic (DeepFake) Knee Osteoarthritis X-ray Images from GANs](https://data.mendeley.com/datasets/fyybnjkw7v/3)

---

## Features

- **Ensemble inference** — ResNet50 + VGG19_BN + YOLOv8m-cls with majority voting
- **Doctor authentication** — role-based login and access management
- **Automated PDF reports** — downloadable reports with prediction confidence and patient details
- **Full-stack dashboard** — Flask backend with a clean clinical UI

---

## Tech Stack

**ML:** PyTorch, Ultralytics YOLOv8, ONNX  
**Backend:** Flask, Python  
**Frontend:** HTML/CSS/JS  
**Database:** [SQLite / PostgreSQL]  
**Deployment:** [Local / Hosted URL if available]

---

## Run Locally

````bash
git clone https://github.com/prajwalnayaka/TrueScan.git
pip install -r requirements.txt
cd Python_Scripts
python app.py
````

**Note:** Trained model weights are not included due to file size. Download from [Google Drive](https://drive.google.com/drive/folders/1b9jZsx7kUbaTyNoDIdPPbtE5Ds3LdvE8?usp=sharing) and update the model paths in `test.py` accordingly.


---

## Contributors

- **Prajwal Nayaka** T ([GitHub](https://github.com/prajwalnayaka))

    - Trained the Core ML models (ResNet50, VGG19_BN, YOLOv8m-cls).
    - Developed the Ensemble Voting Mechanism and Inference Pipeline.
    - Developed the Report Generation module.
    - Integrated above mentioned features into the Flask API.  

- **Pragya MV** ([GitHub](https://github.com/pragyamv))

    - Designed and developed frontend files along with styling.
    - Initialized the database structure.
    - Built the organized Flask API.
---

## What I'd Improve Next

- Train on a larger, more diverse dataset across multiple human anatomical regions
- Add Grad-CAM visualizations so clinicians can see which regions triggered the prediction
- Explore diffusion-model-generated fakes, which are harder to detect than GAN-generated ones
