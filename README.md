# TrueScan: Medical Image Authenticity & Analysis System 🏥

TrueScan is a deep learning-powered system that detects AI-generated or manipulated knee X-rays
and classifies them for anomalies. It uses an ensemble of three state-of-the-art CV models
served through a full-stack clinical web dashboard.

---
>  🏆 **3rd Place — ML/DL Track** | State Level Inter-Collegiate Tech Exhibition at AIT, CKM 

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
## A Note on Evaluation

All three models achieve near-perfect classification on the evaluation set 
(ResNet50: 98% F1, VGG19_BN: 100% F1). These numbers are not meaningful 
benchmarks; the dataset consists of GAN-generated images with pronounced 
visual artifacts that are trivially detectable by any modern CNN and even by an untrained eye. With all honesty, the 
figures are statistically unreliable.

The value of TrueScan lies in the ensemble architecture, inference pipeline, 
and clinical tooling — not the classification numbers on this particular 
dataset. A meaningful evaluation would require diffusion-model-generated 
samples and a significantly larger test set, which remains future work.

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
**Database:** Supabase
**Deployment:** Local deployment only

---

## Run Locally

````bash
git clone https://github.com/prajwalnayaka/TrueScan.git
pip install -r requirements.txt
cd Python_Scripts
python app.py
````

**Note:** Trained model weights are not included due to file size. Download them from this [Google Drive](https://drive.google.com/drive/folders/1b9jZsx7kUbaTyNoDIdPPbtE5Ds3LdvE8?usp=sharing) and update the model paths in `test.py` accordingly.


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

---

## Screenshots
### Landing page
<img width="1918" height="970" alt="landing_page" src="https://github.com/user-attachments/assets/2e302822-d493-4b13-808b-ea66ba212046" />

### User Authentication 
<img width="1918" height="968" alt="user_auth" src="https://github.com/user-attachments/assets/9d074bb9-8f5a-49a0-aa59-3444986a3b95" />

### Dashboard
<img width="1918" height="970" alt="dashboard" src="https://github.com/user-attachments/assets/70f522e5-3f0e-479e-a560-b427aafec6da" />

### Scan Selection
<img width="1918" height="970" alt="scan_selection" src="https://github.com/user-attachments/assets/2b3dd2de-5581-4698-a22c-8a1c84064d94" />

### Analyzing Screen
<img width="1918" height="967" alt="analyzing" src="https://github.com/user-attachments/assets/5159b032-2f3c-4a1e-a170-24f1ea0fc256" />

### Analysis Page
<img width="1918" height="968" alt="anaysis_page" src="https://github.com/user-attachments/assets/9292b8b6-f6e2-4914-a7ab-0de912002eb2" />

### Report Form 
<img width="1918" height="958" alt="report_form" src="https://github.com/user-attachments/assets/fe9898a5-8ae3-4a31-8706-d6a8798724cd" />

### Report
<img width="472" height="547" alt="Report" src="https://github.com/user-attachments/assets/3a4ddef7-3bee-430d-be65-df89a8187662" />








