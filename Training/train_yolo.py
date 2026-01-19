from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('yolov8m-cls.pt')
    results = model.train(data = r'D:\Medical_Deepfake_2\dataset_classification',epochs=100,imgsz=224,batch=8,seed=42)