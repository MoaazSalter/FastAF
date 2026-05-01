from ultralytics import YOLO

MODEL_PATH = r"model_development\model_training\model_4_yolo11m_datasetv4\best.pt"
IMAGES_FOLDER_PATH = r"drug_database\valid\images"
OUTPUT_FOLDER_PATH = r"drug_database\valid\output_images"

model = YOLO(MODEL_PATH)  
results = model.predict(
    source=IMAGES_FOLDER_PATH,
    save=True,
    project=OUTPUT_FOLDER_PATH,
    exist_ok=True)