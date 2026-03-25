from ultralytics import YOLO
import os
import textwrap

# ✅ Base path to your project data folder
BASE = r"C:\Users\Shreya\OneDrive\Desktop\BCA AI FOLDER\BCA 5sem\Deep learning Project\helmet_number_plate_detection\data"
DATA_DIR = os.path.join(BASE, "Helmet_Detection", "yolo_split")

# ✅ Save the YAML inside Helmet_Detection folder
yaml_path = os.path.join(BASE, "Helmet_Detection", "helmet_data.yaml")

# ✅ Create the YAML content dynamically
yaml_text = textwrap.dedent(f"""
path: {os.path.dirname(DATA_DIR).replace('\\', '/')}
train: yolo_split/images/train
val: yolo_split/images/val
names:
  0: helmet
""").strip()

# ✅ Write the YAML configuration file
with open(yaml_path, "w") as f:
    f.write(yaml_text)

print(f"✅ YAML file saved at: {yaml_path}\n")

# ✅ Initialize YOLOv8 model (nano version for faster training)
model = YOLO("yolov8n.pt")

# ✅ Train the model
model.train(
    data=yaml_path,                    # Path to YAML file
    imgsz=640,                         # Image size
    epochs=30,                         # You can increase later for better accuracy
    batch=8,                           # Depends on GPU/CPU memory
    project="runs_helmet",             # Output directory for results
    name="yolov8n_helmet",             # Experiment name
    pretrained=True                    # Use pretrained weights (fine-tuning)
)

# ✅ Export path of best weights after training
best = model.ckpt_path if hasattr(model, "ckpt_path") else None
print("\n🎉 Training complete!")
print("✅ Best model saved at: runs_helmet/yolov8n_helmet/weights/best.pt")
