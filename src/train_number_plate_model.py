from ultralytics import YOLO
import os

# ✅ Path to your last checkpoint (where training stopped previously)
checkpoint_path = r"C:\Users\Shreya\OneDrive\Desktop\BCA AI FOLDER\BCA 5sem\Deep learning Project\Number_Plate_Detection\runs_numberplate\small_numberplate\weights\last.pt"

# ✅ Path to your dataset YAML file
data_yaml = r"C:\Users\Shreya\OneDrive\Desktop\BCA AI FOLDER\BCA 5sem\Deep learning Project\Number_Plate_Detection\small_numberplate_data.yaml"

# ✅ Output directory for saving continued training results
output_dir = r"C:\Users\Shreya\OneDrive\Desktop\BCA AI FOLDER\BCA 5sem\Deep learning Project\Number_Plate_Detection\runs_numberplate"

# Make sure output directory exists
os.makedirs(output_dir, exist_ok=True)

# ✅ Load the model from the last checkpoint
model = YOLO(checkpoint_path)

# ✅ Resume training from the checkpoint
model.train(
    data=data_yaml,           # dataset path
    epochs=20,                # number of epochs to continue
    batch=8,                  # smaller batch size for CPU
    imgsz=640,                # image size
    project=output_dir,       # where results will be saved
    name="small_numberplate", # name of the training run
    device="cpu",             # since your system doesn’t have CUDA
    resume=True               # resume from last.pt checkpoint
)

print("✅ Training resumed successfully! Check 'runs_numberplate/small_numberplate' for progress and results.")
