from ultralytics import YOLO
import cv2
import os

# -------------------------
# ✅ Load Both Models
# -------------------------
helmet_model_path = r"runs_helmet/yolov8n_helmet/weights/best.pt"
number_model_path = r"c:\Users\Shreya\OneDrive\Desktop\BCA AI FOLDER\BCA 5sem\Deep learning Project\Number_Plate_Detection\runs_numberplate\small_numberplate\weights\best.pt"

helmet_model = YOLO(helmet_model_path)
number_model = YOLO(number_model_path)

# -------------------------
# ✅ Print helmet class names (for debugging)
# -------------------------
print("\n🧠 Helmet Model Class Names:")
print(helmet_model.names)
print("--------------------------------------------------\n")

# -------------------------
# ✅ Input folder for combined test images
# -------------------------
test_images_folder = r"C:\Users\Shreya\OneDrive\Desktop\BCA AI FOLDER\BCA 5sem\Deep learning Project\Number_Plate_Detection\combined_test_images"
output_folder = r"C:\Users\Shreya\OneDrive\Desktop\BCA AI FOLDER\BCA 5sem\Deep learning Project\helmet_number_plate_detection\results_combined"

os.makedirs(output_folder, exist_ok=True)

# -------------------------
# ✅ Define Colors
# -------------------------
COLOR_HELMET = (0, 255, 0)        # Green for helmet
COLOR_NO_HELMET = (0, 0, 255)     # Red for no helmet
COLOR_PLATE = (255, 255, 0)       # Yellow for number plate

# -------------------------
# ✅ Process Each Image
# -------------------------
for img_name in os.listdir(test_images_folder):
    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(test_images_folder, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"⚠️ Skipping unreadable image: {img_name}")
            continue

        # -------------------------
        # 🪖 Helmet Detection
        # -------------------------
        helmet_results = helmet_model(img, conf=0.5, verbose=False)

        for result in helmet_results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Get label name and normalize
                label = helmet_model.names[cls].lower()
                print(f"Detected Helmet Label in {img_name}: {label}")

                # ✅ Smart color detection logic
                if cls in [1] or "no" in label or "without" in label or "not" in label:
                    color = COLOR_NO_HELMET  # Red for no helmet
                else:
                    color = COLOR_HELMET  # Green for helmet

                # Draw bounding box & label
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"{helmet_model.names[cls]} ({conf:.2f})",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # -------------------------
        # 🚗 Number Plate Detection
        # -------------------------
        plate_results = number_model(img, conf=0.5, verbose=False)
        for result in plate_results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_PLATE, 2)
                cv2.putText(img, f"Number Plate ({conf:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_PLATE, 2)

        # -------------------------
        # 💾 Save Output Image
        # -------------------------
        output_path = os.path.join(output_folder, f"detected_{img_name}")
        cv2.imwrite(output_path, img)
        print(f"✅ Processed: {img_name} → {output_path}")

print("\n🎯 Combined Detection Completed!")
print(f"📁 Output saved to: {output_folder}")
