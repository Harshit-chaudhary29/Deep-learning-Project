import cv2
import os
from ultralytics import YOLO
import winsound

# -------------------------
# ✅ CONFIGURATION
# -------------------------
CONF_THRESHOLD = 0.5
ALERT_ENABLED = True
output_folder = r"C:\Users\Shreya\OneDrive\Desktop\BCA AI FOLDER\BCA 5sem\Deep learning Project\helmet_number_plate_detection\results_combined"
os.makedirs(output_folder, exist_ok=True)

# -------------------------
# ✅ Load Models
# -------------------------
helmet_model = YOLO("runs_helmet/yolov8n_helmet/weights/best.pt")
number_model = YOLO(r"c:\Users\Shreya\OneDrive\Desktop\BCA AI FOLDER\BCA 5sem\Deep learning Project\Number_Plate_Detection\runs_numberplate\small_numberplate\weights\best.pt")

COLOR_HELMET = (0, 255, 0)
COLOR_NO_HELMET = (0, 0, 255)
COLOR_PLATE = (255, 255, 0)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot access webcam.")
    exit()

# 🎬 Video Writer Setup (to save output video)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video_path = os.path.join(output_folder, "live_detection_output.mp4")
out = None

print("🎥 Live Combined Detection Started... Press 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if out is None:
        h, w = frame.shape[:2]
        out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (w, h))  # 20 FPS

    no_helmet_found = False

    # 🪖 Helmet Detection
    helmet_results = helmet_model(frame, conf=CONF_THRESHOLD, verbose=False)
    for result in helmet_results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = helmet_model.names[cls].lower()

            if cls in [1] or "no" in label or "without" in label or "not" in label:
                color = COLOR_NO_HELMET
                no_helmet_found = True
            else:
                color = COLOR_HELMET

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{helmet_model.names[cls]} ({conf:.2f})",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 🚗 Number Plate Detection
    plate_results = number_model(frame, conf=CONF_THRESHOLD, verbose=False)
    for result in plate_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PLATE, 2)
            cv2.putText(frame, f"Number Plate ({conf:.2f})",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_PLATE, 2)

    # ⚠️ Violation Alert
    if no_helmet_found:
        cv2.putText(frame, "⚠️ No Helmet Detected!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        if ALERT_ENABLED:
            winsound.Beep(1000, 500)

    # 🖥️ Show + Save Frame
    cv2.imshow("🪖 Helmet & 🚗 Number Plate Detection", frame)
    out.write(frame)  # Save to video

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ Release everything
cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
print(f"👋 Live Detection Stopped. Video saved at: {output_video_path}")
