from ultralytics import YOLO
import cv2
import os
import winsound  # for alert sound on Windows
import glob

# ✅ --- CONFIGURATION ---
USE_WEBCAM = False   # 🔁 True = webcam, False = image folder
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to show detections
ALERT_ENABLED = True  # 🔔 Beep alert for no-helmet detection

# ✅ Path to your trained model
model_path = r"C:\Users\Shreya\OneDrive\Desktop\BCA AI FOLDER\BCA 5sem\Deep learning Project\helmet_number_plate_detection\models\best_helmet_model.pt"

# ✅ Load trained YOLO model
model = YOLO(model_path)

# ✅ --- MODE 1: LIVE WEBCAM DETECTION ---
if USE_WEBCAM:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Could not access webcam.")
        exit()

    print("🎥 Live Helmet Detection Started... Press 'Q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break

        # Perform detection
        results = model.predict(source=frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        annotated_frame = results[0].plot()

        # Extract detection results
        boxes = results[0].boxes
        class_names = results[0].names
        no_helmet_detected = True  # assume no helmet until found

        for box in boxes:
            cls_id = int(box.cls[0])
            label = class_names[cls_id].lower()
            if "helmet" in label:
                no_helmet_detected = False  # found helmet

        if no_helmet_detected:
            cv2.putText(
                annotated_frame,
                "⚠️ No Helmet Detected!",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3,
            )
            if ALERT_ENABLED:
                winsound.Beep(1000, 500)  # Beep (frequency, duration)

        cv2.imshow("🪖 Live Helmet Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Detection stopped. Webcam released.")

# ✅ --- MODE 2: IMAGE FOLDER DETECTION ---
else:
    # Path to your folder containing all test images
# ✅ Path to test images (replace old one)
    image_folder = r"C:\Users\Shreya\OneDrive\Desktop\BCA AI FOLDER\BCA 5sem\Deep learning Project\Number_Plate_Detection\combined_test_images"
    output_dir = r"C:\Users\Shreya\OneDrive\Desktop\BCA AI FOLDER\BCA 5sem\Deep learning Project\helmet_number_plate_detection\results\detections"
    os.makedirs(output_dir, exist_ok=True)

    # Run detection on all images in the folder
    print("🖼️ Detecting helmets on all images...")
    results = model.predict(source=image_folder, save=True, project=output_dir, name="helmet_results", conf=CONFIDENCE_THRESHOLD)

    # Locate the latest YOLO result folder
    subfolders = [f.path for f in os.scandir(output_dir) if f.is_dir()]
    if not subfolders:
        print("⚠️ No result folders found in output directory.")
        exit()

    latest_folder = max(subfolders, key=os.path.getmtime)
    print(f"✅ Results saved to: {latest_folder}")

    # Show all result images one by one
    for img_path in glob.glob(os.path.join(latest_folder, "*.jpg")):
        img = cv2.imread(img_path)
        if img is not None:
            cv2.imshow("🪖 Helmet Detection Results", img)
            key = cv2.waitKey(1000)  # show each for 1 second
            if key == ord('q'):  # press q to quit early
                break

    cv2.destroyAllWindows()
    print("✅ All detections displayed successfully.")
