


from ultralytics import YOLO
import cv2
import numpy as np

# models
person_model = YOLO("yolov8n.pt")  # For detecting people
zebra_model = YOLO(r"zebra_crossing_yolov8n.pt")  # trained model

#  video
cap = cv2.VideoCapture(r"zebra_crossing.mp4")

zebra_zone = None  # zebra crossing bounding box

def get_zebra_zone(frame):
    results = zebra_model(frame)[0]
    if results.boxes:
        box = results.boxes[0].xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box

    #     # Extend 350px to the left


        x1_extended = max(0, x1 - 350)
        return np.array([
            [x1_extended, y1],        # top-left 
            [x2, y1 + 30],            # top-right 
            [x2, y2 + 10],            # bottom-right
            [x1_extended, y2 - 20]    # bottom-left
        ], dtype=np.int32)

    return None

#        # For Youtube Zebra Crossing Video



    #     x1_extend = max(0, x1 - 75)
    #     return np.array([
    #         [x1, y1 + 40],          # top-left 
    #         [x2, y1 + 70],          # top-right 
    #         [x2, y2 + 15],          # bottom-right
    #         [x1_extend, y2 - 20]    # bottom-left
    #     ], dtype=np.int32)

    # return None

    


# Read first frame for zebra crossing detection
ret, frame = cap.read()
if not ret:
    print("Failed to read video.")
    exit()

# Detect zebra crossing automatically
zebra_zone = get_zebra_zone(frame)

if zebra_zone is None:
    print("Zebra crossing not detected in the first frame.")
    exit()

# Main loop
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = person_model(frame)[0]

    for box in results.boxes:
        cls = int(box.cls[0])
        if cls != 0:  # Only person
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        leg_x = int((x1 + x2) / 2)
        leg_y = int(y2)

        # Check if leg point is inside zebra zone
        inside = cv2.pointPolygonTest(zebra_zone, (leg_x, leg_y), False) >= 0

        # Draw bounding box
        color = (0, 255, 0) if inside else (0, 0, 255)
        label = "Legal" if inside else "Illegal"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.circle(frame, (leg_x, leg_y), 4, color, -1)

    # Draw zebra zone
    if zebra_zone is not None:
        cv2.polylines(frame, [zebra_zone], True, (255, 255, 0), 2)

    cv2.imshow("Result", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

