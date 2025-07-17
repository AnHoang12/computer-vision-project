import cv2 
import numpy as np
import json
import pickle
from skimage.transform import resize

# ==== 1. Load model SVM ====
with open('/home/anhoang/Basic_DL/cv/scikit-learn/parkinglot_detection/model.p', 'rb') as f:
    model = pickle.load(f)


# ==== 2. Load danh sách vị trí các ô đỗ ====
with open('/home/anhoang/Basic_DL/cv/scikit-learn/parkinglot_detection/data/parking_boxes.json') as f:
    parking_lots = json.load(f)

video_path = "/home/anhoang/Basic_DL/cv/scikit-learn/parkinglot_detection/data/carPark.mp4"
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    empty_count = 0

    # ==== 5. Duyệt từng ô đỗ ====
    for spot in parking_lots:
        x, y, w, h = spot

        # Kiểm tra tọa độ hợp lệ
        if x < 0 or y < 0 or x+w > frame.shape[1] or y+h > frame.shape[0] or w <= 0 or h <= 0:
            continue
        
        roi = frame[y:y+h, x:x+w]
        
        # Kiểm tra roi có kích thước hợp lệ không
        if roi.size == 0:
            continue
            
        roi_resized = resize(roi, (15, 15), anti_aliasing=True)
        roi_flatten = roi_resized.flatten().reshape(1, -1)

        prediction = model.predict(roi_flatten)

        if prediction == 0:
            color = (0, 255, 0)  # Xanh = chỗ trống
            empty_count += 1
        else:
            color = (0, 0, 255)  # Đỏ = có xe

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    # ==== 6. Hiển thị số chỗ trống ====
    cv2.putText(frame, f'Empty spots: {empty_count}/{len(parking_lots)}',
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2, cv2.LINE_AA)

    # ==== 7. Hiển thị và lưu ====
    cv2.imshow("Parking Detection", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==== 8. Giải phóng tài nguyên ====
cap.release()
out.release()
cv2.destroyAllWindows()