import cv2
from ultralytics import YOLO
from supervision import ByteTrack, Detections

model = YOLO('yolov8s.pt')
tracker = ByteTrack()
#vehicle_classes = {2: "car", 4: "motorbike", 1: "bus", 6: "truck"}
vehicle_classes = {2: "car", 3: "motorbike", 5: "bus", 7: "truck"}
video_path = 'test1.mp4'
cap = cv2.VideoCapture(video_path)

total_counts = {
    'car': 0,
    'motorbike': 0,
    'bus': 0,
    'truck': 0
    }

# The counting lines
incoming_lane_y = 490
outgoing_lane_y = 400

# Tracks if a vehicle was already counted
counted_ids = set()

# Tracks the current frame
frame_id = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Processes and shows every third frame
    frame_id += 1
    if frame_id % 3 != 0:
        continue

    results = model(frame)[0]

    # Pass the detections to ByteTrack and filter them only for the required classes
    detections = Detections.from_ultralytics(results)
    class_filter = [int(cls) in vehicle_classes for cls in detections.class_id]
    filtered_detections = detections[class_filter]

    tracked = tracker.update_with_detections(filtered_detections)

    for i in range(len(tracked)):
        x1, y1, x2, y2 = map(int, tracked.xyxy[i])
        cls_id = int(tracked.class_id[i])
        track_id = int(tracked.tracker_id[i])
        conf = float(tracked.confidence[i])
        label = f"{vehicle_classes[cls_id]} {conf:.2f} #{track_id}"

        # Center of the vehicle bounding box, used to check if a vehicle has passed the line
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Check if an uncounted vehicle has passed one of the counting lines
        if (track_id not in counted_ids) and ((incoming_lane_y - 10 < center_y < incoming_lane_y + 10 and frame.shape[1] // 2 < center_x < frame.shape[1]) or 
                                    (outgoing_lane_y - 10 < center_y < outgoing_lane_y + 10 and 0 < center_x < frame.shape[1] // 2 - 90)) :
            counted_ids.add(track_id)
            total_counts[vehicle_classes[cls_id]] += 1

        # Draw vehicle box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Draw the counting lines
    cv2.line(frame, (frame.shape[1] // 2, incoming_lane_y), (frame.shape[1], incoming_lane_y), (0, 0, 255), 2)
    cv2.line(frame, (0, outgoing_lane_y), (frame.shape[1] // 2 - 90, outgoing_lane_y), (0, 0, 255), 2)

    # Write out the vehicle counts
    bike_counts = f"Motorbikes: {total_counts['motorbike']}"
    car_counts = f"Cars: {total_counts['car']}"
    bus_counts = f"Buses: {total_counts['bus']}"
    truck_counts = f"Trucks: {total_counts['truck']}"
    cv2.putText(frame, bike_counts, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(frame, car_counts, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(frame, bus_counts, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(frame, truck_counts, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Show the frame
    cv2.imshow("Vehicle Detection", frame)

    # Exit on 'q' key
    if cv2.waitKey(50) & 0xFF == ord("q"):
        break

print(f"Total vehicles counted: {total_counts}")

cap.release()
cv2.destroyAllWindows()