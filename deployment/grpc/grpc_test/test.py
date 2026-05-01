import cv2
import grpc
import detection_pb2
import detection_pb2_grpc

# load test image
img = cv2.imread("test.jpg")
_, encoded = cv2.imencode(".jpg", img)
image_bytes = encoded.tobytes()

# connect to the server
channel = grpc.insecure_channel("localhost:50051")
stub = detection_pb2_grpc.YOLOServiceStub(channel)

# send image for detection
def generate():
    yield detection_pb2.Frame(image_data=image_bytes)

# get detections
for response in stub.StreamDetection(generate()):
    for det in response.detections:
        x, y, w, h = det.x, det.y, det.width, det.height
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        # draw detections
        cv2.rectangle(img, (x1, y1), (x2, y2), (193, 170, 62), 2)
        cv2.putText(img, f"class {det.class_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (193, 170, 62), 2)

cv2.imwrite("output.jpg", img)
print("saved output.jpg")