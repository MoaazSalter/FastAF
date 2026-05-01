import cv2
import numpy as np
import grpc
from concurrent import futures
import detection_pb2
import detection_pb2_grpc
from ultralytics import YOLO

MODEL_PATH = r"model_development\model_training\model_4_yolo11m_datasetv4\best.pt"

# load the model
model = YOLO(MODEL_PATH)

class YOLOServiceServicer(detection_pb2_grpc.YOLOServiceServicer):
    def StreamDetection(self, request_iterator, context):
        for frame in request_iterator:
            try:
                # image decodeing
                np_arr = np.frombuffer(frame.image_data, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if img is None:
                    yield detection_pb2.DetectionResponse(detections=[])
                    continue

                if img.shape[:2] != (640, 640):
                    img = cv2.resize(img, (640, 640))

                # run inference
                results = model.predict(source=img, verbose=False, conf=0.5)
                
                detections_list = []
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:                        
                        for bbox, cls in zip(boxes.xywh, boxes.cls):
                            x_center, y_center, width, height = bbox

                            detection = detection_pb2.Detection(
                                x=float(x_center),
                                y=float(y_center),
                                width=float(width),
                                height=float(height),
                                class_id=int(cls)
                            )
                            detections_list.append(detection)
                
                response = detection_pb2.DetectionResponse(detections=detections_list)
                yield response
            
            except Exception as e:
                print(f"Error processing frame: {e}")
                yield detection_pb2.DetectionResponse(detections=[])
                continue

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    detection_pb2_grpc.add_YOLOServiceServicer_to_server(YOLOServiceServicer(), server)
    server_address = '[::]:50051'
    server.add_insecure_port(server_address)
    server.start()
    print(f"gRPC AI server running on {server_address}...")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("Server stopping...")
        server.stop(0)

if __name__ == '__main__':
    serve()