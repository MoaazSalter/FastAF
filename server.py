import time
import cv2
import numpy as np
import grpc
from concurrent import futures

# Import the generated classes
import detection_pb2
import detection_pb2_grpc

# Import the YOLO model from ultralytics
from ultralytics import YOLO

# Load the YOLO model using your custom weights file 'best.pt'
model = YOLO("best.pt")

class YOLOServiceServicer(detection_pb2_grpc.YOLOServiceServicer):
    def StreamDetection(self, request_iterator, context):
        """
        Receives a stream of image frames, performs YOLO detection on each frame,
        and streams back the detection results.
        """
        for frame in request_iterator:
            try:
                # Convert the incoming bytes to a NumPy array and decode it as an image
                np_arr = np.frombuffer(frame.image_data, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if img is None:
                    context.set_details("Invalid image format")
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    continue

                # Run the YOLO model inference (the model returns a list of results)
                results = model.predict(source=img, verbose=False, conf=0.5)
                
                # Prepare the DetectionResponse message
                detections_list = []
                # The ultralytics YOLO model result typically contains one result per image.
                # Loop over each detection (if any)
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
                                class_id=int(cls),
                            )
                            detections_list.append(detection)
                
                response = detection_pb2.DetectionResponse(detections=detections_list)
                yield response
            
            except Exception as e:
                # In case of error, log and continue to next frame.
                print(f"Error processing frame: {e}")
                continue

def serve():
    # Set up the server with a thread pool
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    detection_pb2_grpc.add_YOLOServiceServicer_to_server(YOLOServiceServicer(), server)
    
    # Listen on port 50051 (or any open port)
    server_address = '[::]:50051'
    server.add_insecure_port(server_address)
    server.start()
    print(f"gRPC AI server running on {server_address}...")
    
    try:
        # Keep the server running
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        print("Server stopping...")
        server.stop(0)

if __name__ == '__main__':
    serve()