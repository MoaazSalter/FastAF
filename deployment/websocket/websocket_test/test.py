import asyncio
import websockets
import time
from datetime import datetime
import cv2
import json

SERVER_URL = r"ws://localhost:7860/ws/detect"
TEST_IMAGE_PATH = r"test.jpg"
OUTPUT_IMAGE_PATH = r"output.jpg"

async def test_connection():
    print(f"testing WebSocket connection to {SERVER_URL}")
    
    try:
        # Load test image
        img = cv2.imread(TEST_IMAGE_PATH)
        if img is None:
            print(f"failed to load test image at {TEST_IMAGE_PATH}")
            return
        
        # image to bytes
        _, img_bytes = cv2.imencode('.jpg', img)
        img_bytes = img_bytes.tobytes()
        
        async with websockets.connect(SERVER_URL, open_timeout=120) as ws:
            print("connected to server!")
            
            print(f"\nsending test image at {datetime.now().isoformat()}")
            
            start_time = time.time()
            print(f"sending {len(img_bytes)} bytes of image data to {SERVER_URL}")
            await ws.send(img_bytes)
            try:
                response = await asyncio.wait_for(ws.recv(), timeout=30)
                processing_time = (time.time() - start_time) * 1000
                print(f"received response in {processing_time:.2f}ms")
                response_data = json.loads(response)
                
                # draw detections
                if len(response_data['detections']) > 0:
                    print(f"found {len(response_data['detections'])} detections")
                    
                    for detection in response_data['detections']:
                        leftmost = detection['leftmost']
                        rightmost = detection['rightmost']
                        notify_point = detection['notify_point']
                        class_id = detection['class_id']
                        
                        left_x, left_y = int(leftmost[0]), int(leftmost[1])
                        right_x, right_y = int(rightmost[0]), int(rightmost[1])
                        notify_x, notify_y = int(notify_point[0]), int(notify_point[1])
                        
                        # draw bounding box
                        cv2.rectangle(img, (left_x, left_y ), (right_x, right_y), (193, 170, 62), 2)
                        
                        # draw class label
                        label = f"Class {class_id}"
                        cv2.putText(img, label, (left_x, left_y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (193, 170, 62), 2)
                        
                        # draw notification point
                        cv2.circle(img, (notify_x, notify_y), 6, (79, 62, 193), -1)
                        
                    # Save the output image
                    cv2.imwrite(OUTPUT_IMAGE_PATH, img)
                    print(f"saved output image with detections to {OUTPUT_IMAGE_PATH}")
                else:
                    print("no detections found in the image")
                
            except asyncio.TimeoutError:
                print("timeout waiting for server response")
                
    except websockets.exceptions.InvalidURI:
        print(f"FAILED: invalid URL: {SERVER_URL}") # wss:// for secure connections or ws:// for local testing
    except websockets.exceptions.InvalidHandshake:
        print("FAILED: handshake failed")
    except asyncio.TimeoutError:
        print("FAILED: connection timeout")
    except ConnectionRefusedError:
        print("FAILED: connection refused")
    except Exception as e:
        print(f"FAILED: unexpected error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(test_connection())


