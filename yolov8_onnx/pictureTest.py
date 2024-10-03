from ultralytics import YOLO
import glob

# Load a model
model = YOLO("models/best4.pt")  # pretrained YOLOv8n model

images = glob.glob("test/*.jpg")

target_class = 5

# Run batched inference on a list of images
results = model(images, stream=True)  # return a generator of Results objects

# Process results generator
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    
    # Check if there are any boxes detected
    if len(boxes) > 0:
        # Filter boxes to keep only the target class
        filtered_boxes = [box for box in boxes if box.cls >= 0]
        
        # If there are filtered boxes, update the result object
        if len(filtered_boxes) > 0:
            # Update the boxes attribute directly
            result.boxes = filtered_boxes
            
            # Optionally, you can also filter other attributes like masks and keypoints
            # filtered_masks = [mask for mask, box in zip(result.masks, boxes) if box.cls == target_class]
            # result.masks = filtered_masks
            
            # filtered_keypoints = [kp for kp, box in zip(result.keypoints, boxes) if box.cls == target_class]
            # result.keypoints = filtered_keypoints
            
            result.show()  # display to screen
        else:
            print("No boxes of target class detected in this image.")
    else:
        print("No boxes detected in this image.")