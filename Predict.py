from ultralytics import YOLO
from PIL import Image

def Predict_disease(img_path):
    
# Load a model
    model = YOLO(r'./runs/detect/train4/weights/best.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of image
    results = model.predict(img_path)  # return a list of Results objects

# Show the results
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        boxes = r.boxes
        classes = r.names[int(boxes.cls[0])]
    
    return im,classes

    