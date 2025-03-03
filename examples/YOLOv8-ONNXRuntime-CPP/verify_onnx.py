import time
import cv2
import numpy as np
import onnxruntime
import matplotlib.pyplot as plt

# Utility functions
def xywh2xyxy(x):
    """Convert [x_center, y_center, width, height] to [x1, y1, x2, y2]"""
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
    return y

def draw_detections(image, boxes, scores, class_ids, conf_threshold=0.3):
    """Draw bounding boxes on image"""
    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes[i])
        conf = scores[i]

        # Draw bounding box
        color = (0, 255, 0)  # Green
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Label
        label = f"Class {int(class_ids[i])}: {conf:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

class YOLOv8_ONNX:
    def __init__(self, model_path, conf_thres=0.3, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres

        # Load ONNX model
        self.session = onnxruntime.InferenceSession(model_path, providers=onnxruntime.get_available_providers())

        # Get input & output details
        self.get_input_details()
        self.get_output_details()

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def preprocess_image(self, image):
        """Prepare image for YOLOv8 model"""
        self.img_height, self.img_width = image.shape[:2]

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.input_width, self.input_height))

        img_resized = img_resized / 255.0  # Normalize
        img_transposed = img_resized.transpose(2, 0, 1)  # HWC -> CHW
        img_input = np.expand_dims(img_transposed, axis=0).astype(np.float32)

        return img_input

    def infer(self, image):
        """Run inference on an image"""
        input_tensor = self.preprocess_image(image)
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        return self.process_output(outputs)

    def process_output(self, output):
        """Process YOLOv8 output"""
        predictions = np.squeeze(output[0]).T

        # Get confidence scores and filter by threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        class_ids = np.argmax(predictions[:, 4:], axis=1)
        boxes = self.rescale_boxes(predictions[:, :4])

        return boxes, scores, class_ids

    def rescale_boxes(self, boxes):
        """Convert YOLO format (center x, center y, width, height) to image coordinates"""
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return xywh2xyxy(boxes)

# Load YOLOv8 ONNX model
model_path = "./best.onnx"  # Update with your actual path
yolo_detector = YOLOv8_ONNX(model_path, conf_thres=0.3, iou_thres=0.5)

# Load Test Image
image_path = "./build/frame_000000.jpg"  # Update with your actual path
img = cv2.imread(image_path)
if img is None:
    raise ValueError(f"❌ Error: Unable to load image from {image_path}")

# Detect objects
boxes, scores, class_ids = yolo_detector.infer(img)

# Draw detections
img_with_detections = draw_detections(img, boxes, scores, class_ids)

# Show the image with bounding boxes
cv2.imshow("Detections", img_with_detections)  # Give a window name
cv2.waitKey(0)  # Wait for a key press
cv2.destroyAllWindows()  # Close the window
