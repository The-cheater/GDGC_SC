import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import argparse
from datetime import datetime

# Try to import TensorFlow object detection utilities
try:
    # Add TensorFlow models research directory to path
    TENSORFLOW_MODELS_PATH = "/home/shiva/work/tensorflow/models/research"
    sys.path.append(TENSORFLOW_MODELS_PATH)
    sys.path.append(os.path.join(TENSORFLOW_MODELS_PATH, "object_detection"))
    
    from object_detection.utils import label_map_util
    from object_detection.utils import visualization_utils as vis_util
except ImportError as e:
    print(f"Error importing TensorFlow object detection utilities: {e}")
    print("Please ensure you've:")
    print("1. Cloned the TensorFlow models repository")
    print("2. Compiled the protobuf files (protoc object_detection/protos/*.proto --python_out=.)")
    print("3. Added the research directory to your PYTHONPATH")
    raise

class AadhaarValidator:
    def __init__(self):
        self.MODEL_NAME = 'inference_graph'
        self.CWD_PATH = os.path.join(os.getcwd(), 'aadhaar_card')
        self.PATH_TO_CKPT = os.path.join(self.CWD_PATH, self.MODEL_NAME, 'frozen_inference_graph.pb')
        self.PATH_TO_LABELS = os.path.join(self.CWD_PATH, 'label_map.pbtxt')
        self.NUM_CLASSES = 3
        self.detection_graph = None
        self.sess = None
        self.category_index = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the TensorFlow model into memory"""
        try:
            # Load label map
            label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
            categories = label_map_util.convert_label_map_to_categories(
                label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)
            self.category_index = label_map_util.create_category_index(categories)
            
            # Load detection graph
            self.detection_graph = tf.Graph()
            with self.detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.io.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')
                
                self.sess = tf.Session(graph=self.detection_graph)
                
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def validate_aadhaar(self, image_file):
        """
        Validate an Aadhaar card image
        Args:
            image_file: File object or file path
        Returns:
            dict: {
                'is_valid': bool,
                'confidence': float,
                'error': str or None,
                'output_image': numpy array (optional)
            }
        """
        try:
            # Read image
            if isinstance(image_file, str):
                image = cv2.imread(image_file)
            else:
                image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
            
            if image is None:
                return {
                    'is_valid': False,
                    'confidence': 0.0,
                    'error': 'Could not read image file'
                }
            
            # Convert to RGB and expand dimensions
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_expanded = np.expand_dims(image_rgb, axis=0)
            
            # Get tensors
            image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
            
            # Perform detection
            (boxes, scores, classes, num) = self.sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_expanded})
            
            # Process results
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes).astype(np.int32)
            
            # Visualize results
            output_image = image.copy()
            vis_util.visualize_boxes_and_labels_on_image_array(
                output_image,
                boxes,
                classes,
                scores,
                self.category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=0.8)
            
            # Determine validation result (modify this based on your requirements)
            max_score = np.max(scores) if len(scores) > 0 else 0
            is_valid = max_score > 0.8  # Adjust threshold as needed
            
            return {
                'is_valid': is_valid,
                'confidence': float(max_score),
                'error': None,
                'output_image': output_image
            }
            
        except Exception as e:
            return {
                'is_valid': False,
                'confidence': 0.0,
                'error': str(e)
            }

    def __del__(self):
        """Clean up TensorFlow session"""
        if self.sess is not None:
            self.sess.close()

# Singleton instance
validator = AadhaarValidator()

def validate_aadhaar(image_file):
    """
    Public interface for Aadhaar validation
    Args:
        image_file: File object or file path
    Returns:
        dict: Validation results
    """
    return validator.validate_aadhaar(image_file)

# For testing
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='Path to Aadhaar card image')
    args = parser.parse_args()
    
    if args.image:
        result = validate_aadhaar(args.image)
        print("Validation Result:")
        print(f"Valid: {result['is_valid']}")
        print(f"Confidence: {result['confidence']:.2f}")
        if result.get('output_image'):
            cv2.imwrite('output.jpg', result['output_image'])
            print("Saved visualization to output.jpg")
    else:
        print("Please provide an image path with --image argument")