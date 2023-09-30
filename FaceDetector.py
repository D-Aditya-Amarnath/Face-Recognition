import cv2
from facenet_pytorch import MTCNN
import numpy as np
import os
import glob
from tqdm import tqdm

class FaceDetector:

    def __init__(self, mtcnn) -> None:
        
        self.mtcnn = mtcnn

    def _draw(self, frame, boxes, probs, landmarks):
        if boxes is not None:
            for box, prob, landmark in zip(boxes, probs, landmarks):
                x1, y1, x2, y2 = map(int, box)

                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)

                # Show Probability
                cv2.putText(frame, f'Probability: {prob:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

                # Draw Landmarks
                for point in landmark:
                    x, y = map(int, point)
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        return frame

    def detect_faces(self, folder_path, output_folder):
        
        """cv2.namedWindow("Detect", cv2.WINDOW_NORMAL)  # Use WINDOW_NORMAL for resizable window

        # Resize the window to your desired dimensions
        cv2.resizeWindow("Detect", 800, 600)
        
        for image_path in image_paths:
            frame = cv2.imread(image_path)
            try:
                boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)
                print(boxes, probs, landmarks)
                frame_with_faces = self._draw(frame.copy(), boxes, probs, landmarks)
                cv2.imshow("Detect", frame_with_faces)
                cv2.waitKey(0)
            except:
                pass

        cv2.destroyAllWindows()"""
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        failed_images = []    
        image_paths = glob.glob(os.path.join(folder_path, '*'))
        
        for image_path in tqdm(image_paths, desc='Processing Images'):
            frame = cv2.imread(image_path)
            try:
                boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = map(int, box)
                        face_region = frame[y1:y2, x1:x2]
                        output_path = os.path.join(output_folder, f"face_{os.path.basename(image_path).replace('.', f'_{i}.')}")
                        cv2.imwrite(output_path, face_region)
            except Exception as e:
                failed_images.append(image_path)
                #print(f"Error processing image {image_path}: {e}")
        return failed_images
    
    def failed_faces(self, image_paths):
        
        cv2.namedWindow("Detect", cv2.WINDOW_NORMAL)  # Use WINDOW_NORMAL for resizable window

        # Resize the window to your desired dimensions
        cv2.resizeWindow("Detect", 800, 600)
        
        for image_path in image_paths:
            frame = cv2.imread(image_path)
            try:
                boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)
                print(boxes, probs, landmarks)
                frame_with_faces = self._draw(frame.copy(), boxes, probs, landmarks)
                
                display_width = 800  # You can change this to your desired width
                height, width = frame_with_faces.shape[:2]
                aspect_ratio = width / height
                display_height = int(display_width / aspect_ratio)
                display_frame = cv2.resize(frame_with_faces, (display_width, display_height))
                
                cv2.imshow("Detect", display_frame)
                #cv2.waitKey(0)
                
                #cv2.imshow("Detect", frame_with_faces)
                cv2.waitKey(0)
            except Exception as e:
                print(e)

        cv2.destroyAllWindows()
        
    
mtcnn = MTCNN(margin=20, keep_all=True, post_process=False, device='cuda:0')
fcd = FaceDetector(mtcnn)

folder_path = './Data'
output_folder = "./Output"
#failed_images = fcd.detect_faces(folder_path=folder_path,output_folder=output_folder)
#print(failed_images)

fcd.failed_faces(['./Data\\1N1A7777.JPG', './Data\\1N1A7779.JPG', './Data\\1N1A7783.JPG', './Data\\1N1A7810.JPG', './Data\\1N1A7827.JPG', './Data\\1N1A7857.JPG', './Data\\1N1A7860.JPG', './Data\\1N1A7880.JPG', './Data\\1N1A7881.JPG', './Data\\1N1A7882.JPG', './Data\\1N1A7890.JPG', './Data\\1N1A7891.JPG', './Data\\1N1A7893.JPG', './Data\\1N1A7897.JPG', './Data\\1N1A8006.JPG', './Data\\1N1A8087.JPG', './Data\\1N1A8090.JPG', './Data\\1N1A8096.JPG', './Data\\1N1A8098.JPG', './Data\\1N1A9111.JPG', './Data\\1N1A9112.JPG', './Data\\1N1A9301.JPG', './Data\\1N1A9391.JPG', './Data\\1N1A9402.JPG', './Data\\1N1A9470.JPG', './Data\\1N1A9601.JPG', './Data\\1N1A9753.JPG', './Data\\1N1A9758.JPG', './Data\\1N1A9762.JPG', './Data\\1N1A9768.JPG', './Data\\1N1A9772.JPG'])