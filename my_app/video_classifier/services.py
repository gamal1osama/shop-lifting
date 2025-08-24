import os
import cv2
import numpy as np
import tensorflow as tf
from django.conf import settings
from django.utils import timezone
from .models import VideoUpload

# Configuration (matching your training setup)
IMG_SIZE = (128, 128)
NUM_FRAMES = 16
CHANNELS = 3

# Define F1-Score metric class (same as used in training)
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)
    
    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
    
    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()

class VideoClassificationService:
    def __init__(self):
        model_path = os.path.join(settings.BASE_DIR, 'models_storage', 'model.keras')
        # Load model with custom objects
        self.model = tf.keras.models.load_model(
            model_path, 
            custom_objects={'F1Score': F1Score}
        )
    
    def load_video(self, video_path, num_frames=NUM_FRAMES, img_size=IMG_SIZE):
        """Load and preprocess video for prediction."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, total_frames // num_frames)
        
        frames = []
        for i in range(num_frames):
            frame_idx = min(i * frame_interval, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, img_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
            frames.append(frame)
        
        cap.release()
        
        # Pad if video has fewer frames than required
        if len(frames) < num_frames:
            pad_frame = frames[-1] if frames else np.zeros((*img_size, CHANNELS))
            frames.extend([pad_frame] * (num_frames - len(frames)))
        
        return np.array(frames, dtype=np.float32)
    
    def predict_video(self, video_upload_id):
        """Process video and update database with prediction."""
        try:
            video_upload = VideoUpload.objects.get(id=video_upload_id)
            video_path = video_upload.video_file.path
            
            # Validate video file exists
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # Load and preprocess video
            video_data = self.load_video(video_path)
            video_batch = np.expand_dims(video_data, axis=0)  # Add batch dimension
            
            # Make prediction
            prediction = self.model.predict(video_batch, verbose=0)[0][0]  # Get scalar prediction
            
            # Convert prediction to label and confidence
            confidence = float(prediction)
            is_shoplifter = prediction > 0.5
            prediction_label = 'shoplifter' if is_shoplifter else 'non_shoplifter'
            
            # Update database
            video_upload.prediction = prediction_label
            video_upload.confidence = confidence
            video_upload.processed_at = timezone.now()
            video_upload.save()
            
            return {
                'success': True,
                'prediction': prediction_label,
                'confidence': confidence,
                'is_shoplifter': is_shoplifter
            }
            
        except VideoUpload.DoesNotExist:
            return {
                'success': False,
                'error': f'Video upload with id {video_upload_id} not found'
            }
        except Exception as e:
            # Update database with error
            try:
                video_upload.prediction = 'error'
                video_upload.error_message = str(e)
                video_upload.processed_at = timezone.now()
                video_upload.save()
            except:
                pass  # If we can't update the database, at least return the error
            
            return {
                'success': False,
                'error': str(e)
            }

# Global service instance
video_service = VideoClassificationService()