from django.core.management.base import BaseCommand
from video_classifier.services import VideoClassificationService
import os
from django.conf import settings

class Command(BaseCommand):
    help = 'Test if the model loads correctly'
    
    def handle(self, *args, **options):
        self.stdout.write('Testing model loading...')
        
        try:
            # Check if model file exists
            model_path = os.path.join(settings.BASE_DIR, 'models_storage', 'model.keras')
            if not os.path.exists(model_path):
                self.stdout.write(
                    self.style.ERROR(f'Model file not found at: {model_path}')
                )
                return
            
            # Try to load the service
            service = VideoClassificationService()
            self.stdout.write(
                self.style.SUCCESS('Model loaded successfully!')
            )
            
            # Print model summary
            self.stdout.write('Model summary:')
            service.model.summary()
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error loading model: {str(e)}')
            )