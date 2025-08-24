from django.db import models
from django.core.validators import FileExtensionValidator
import uuid

class VideoUpload(models.Model):
    PREDICTION_CHOICES = [
        ('processing', 'Processing'),
        ('non_shoplifter', 'Non-Shoplifter'),
        ('shoplifter', 'Shoplifter'),
        ('error', 'Error'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    video_file = models.FileField(
        upload_to='videos/',
        validators=[FileExtensionValidator(allowed_extensions=['mp4', 'avi', 'mov'])]
    )
    uploaded_at = models.DateTimeField(auto_now_add=True)
    prediction = models.CharField(max_length=20, choices=PREDICTION_CHOICES, default='processing')
    confidence = models.FloatField(null=True, blank=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    error_message = models.TextField(blank=True)
    
    class Meta:
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return f"Video {self.id} - {self.prediction}"