from django import forms
from .models import VideoUpload

class VideoUploadForm(forms.ModelForm):
    class Meta:
        model = VideoUpload
        fields = ['video_file']
        widgets = {
            'video_file': forms.FileInput(attrs={
                'class': 'form-control-file',
                'accept': 'video/mp4,video/avi,video/quicktime'
            })
        }