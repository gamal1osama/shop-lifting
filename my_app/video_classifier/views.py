from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import threading
from .forms import VideoUploadForm
from .models import VideoUpload
from .services import video_service

def upload_video(request):
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video_upload = form.save()
            
            # Process video in background thread
            thread = threading.Thread(
                target=video_service.predict_video,
                args=(video_upload.id,)
            )
            thread.daemon = True
            thread.start()
            
            messages.success(request, 'Video uploaded successfully! Processing...')
            return redirect('video_result', video_id=video_upload.id)
    else:
        form = VideoUploadForm()
    
    return render(request, 'video_classifier/upload.html', {'form': form})

def video_result(request, video_id):
    video_upload = get_object_or_404(VideoUpload, id=video_id)
    return render(request, 'video_classifier/result.html', {'video': video_upload})

def check_status(request, video_id):
    video_upload = get_object_or_404(VideoUpload, id=video_id)
    return JsonResponse({
        'status': video_upload.prediction,
        'confidence': video_upload.confidence,
        'error_message': video_upload.error_message
    })

def video_list(request):
    videos = VideoUpload.objects.all()
    return render(request, 'video_classifier/list.html', {'videos': videos})