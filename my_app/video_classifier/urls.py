from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_video, name='upload_video'),
    path('result/<uuid:video_id>/', views.video_result, name='video_result'),
    path('status/<uuid:video_id>/', views.check_status, name='check_status'),
    path('list/', views.video_list, name='video_list'),
]