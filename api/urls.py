from django.urls import path
from . import views

urlpatterns = [
    path('remove_bg/', views.remove_bg),
    path('replace_bg_image/', views.replace_bg_image),
    path('replace_bg_color/', views.replace_bg_color),
]