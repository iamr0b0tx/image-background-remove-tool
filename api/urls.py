from django.urls import path
from . import views

urlpatterns = [
    path('remove_bg/', views.remove_bg),
    path('replace_bg/', views.replace_bg),
]