from django.urls import path
from . import views

urlpatterns = [
    path('system/', views.show_nersystem,name='show_nersystem'),
    path('recognize/', views.recognize_flag, name="recognize_flag"),
    path('save/', views.save_entity_set, name="save_entity_set")
]