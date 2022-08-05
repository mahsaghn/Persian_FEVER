from django.urls import path
from . import views

urlpatterns = [
    path('',views.Stance,name = 'fact_extraction')
]