from django.urls import path
from .views import *

urlpatterns = [
    path('question/', ChatBotView.as_view(),name='pregunta'),
]