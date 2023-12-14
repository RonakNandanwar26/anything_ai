from django.urls import path
from .views import *

urlpatterns = [
    path('summarize',Summarizedoc.as_view()),
]
