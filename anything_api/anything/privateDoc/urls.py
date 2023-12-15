from django.urls import path
from .views import *

urlpatterns = [
    path('summarize',Summarizedoc.as_view()),
    path('qna',QnADoc.as_view()),
]
