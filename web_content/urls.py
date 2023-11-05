from django.urls import path
from . import views

urlpatterns = [
    path(r'', views.web_content_list, name='web_content_list'),
]