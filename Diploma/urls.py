"""
URL configuration for Diploma project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include

from web_content.views import harry
from web_content.views import hermione
from web_content.views import ron
from web_content.views import wordcloud_view

urlpatterns = [
    path(r'', include('web_content.urls')),
    path('admin/', admin.site.urls),
    path('harry/', harry, name='harry'),
    path('ron/', ron, name='ron'),
    path('hermione/', hermione, name='hermione'),
    path('wordcloud/', wordcloud_view, name='wordcloud'),
]
