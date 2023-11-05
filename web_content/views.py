from django.shortcuts import render

# Create your views here.
def web_content_list(request):
    return render(request, 'web_content/web_content_list.html', {})
