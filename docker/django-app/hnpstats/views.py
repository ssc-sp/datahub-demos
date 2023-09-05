from django.shortcuts import render
from django.http import HttpResponse, FileResponse
from django.db.models import Count, Q
from .models import HNPStats

# Create your views here.
def my_view(request):
        return HttpResponse("Hello, World!")

# Create your views here.
def all_stats(request):
        stats = HNPStats.objects.all()
        context = {'stats': stats}
        return render(request, 'stats.html', context)
