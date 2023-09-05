from django.urls import path
from . import views

urlpatterns = [
        path('example/', views.my_view, name='example-view'),
	path('stats/', views.all_stats, name='stats'),
]
