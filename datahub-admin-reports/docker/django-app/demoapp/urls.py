from django.urls import path
from . import views

urlpatterns = [
	path('example/', views.my_view, name='example-view'),
	path('books/', views.all_books, name='all_books'),
	path('books_by_year/', views.books_by_year, name='books_by_year'),
]
