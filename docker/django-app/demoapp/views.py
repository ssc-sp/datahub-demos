import matplotlib.pyplot as plt
import io
import os

from django.conf import settings
from django.shortcuts import render
from django.http import HttpResponse, FileResponse
from django.db.models import Count, Q
from .models import Book

# Create your views here.
def my_view(request):
	return HttpResponse("Hello, World!")

def filter_books(request):
	search_query = request.GET.get('search', '')
	author_filter = request.GET.get('author', '')
	year_filter = request.GET.get('year', '')

	books = Book.objects.all()

	if search_query:
		books = books.filter(Q(name__icontains=search_query) | Q(author__icontains=search_query))

	if author_filter:
		books = books.filter(author__icontains = author_filter)

	if year_filter:
		books = books.filter(year=year_filter)

	return books

def all_books(request):
	books = filter_books(request)
	context = {'books': books}
	return render(request, 'books.html', context)

def books_by_year(request):
	years = []
	counts = []

	books = filter_books(request)
	books = Book.objects.values('year').annotate(count=Count('year')).order_by('year')

	for book in books:
		years.append(book['year'])
		counts.append(book['count'])

	plt.bar(years, counts)
	plt.xlabel('Year')
	plt.ylabel('Number of books')
	plt.title("Books by Release Year")

	image_path = os.path.join(settings.STATICFILES_DIRS[0], 'images', 'books_by_year.png')
	#os.makedirs(image_path, exist_ok = True)

	plt.savefig(image_path, format='png')
	context = {'books': books}

	return render(request, 'books/book_list.html', context)
