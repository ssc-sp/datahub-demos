from django.db import models

# Create your models here.
#class MyModel(models.Model):
#	name=models.CharField(max_length=100)

class Book(models.Model):
	name=models.CharField(max_length=100)
	author=models.CharField(max_length=100)
	year=models.IntegerField()
