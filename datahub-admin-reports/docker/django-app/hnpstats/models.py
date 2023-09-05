from django.db import models

# Create your models here.
class HNPStats(models.Model):
	country_key = models.CharField(max_length=255)
	country_name = models.CharField(max_length=255)
	region = models.CharField(max_length=255)
	continent = models.CharField(max_length=255)
	currency = models.CharField(max_length=255)
	capital = models.CharField(max_length=255)
	pop = models.FloatField()
	birth_rate = models.FloatField()
	birth_registration_rate = models.FloatField()
	death_rate = models.FloatField()
	death_registration_rate = models.FloatField()
	fertility_rate = models.FloatField()
	human_capital_index = models.FloatField()
	labour_force = models.FloatField()
	net_migration = models.FloatField()
	consumption_iodized_salt = models.FloatField()
	pop_male = models.FloatField()
	pop_female = models.FloatField()
