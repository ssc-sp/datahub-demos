# Generated by Django 4.0.10 on 2023-08-16 13:18

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='HNPStats',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('country_key', models.CharField(max_length=255)),
                ('country_name', models.CharField(max_length=255)),
                ('region', models.CharField(max_length=255)),
                ('continent', models.CharField(max_length=255)),
                ('currency', models.CharField(max_length=255)),
                ('capital', models.CharField(max_length=255)),
                ('pop', models.FloatField()),
                ('birth_rate', models.FloatField()),
                ('birth_registration_rate', models.FloatField()),
                ('death_rate', models.FloatField()),
                ('death_registration_rate', models.FloatField()),
                ('fertility_rate', models.FloatField()),
                ('human_capital_index', models.FloatField()),
                ('labour_force', models.FloatField()),
                ('net_migration', models.FloatField()),
                ('consumption_iodized_salt', models.FloatField()),
                ('pop_male', models.FloatField()),
                ('pop_female', models.FloatField()),
            ],
        ),
    ]
