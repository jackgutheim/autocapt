from django.db import models

# Create your models here.


class TranslateVideos(models.Model):
    video = models.FileField()