from django.db import models as dj_models

class Prediction(dj_models.Model):
    date_created = dj_models.DateTimeField(auto_now_add=True)
    csv_file = dj_models.FileField(upload_to='predictions/')