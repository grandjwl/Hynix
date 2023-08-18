from django.urls import path
from . import views

urlpatterns = [
    path("", views.main, name="main"),
    path('simulator/',views.simulation,name='simulator'),
    path('mlc/',views.lifecycle,name='mlc'),
]

