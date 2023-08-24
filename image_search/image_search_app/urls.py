from django.urls import path
from . import views # . means current directory and views is a file in that
# the control here comes from project urls 
urlpatterns = [
    path('', views.home, name = 'home'), 
]
#views is the file that contains the about page function and the home page function