from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("api/new/", views.new_game, name="new_game"),
    path("api/state/", views.state, name="state"),
]
