from django.conf.urls import url
from melpapp import settings
from .views import *

urlpatterns = [
    url(r'^eng_to_spa/$', EngToSpanish.as_view(), name='eng_to_spa'), 
    url(r'^spa_to_eng/$', SpaToEng.as_view(), name='spa_to_eng'), 
    url(r'^lang_detect/$', LanguageDetect.as_view(), name='lang_detect'), 
]