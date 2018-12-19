import time
import datetime
import nltk
from newsapi import NewsApiClient

class Newscorpus:
	def __init__(self,key,source):
		self.key = key
		self.source = source
		news_api = NewsApiClient(api_key=key)
		headlines = []
		x = 1
		while x <=10:
			htitles = news_api.get_everything(sources=self.source, from_param='2018-11-20', to='2018-12-18',language='en',sort_by='relevancy', page_size=100, page=x)['articles']
			headline_titles = [article['title'] for article in htitles]
			headline_titles = list(filter(None.__ne__, headline_titles))
			headlines.extend(headline_titles)
			x +=1
		self.headlines = headlines

    