"""
Lucino Chiafullo
Quick/dirty file that collects & pickles data
"""

import pickle
import random
from newsapi import NewsApiClient


def create_labeled_data():
    news_api = NewsApiClient(api_key='99a7e430ec8f4cc1b2e6a6b77b65a5bc')
    buzzfeed_headlines = []
    reuters_headlines = []
    x = 1
    while x <= 10:
        b_headlines = news_api.get_everything(sources='buzzfeed',
                                              from_param='2018-11-11',
                                              to='2018-12-09',
                                              language='en',
                                              sort_by='relevancy',
                                              page_size=100,
                                              page=x)['articles']
        r_headlines = news_api.get_everything(sources='reuters',
                                              from_param='2018-11-11',
                                              to='2018-12-09',
                                              language='en',
                                              sort_by='relevancy',
                                              page_size=100,
                                              page=x)['articles']
        b_titles = [article['title'] for article in b_headlines]
        b_titles = list(filter(None.__ne__,b_titles))
        b_titles = [(article, 'bait') for article in b_titles]
        buzzfeed_headlines.extend(b_titles)
        r_titles = [(article['title'], 'not_bait') for article in r_headlines]
        reuters_headlines.extend(r_titles)
        x += 1

    all_headlines = buzzfeed_headlines + reuters_headlines
    random.shuffle(all_headlines)
    pickle.dump(all_headlines, open('headlines.p', 'wb'))


if __name__ == '__main__':
    create_labeled_data()
