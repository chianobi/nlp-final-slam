"""
Lucino Chiafullo
Quick/dirty file that collects & pickles data
"""

import pickle
import random
from newsapi import NewsApiClient
import nltk

bait_headlines = []
legit_headlines = []


def create_labeled_data():
    news_api = NewsApiClient(api_key='063f02817dbb49528058d7372964f645')
    x = 1
    while x <= 10:
        b_headlines = \
            news_api.get_everything(sources='buzzfeed', from_param='2018-11-16', to='2018-12-15',
                                    language='en',
                                    sort_by='relevancy', page_size=100, page=x)['articles']
        m_headlines = \
            news_api.get_everything(sources='mtv-news', from_param='2018-11-16', to='2018-12-15',
                                    language='en',
                                    sort_by='relevancy', page_size=100, page=x)['articles']
        r_headlines = \
            news_api.get_everything(sources='reuters', from_param='2018-11-16', to='2018-12-15',
                                    language='en', sort_by='relevancy', page_size=100, page=x)['articles']
        a_headlines = \
            news_api.get_everything(sources='associated-press', from_param='2018-11-16', to='2018-12-15',
                                    language='en', sort_by='relevancy', page_size=100, page=x)['articles']
        b_titles = [article['title'] for article in b_headlines]
        b_titles = list(filter(None.__ne__, b_titles))
        b_titles = [(article, 'bait') for article in b_titles]
        bait_headlines.extend(b_titles)
        m_titles = [(article['title'], 'bait') for article in m_headlines]
        bait_headlines.extend(m_titles)
        r_titles = [(article['title'], 'not_bait') for article in r_headlines]
        legit_headlines.extend(r_titles)
        a_titles = [(article['title'], 'not_bait') for article in a_headlines]
        legit_headlines.extend(a_titles)
        x += 1
    all_headlines = bait_headlines + legit_headlines
    random.shuffle(all_headlines)
    pickle.dump(all_headlines, open('headlines.p', 'wb'))


if __name__ == '__main__':
    create_labeled_data()