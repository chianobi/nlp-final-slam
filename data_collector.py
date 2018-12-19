"""
Lucino Chiafullo
Quick/dirty file that collects & pickles data
"""

import pickle
import random
from newsapi import NewsApiClient
import nltk
import datetime

bait_headlines = []
legit_headlines = []

now = datetime.datetime.now()
now_str = now.strftime('%Y') + "-" + now.strftime('%m') + "-" + now.strftime('%d')
monthago = now - datetime.timedelta(days=30)
monthago_str = monthago.strftime('%Y') + "-" + monthago.strftime('%m') + "-" + monthago.strftime('%d')




def create_labeled_data():
    news_api = NewsApiClient(api_key='063f02817dbb49528058d7372964f645')
    x = 1
    while x <= 10:
        b_headlines = \
            news_api.get_everything(sources='buzzfeed', from_param=monthago_str, to=now_str,
                                    language='en',
                                    sort_by='relevancy', page_size=100, page=x)['articles']
        m_headlines = \
            news_api.get_everything(sources='mtv-news', from_param=monthago_str, to=now_str,
                                    language='en',
                                    sort_by='relevancy', page_size=100, page=x)['articles']
        r_headlines = \
            news_api.get_everything(sources='reuters', from_param=monthago_str, to=now_str,
                                    language='en', sort_by='relevancy', page_size=100, page=x)['articles']
        a_headlines = \
            news_api.get_everything(sources='associated-press', from_param=monthago_str, to=now_str,
                                    language='en', sort_by='relevancy', page_size=100, page=x)['articles']
        b_titles = [article['title'] for article in b_headlines]
        b_titles = list(filter(None.__ne__, b_titles))
        b_titles = [(article, 'bait') for article in b_titles]
        bait_headlines.extend(b_titles)
        m_titles = [(article['title'], 'bait') for article in m_headlines]
        print(len(m_titles))
        bait_headlines.extend(m_titles)
        r_titles = [(article['title'], 'not_bait') for article in r_headlines]
        legit_headlines.extend(r_titles)
        a_titles = [(article['title'], 'not_bait') for article in a_headlines]
        legit_headlines.extend(a_titles)
        x += 1

    print(len(bait_headlines))
    print(len(legit_headlines))
    all_headlines = bait_headlines + legit_headlines
    random.shuffle(all_headlines)
    pickle.dump(all_headlines, open('headlines.p', 'wb'))


if __name__ == '__main__':
    create_labeled_data()