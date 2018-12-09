"""

File to fill with fun clickbait analysis!
Love, Lucino

"""

from newsapi import NewsApiClient


def create_labeled_data():
    news_api = NewsApiClient(api_key='063f02817dbb49528058d7372964f645')
    buzzfeed_headlines = []
    reuters_headlines = []
    x = 1
    while x <= 10:
        b_headlines = news_api.get_everything(sources='buzzfeed',
                                              from_param='2018-11-09',
                                              to='2018-12-09',
                                              language='en',
                                              sort_by='relevancy',
                                              page_size=100,
                                              page=x)['articles']
        r_headlines = news_api.get_everything(sources='reuters',
                                              from_param='2018-11-09',
                                              to='2018-12-09',
                                              language='en',
                                              sort_by='relevancy',
                                              page_size=100,
                                              page=x)['articles']
        b_titles = [(article['title'], 'bait') for article in b_headlines]
        buzzfeed_headlines.extend(b_titles)
        r_titles = [(article['title'], 'not_bait') for article in r_headlines]
        reuters_headlines.extend(r_titles)
        x += 1

    all_headlines = buzzfeed_headlines + reuters_headlines
    return all_headlines


def bait_features(headline):
    features = {}


if __name__ == '__main__':
    myheadlines = create_labeled_data()
    print(myheadlines[990:1010])
