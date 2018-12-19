"""

In which we look at how accurately a classifier predicts sources based on training data

"""

from newsapi import NewsApiClient


# build a toy corpus of headlines given a source
def build_source(source):
    source_headlines = []
    news_api = NewsApiClient(api_key='063f02817dbb49528058d7372964f645')
    x = 1
    while x <= 4:
        s_headlines = \
            news_api.get_everything(sources=source, from_param='2018-11-19', to='2018-12-19',
                                    language='en',
                                    sort_by='relevancy', page_size=100, page=x)['articles']
        s_titles = [article['title'] for article in s_headlines]
        s_titles = list(filter(None.__ne__, s_titles))
        source_headlines.extend(s_titles)
    return source_headlines





if __name__ == '__main__':
    text = input('Would you like to test headlines or sources? (Enter headlines or sources)\n')
    if text == 'headlines':
        headlines = input('please type in your headline')
    if text == 'sources':
        source = input('Please type in your source, found on the sources.txt list')