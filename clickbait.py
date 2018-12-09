"""

File to fill with fun clickbait analysis!
Love, Lucino

"""

from newsapi import NewsApiClient

news_api = NewsApiClient(api_key='063f02817dbb49528058d7372964f645')
sources = news_api.get_sources()
print(sources)