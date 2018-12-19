"""
File to fill with fun clickbait analysis!
Love, Lucino
"""

from newsapi import NewsApiClient
import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import random
import pickle
from collections import Counter
from nltk.corpus import stopwords

common_bigrams = []

"""
Opens the pickled file and shuffles it up, returning a list of labeled headlines
"""


def create_labeled_data():
    all_headlines = pickle.load(open('headlines.p', 'rb'))
    random.shuffle(all_headlines)
    return all_headlines


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
        x += 1
    return source_headlines

"""
Generate bigrams based on the training data, then 
"""


def create_feature_sets(labeled_data):
    common = get_bigrams(labeled_data)
    common_bigrams.extend(common)
    feature_sets = [(bait_features(headline), label) for (headline, label) in labeled_data]
    train_set, test_set = feature_sets[:2700], feature_sets[2700:]
    return train_set, test_set


#pulls out most frequent bigrams from the training set each time (they are almost always the same!)
def get_bigrams(lst):
    training = lst[:2700]
    training = [w for w in training if w[1] == 'bait']
    training = [w[0] for w in training]
    raw = " ".join(training)
    tokens = raw.split(" ")
    bigrams = list(nltk.bigrams(tokens))
    fdist = nltk.FreqDist(bigrams)
    most_common = fdist.most_common(25)
    common_bigrams = [x[0] for x in most_common]
    return common_bigrams


def bait_features(headline):
    featureset = {}
    featureset['procount'] = procount(headline)
    featureset['punct'] = punct(headline)
    featureset['averagewordlength'] = averagewordlength(headline)
    featureset['mostcommontag'] = mostcommontag(headline)
    featureset['wh'] = wh(headline)
    featureset['startswithnum'] = startswithnum(headline)
    featureset['imperative'] = imperative(headline)
    featureset['bigrams'] = bigrams(headline)
    featureset['function_words'] = function_words(headline)
    featureset['flag_words'] = flag_words(headline)
    return featureset

# Checks for the use of first and second-person pronouns in article headline; returns true if any found.
def procount(headline):
    pronouns = ["we", "you", "i", "everyone", "us", "your", "our"]
    for w in word_tokenize(headline):
        if w.lower() in pronouns:
            return True
    return False


# Checks end-of-sentence punctuation count in headline; returns true if count is greater than 0.
def punct(headline):
    count = 0
    punct = [".", "!", "?"]
    for w in word_tokenize(headline):
        if w in punct:
            count += 1
    return count > 0


# Checks whether the headline starts with a digit; returns true if so.
def startswithnum(headline):
    tags = [w[1] for w in pos_tag(word_tokenize(headline))]
    if tags[0] == 'CD':
        return True
    return False


# Calculates average word length within headline; returns true if the average is greater than 4.
def averagewordlength(headline):
    charactercount = 0
    wordcount = 0
    commonshortwords = ["the", "a", "for", "an", "of", "and", "so", "but", "with", ",", ".", ":", ";"]
    for w in word_tokenize(headline):
        if w.lower() not in commonshortwords:
            charactercount += len(w)
            wordcount += 1
    avg = charactercount / wordcount
    return avg < 4


# Checks for the most common POS tag in the headline; returns true if the most common tag is NN.
def mostcommontag(headline):
    tags = [w[1] for w in pos_tag(word_tokenize(headline))]
    counts = Counter(tags)
    return counts.most_common()[0][0] == "NN"


# Checks for the use of superlative adjectives in the headline; returns true if any found.
def superlative(headline):
    tags = [w[1] for w in pos_tag(word_tokenize(headline))]
    for tag in tags:
        if tag == 'JJS' or tag == 'RBS':
            return True
    return False


# Checks for the use of wh-words in the headline; returns true if any found.
def wh(headline):
    tags = [w[1] for w in pos_tag(word_tokenize(headline))]
    for tag in tags:
        if tag == 'WP':
            return True
    return False


# Checks whether the first word in a headline is tagged as a bare-form verb, indicating an imperative;
# returns true if it is.
def imperative(headline):
    tags = [w[1] for w in pos_tag(word_tokenize(headline))]
    if tags[0] == 'VB':
        return True
    return False


# Checks all bigrams in the headline, and compares them against a list of most common clickbait bigrams. Returns
# true if any match.
def bigrams(headline):
    bigrams = nltk.bigrams(headline.split(" "))

    for x in bigrams:
        if x in common_bigrams:
            return True
    return False


# Calculates proportion of word in headline that are stopwords or function words
def function_words(headline):
    fun_words = [w for w in [word.lower() for word in word_tokenize(headline)] if w in stopwords.words('english')]
    #return len(fun_words)/len(word_tokenize(headline))
    return True if (len(fun_words)/len(word_tokenize(headline)) == .5) else False


def flag_words(headline):
    flags = ['this', 'will', 'believe', 'surprise']
    found = False
    for word in word_tokenize(headline):
        word = word.lower()
        if word in flags:
            found = True
    return found


def train_classifier(training_set):
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    return classifier


def evaluate_classifier(classifier, test_set):
    print(nltk.classify.accuracy(classifier, test_set))


def classify_headlines(lines, classifier):
    features = [bait_features(line) for line in lines]
    label_list = []
    for feat in features:
        label_list.append(classifier.classify(feat))
    bait_count = label_list.count('bait')
    return bait_count/len(label_list)


if __name__ == '__main__':
    labeled_data = create_labeled_data()
    training_set, test_set = create_feature_sets(labeled_data)
    # classifier = train_classifier(training_set)
    # evaluate_classifier(classifier, test_set)
    # classifier.show_most_informative_features(20)

    # opens a ready-trained classifier to save time on training and evaluating.
    # uncomment the above lines and run the program to see a 'new' classifier's
    # accuracy!
    classifier = pickle.load( open('trained_classifier.p', 'rb'))
    source = input('Welcome to the Clickbait Classifier! '
                   'Choose a news source to classify from sources.txt.\n')

    print('The headlines found from this source are about ' +
          str(classify_headlines(build_source(source), classifier)*100) + '% clickbait!')



