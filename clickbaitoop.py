"""
Instructions:

Run clickbaitoop.py on your command line to use the Clickbait Classifier. You will be prompted to choose one of two 
functionalities:

1) you may use the classifier to classify a particular headline, or 
2) you may use it to classify a corpus of headlines from a particular news source. 

Below are some sample interactions:

Welcome to the Clickbait Classifier! You can either view the percentage of clickbait headlines in a news source 
or enter a single headline to classify it. Type 'headline' to classify a single headline or 'source' to get information 
about a news source:
prompt: headline
Enter your headline:
prompt: You wont believe how hard these students worked on their NLP final project.
Calculating...
This article is probably bait.
Would you like to enter another headline? (y/n) 
prompt: n

***

Welcome to the Clickbait Classifier! You can either view the percentage of clickbait headlines in a news source 
or enter a single headline to classify it. Type 'headline' to classify a single headline or 'source' to get information 
about a news source:
prompt: source
Enter the name of a news source from sources.txt: 
prompt: mtv-news
Calculating...
This news source is approximately 47.24 % clickbait.
Would you like to try another news source? (y/n)
prompt: n


"""

import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import random
import pickle
from collections import Counter
from nltk.corpus import stopwords
from headline import Headline
from newscorpus import Newscorpus
from neighborsclassifier import *

common_bigrams = []
common_words = []
splitpoint = 2700

"""
Opens the pickled file and shuffles it up, returning a list of labeled headlines
"""


def create_headlines():
    headline_tuples = pickle.load(open('headlines.p', 'rb'))
    headlines = [Headline(h[0], h[1]) for h in headline_tuples]
    random.shuffle(headlines)
    return headlines


def create_feature_sets(headlines):
    common_bigrams.extend(get_bigrams(headlines))
    common_words.extend(get_common_words(headlines))
    feature_sets = [(bait_features(headline), headline.label) for headline in headlines]
    train_set, test_set = feature_sets[:splitpoint], feature_sets[splitpoint:]
    return train_set, test_set


"""
Pulls out most frequent bigrams from the training set each time (they are almost always the same!)
"""


def get_bigrams(headlines):
    all_bgrams = []
    training = [h.tokens_lower for h in headlines[:splitpoint] if h.label == 'bait']
    for h in training:
        all_bgrams.extend(list(nltk.bigrams(h)))
    fdist = nltk.FreqDist(all_bgrams)
    most_common = fdist.most_common(25)
    common_bigrams = [x[0] for x in most_common]
    return common_bigrams


def get_common_words(headlines):
    # get list of all tokens in corpus
    corpus_tokens = []
    for h in headlines:
        corpus_tokens.extend(h.tokens_lower)
    # return most common words
    fdist = nltk.FreqDist(corpus_tokens)
    common_words = fdist.most_common(50)
    return common_words


def bait_features(headline):
    featureset = {}
    featureset['procount'] = procount(headline)
    featureset['punct'] = punct(headline)
    featureset['averagewordlength'] = averagewordlength(headline)
    featureset['mostcommontag'] = mostcommontag(headline)
    featureset['wh'] = wh(headline)
    featureset['startswithnum'] = startswithnum(headline)
    featureset['superlative'] = superlative(headline)
    featureset['imperative'] = imperative(headline)
    featureset['bigrams'] = bigrams(headline)
    # featureset['function_words'] = function_words(headline)
    featureset['flag_words'] = flag_words(headline)
    featureset['rare_words'] = rare_words(headline)
    return featureset


"""
Checks for the use of first and second-person pronouns in article headline; returns true if any found.
"""


def procount(headline):
    pronouns = ["we", "you", "i", "everyone", "us", "your", "our"]
    for w in headline.tokens_lower:
        if w in pronouns:
            return True
    return False


"""
Checks end-of-sentence punctuation count in headline; returns true if count is greater than 0.
"""


def punct(headline):
    punct = [".", "!", "?"]
    found = False
    for w in headline.tokens:
        if w in punct:
            found = True
    return found


"""
Checks whether the headline starts with a digit; returns true if so.
"""


def startswithnum(headline):
    tags = [w[1] for w in headline.pos_tagged]
    if tags[0] == 'CD':
        return True
    return False


"""
Calculates average word length within headline; returns true if the average is greater than 4.
"""


def averagewordlength(headline):
    charactercount = 0
    wordcount = 0
    commonshortwords = ["the", "a", "for", "an", "of", "and", "so", "but", "with", ",", ".", ":", ";"]
    for w in headline.tokens_lower:
        if w not in commonshortwords:
            charactercount += len(w)
            wordcount += 1
    avg = charactercount / wordcount
    return avg < 4


"""
Checks for the most common POS tag in the headline; returns true if the most common tag is NN.
"""


def mostcommontag(headline):
    counts = Counter(headline.pos_tags)
    return counts.most_common()[0][0] == "NN"


"""
Checks for the use of superlative adjectives in the headline; returns true if any found.
"""


def superlative(headline):
    for tag in headline.pos_tags:
        if tag == 'JJS' or tag == 'RBS':
            return True
    return False


"""
Checks for the use of wh-words in the headline; returns true if any found.
"""


def wh(headline):
    for tag in headline.pos_tags:
        if tag == 'WP':
            return True
    return False


"""
Checks whether the first word in a headline is tagged as a bare-form verb, indicating an imperative; returns true if it is.
"""


def imperative(headline):
    if headline.pos_tags[0] == 'VB':
        return True
    return False


"""
Checks all bigrams in the headline, and compares them against a list of most common clickbait bigrams. Returns
true if any match.
"""


def bigrams(headline):
    bigrams = nltk.bigrams(headline.tokens_lower)

    for x in bigrams:
        if x in common_bigrams:
            return True
    return False


"""
Calculates proportion of word in headline that are stopwords or function words; returns true if the ratio is .5.
"""


def function_words(headline):
    fun_words = [w for w in headline.tokens_lower if w in stopwords.words('english')]
    # return len(fun_words)/headline.num_tokens
    return True if ((len(fun_words) / headline.num_tokens) == .5) else False


"""
Checks the tokenized headline for instances of common clickbait flagwords; returns true if any found.
"""


def flag_words(headline):
    flags = ['this', 'these', 'will', 'll', 'believe', 'surprise']
    found = False
    for word in headline.tokens_lower:
        if word in flags:
            found = True
    return found


"""
Checks how many words in the headline are not in the common_words list; returns true if the number is greater than 17.
"""


def rare_words(headline):
    rare = 0
    for word in headline.tokens_lower:
        if word not in common_words:
            rare += 1
    return rare > 17


def train_classifier(training_set):
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    return classifier


def evaluate_classifier(classifier, test_set):
    print(nltk.classify.accuracy(classifier, test_set))


def classify_headlines(headlines, classifier):
    features = [bait_features(Headline(headline)) for headline in headlines]
    label_list = []
    for feat in features:
        label_list.append(classifier.classify(feat))
    bait_count = label_list.count('bait')
    return bait_count / len(label_list)


def classify_headline(headline, classifier):
    return classifier.predict(create_predictable_list(headline))


if __name__ == '__main__':
    # headlines = create_headlines()
    # training_set, test_set = create_feature_sets(headlines)
    # classifier = train_classifier(training_set)
    # evaluate_classifier(classifier, test_set)
    # classifier.show_most_informative_features()

    print(
        "Welcome to the Clickbait Classifier! You can either view the percentage of clickbait headlines in a news source ")
    print(
        "or enter a single headline to classify it. Type 'headline' to classify a single headline or 'source' to get information ")
    print("about a news source:", end=" ")
    arm = input()

    while (arm == 'headline'):
        classifier = pickle.load(open('trained_classifier2.p', 'rb'))
        print()
        print("Enter your headline:", end=" ")
        hline = input()
        print("Calculating...")
        classification = classify_headline(hline, classifier)
        print("This article is probably " + classification[0] + ".")
        print()
        print("Would you like to enter another headline? (y/n) ", end=" ")
        inp = input()
        if inp == 'y':
            arm = 'headline'
        else:
            arm = ''

    while (arm == 'source'):
        classifier = pickle.load(open('trained_classifier.p', 'rb'))
        print("Enter the name of a news source from sources.txt:", end=" ")
        usersource = input()
        print("Calculating...")
        key = '89cf7e72dd4749c2b29e80c44da173f3'
        newssource = Newscorpus(key, usersource)
        print("This news source is approximately %.2f %% clickbait." % (
                classify_headlines(newssource.headlines, classifier) * 100))
        print()
        print("Would you like to try another news source? (y/n)", end=" ")
        inp = input()
        if inp == 'y':
            arm = 'source'
        else:
            arm = ''
            print('\n')
