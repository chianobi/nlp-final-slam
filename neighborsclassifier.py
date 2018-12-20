import nltk
from collections import Counter
from nltk.corpus import stopwords
import pickle
from headline import Headline
from sklearn.neighbors import KNeighborsClassifier

common_bigrams = []
common_words = []

#Creates a set of headline objects to read
def create_headlines():
    headline_tuples = pickle.load(open('headlines.p', 'rb'))
    headlines = [Headline(h[0], h[1]) for h in headline_tuples]
    return headlines

# Creates a list of the results of finding the features for each headline
def create_x_vals(headline):
    x_vals = []
    x_vals.append(procount(headline))
    x_vals.append(punct(headline))
    x_vals.append(averagewordlength(headline))
    x_vals.append(mostcommontag(headline))
    x_vals.append(wh(headline))
    x_vals.append(startswithnum(headline))
    x_vals.append(superlative(headline))
    x_vals.append(imperative(headline))
    x_vals.append(bigrams(headline))
    x_vals.append(function_words(headline))
    x_vals.append(flag_words(headline))
    x_vals.append(rare_words(headline))
    return x_vals

# Checks for the use of first and second-person pronouns in article headline; returns true if any found.
def procount(headline):
    pronouns = ["we", "you", "i", "everyone", "us", "your", "our"]
    for w in headline.tokens_lower:
        if w in pronouns:
            return True
    return False


# Checks end-of-sentence punctuation count in headline; returns true if count is greater than 0.
def punct(headline):
    punct = [".", "!", "?"]
    found = False
    for w in headline.tokens:
        if w in punct:
            found = True
    return found


# Checks whether the headline starts with a digit; returns true if so.
def startswithnum(headline):
    tags = [w[1] for w in headline.pos_tagged]
    if tags[0] == 'CD':
        return True
    return False


# Calculates average word length within headline; returns true if the average is greater than 4.
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


# Checks for the most common POS tag in the headline; returns true if the most common tag is NN.
def mostcommontag(headline):
    counts = Counter(headline.pos_tags)
    return counts.most_common()[0][0] == "NN"


# Checks for the use of superlative adjectives in the headline; returns true if any found.
def superlative(headline):
    for tag in headline.pos_tags:
        if tag == 'JJS' or tag == 'RBS':
            return True
    return False


# Checks for the use of wh-words in the headline; returns true if any found.
def wh(headline):
    for tag in headline.pos_tags:
        if tag == 'WP':
            return True
    return False


# Checks whether the first word in a headline is tagged as a bare-form verb, indicating an imperative;
# returns true if it is.
def imperative(headline):
    if headline.pos_tags[0] == 'VB':
        return True
    return False


# Checks all bigrams in the headline, and compares them against a list of most common clickbait bigrams. Returns
# true if any match.
def bigrams(headline):
    bigrams = nltk.bigrams(headline.tokens_lower)

    for x in bigrams:
        if x in common_bigrams:
            return True
    return False

# Calculates proportion of word in headline that are stopwords or function words
def function_words(headline):
    fun_words = [w for w in headline.tokens_lower if w in stopwords.words('english')]
    return True if ((len(fun_words)/headline.num_tokens) == .5) else False

def flag_words(headline):
    flags = ['this', 'these', 'will', 'll', 'believe', 'surprise']
    found = False
    for word in headline.tokens_lower:
        if word in flags:
            found = True
    return found

def rare_words(headline):
    rare = 0
    for word in headline.tokens_lower:
        if word not in common_words:
            rare += 1
    return rare > 17

# Creates a list of all the values of each headline to train the classifier
def create_training_x(headlines):
    x_training_list = []
    for headline in headlines:
        x_training_list.append(create_x_vals(headline))
    return x_training_list

# Creates the list of all the Y-value mappings for each list of values in the list of X
def create_training_y(headlines):
    y_training_list = []
    for headline in headlines:
        y_training_list.append(headline.label)
    return y_training_list

# Creates a list of values that can be used to make a prediction given a headline as a string
def create_predictable_list(headline_string):
    predictor_as_object = Headline(headline_string, "none")
    predictable_values = []
    predictable_values.append(create_x_vals(predictor_as_object))
    return predictable_values


if __name__ == '__main__':

    headlines = create_headlines()
    training_x = (create_training_x(headlines))
    training_y = (create_training_y(headlines))
    neighbor_classifier = KNeighborsClassifier(n_neighbors=101)
    neighbor_classifier.fit(training_x, training_y)
    pickle.dump(neighbor_classifier, open("trained_classifier2.p",'wb'))
    #print(neighbor_classifier.score(training_x, training_y))
    #print(neighbor_classifier.predict(create_predictable_list("You can make up a headline and test it here.")))
