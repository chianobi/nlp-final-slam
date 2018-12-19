import random
import nltk
import pickle
import clickbaitoop
from headline import Headline
from sklearn.neighbors import KNeighborsClassifier

#Creates a set of headline objects to read
def create_headlines():
    headline_tuples = pickle.load(open('headlines.p', 'rb'))
    headlines = [Headline(h[0], h[1]) for h in headline_tuples]
    return headlines

# Creates a list of the results of finding the features for each headline
def create_x_vals(headline):
    x_vals = []
    x_vals.append(clickbaitoop.procount(headline))
    x_vals.append(clickbaitoop.punct(headline))
    x_vals.append(clickbaitoop.averagewordlength(headline))
    x_vals.append(clickbaitoop.mostcommontag(headline))
    x_vals.append(clickbaitoop.wh(headline))
    x_vals.append(clickbaitoop.startswithnum(headline))
    x_vals.append(clickbaitoop.superlative(headline))
    x_vals.append(clickbaitoop.imperative(headline))
    x_vals.append(clickbaitoop.bigrams(headline))
    x_vals.append(clickbaitoop.function_words(headline))
    x_vals.append(clickbaitoop.flag_words(headline))
    x_vals.append(clickbaitoop.rare_words(headline))
    return x_vals

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
    neighbor_classifier = KNeighborsClassifier(n_neighbors=25)
    neighbor_classifier.fit(training_x, training_y)
    print(neighbor_classifier.score(training_x, training_y))
    print(neighbor_classifier.predict(create_predictable_list("You can make up a headline and test it here.")))
