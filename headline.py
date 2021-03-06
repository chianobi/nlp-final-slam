from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

"""
The headline class creates objects from headlines. The class performs various types of token-level processing
on the headline and stores these results in various attributes of the Headline object.
"""
class Headline:
    def __init__(self, string, label='Unknown'):
        self.string = string
        self.label = label
        self.tokens = word_tokenize(string)
        self.tokens_lower = [token.lower() for token in self.tokens]
        self.num_tokens = len(self.tokens)
        self.pos_tagged = pos_tag(self.tokens)
        self.pos_tags = [tag[1] for tag in self.pos_tagged]


