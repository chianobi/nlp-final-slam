# LING 131A Final Project
Molly Moran, Lucino Chiafullo, Samantha Richards, Arjuna Mahenthiran

This project aims to gather news headlines from various sources and classify them as clickbait or non-clickbait.  Multiple files make up the project, and are summarized below
# clickbait.py and clickbaitoop.py
These files build the classifier, train it, and test it on headline data pulled using NewsAPI.  It opens headlines.p to get the proper list of headlines, and at the end, pickles the trained classifier for future use in other files (to test on unseen data).  
One is object oriented (clickbaitoop)...

# data_collector.py
Uses the NewsAPI to create a list of headlines from clickbait news sources (Buzzfeed and MTV) and non-clickbait news sources (Reuters and AP).  We use the broad assumption that "clickbait" sources are clickbait headlines and are classified as such (and vice versa).  They are labeled, then added to a list of all headlines.  At the end, the list is pickled and stored in headlines.p.  This file could be run, but should not be needed.
