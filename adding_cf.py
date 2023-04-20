from afinn import Afinn
import pandas as pd
import re
from labMTsimple.storyLab import *
import nltk
from nltk.tree import Tree
from nltk.tree import ParentedTree
from nltk.tokenize import sent_tokenize
import numpy as np
import spacy
import stanza

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('large_grammars')
stanza.download('en')

reddit_data = reddit_data.rename(columns={'User':'user', 'Post':'post', 'Label':'label'})

afn = Afinn()

def get_sentiment(text):
    score = afn.score(text)
    if score > 0:
        return 'positive'
    elif score < 0:
        return 'negative'
    else:
        return 'neutral'

reddit_data['afinn_score'] = reddit_data['post'].apply(afn.score)
reddit_data['sentiment'] = reddit_data['post'].apply(get_sentiment)

def get_pronoun_count(text):
    pronouns = re.findall(r'\b(I|me|mine|myself|we|us|ours|ourselves)\b', text, re.IGNORECASE)
    return len(pronouns)

reddit_data['personal_pronoun_count'] = reddit_data['post'].apply(get_pronoun_count)
reddit_data['personal_pronoun_itra_ratio'] = reddit_data['personal_pronoun_count'] / reddit_data['personal_pronoun_count'].sum()

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def count_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return len(sentences)

reddit_data['num_sentences'] = reddit_data['post'].apply(count_sentences)

def count_articles(text):
    articles = re.findall(r'\b(the)\b', text, re.IGNORECASE)
    return len(articles)

reddit_data["num_articles"] = reddit_data['post'].apply(count_articles)

def count_pronouns(text):
    sentences = nltk.word_tokenize(text)
    tags = nltk.pos_tag(sentences)
    pronouns = ['PRP', 'PRP$', 'WP', 'WP$']
    
    num_pronouns = sum([1 for word, tag in tags if tag in pronouns])
    
    return num_pronouns

reddit_data["num_pronouns"] = reddit_data['post'].apply(count_pronouns)
reddit_data["intra_pronoun_to_avg_ratio"] = reddit_data["num_pronouns"] / np.mean(reddit_data["num_pronouns"])
