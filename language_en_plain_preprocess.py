import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string
from nltk.tokenize import sent_tokenize
from collections import Counter
from music_preprocess import X_music, Y_music, X, Y, X_music_original, Y_music_original
from copy import deepcopy

def stemming_preprocessor(text):
    # Initialize stemmer
    stemmer = PorterStemmer()

    # stem words
    stemmed_output = [stemmer.stem(word = w) for w in text]
    return stemmed_output

# function used to preprocess with lemmatizing
def lemmatize_preprocessor(text):
    # Initialize the Wordnet Lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Lemmatize list of words and join
    lemmatized_output = [lemmatizer.lemmatize(w) for w in text]
    return lemmatized_output

def preprocess_language():
    X = []
    y = []
    file = open('./language_dataset/en_plain/en_joke.txt', "r",encoding="utf-8")
    for line in file:  #every line is a poem
        joke = line.strip() #get poem  
        pattern = re.compile('[^\t\n]+')
        sentences = []
        labels = []
        for sentence in sent_tokenize(joke.replace('\n','')):
            # remove all punctuation
            punctuation_regex = re.compile('[%s]' % re.escape(string.punctuation))
            sentence = punctuation_regex.sub(' ',sentence)
            sentence = [val for values in map(pattern.findall, sentence.lower().split(' ')) for val in values]
            if len(sentence) >=2 :
                label = [0] * len(sentence)
                label[0] = 1
                    
                # lematization on the joke
                sentence = lemmatize_preprocessor(sentence)
                # stemming on the joke
                sentence = stemming_preprocessor(sentence)

                sentences.extend(sentence)
                labels.extend(label)

        if len(sentences) < 128:
            if len(sentences) == len(labels):
                X.append(sentences)
                y.append(labels)

    X = X[:21000]
    y = y[:21000]

    return X, y

X_language, Y_language = preprocess_language()
# join language and music

X_language_id = deepcopy(X_language)
Y_language_id = deepcopy(Y_language)
X_language_id.extend(X)
Y_language_id.extend(Y)

X_language_original = deepcopy(X_language)
Y_language_original = deepcopy(Y_language)

X_language_pretrain = deepcopy(X_language)
Y_language_pretrain = deepcopy(Y_language)
X_language_pretrain.extend(X_music_original) #all music data after 800
Y_language_pretrain.extend(Y_music_original)

X_language_mix = deepcopy(X_language)
Y_language_mix = deepcopy(Y_language)
X_language_mix.extend(X_music) # add sample data
Y_language_mix.extend(Y_music)

word_count = Counter()
for i in range(len(X_language_id)):
    for item in X_language_id[i]:
        word_count[item] += 1

vocab = dict(word_count.most_common())

idx_to_word = list(vocab.keys())
word2id = {word: i for i, word in enumerate(idx_to_word)}
print('The size of en_plain and music is {}'.format(len(X_language_mix)))
print('The size of en_plain and music vocabulary is {}'.format(len(word2id)))