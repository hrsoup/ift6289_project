import collections
from pypinyin import lazy_pinyin, Style
from nltk.tokenize import sent_tokenize
from collections import Counter
import jieba.posseg
from copy import deepcopy

from music_preprocess import X_music, Y_music, X, Y, X_music_original, Y_music_original
from utils import args

def is_Chinese(ch):
    if '\u4e00' <= ch <= '\u9fff':
            return True
    return False

def process_pos(sentence):
    #input:['我','是','一','名','大','学','生']#
    #output:['r', 'v', 'm', 'm', 'n', 'n', 'n']#
    sentence_seged = jieba.posseg.cut(''.join(sentence))
    word_list = []
    pos_list = []
    for x in sentence_seged:
        word_list.append(x.word)
        pos_list.append(x.flag)
    out_pos = []
    for i in range(len(word_list)):
        temp = [pos_list[i] for j in range(len(word_list[i]))]
        out_pos.extend(temp)
    return out_pos

def preprocess_dataset(): 
    poems = []
    poems_label = []
    file = open('./language_dataset/ch_struct/ci.txt', "r",encoding="utf-8")
    for line in file:  #every line is a poem
        poem = line.strip() #get poem
        poem = poem.replace(' ','')
        if len(poem) < 11 or len(poem) > 219:  #filter poem
            continue
        if '_' in poem or '《' in poem or '[' in poem or '(' in poem or '（' in poem:
            continue
        poem_label = []
        no_punction_poem = []
        for i in range(len(poem)):
            if is_Chinese(poem[i]) == True:
                poem_label.append(0)
                no_punction_poem.append(poem[i])
            else:
                poem_label.append(1)
                
        temp = []
        true_label = []
        for i in range(len(poem_label)):
            temp.append(poem_label[i])
            if poem_label[i] == 1:
                temp.reverse()
                true_label.extend(temp[:-1])
                temp = []

        # filter_poem_tone = lazy_pinyin(no_punction_poem, style=Style.TONE3)
        # filter_poem_pos = process_pos(no_punction_poem)
        # new_poem = [filter_poem_tone[i][-1] for i in range(len(no_punction_poem))]

        poems.append(no_punction_poem)
        poems_label.append(true_label)

    #counting words
    wordFreq = collections.Counter()
    for poem in poems:
        wordFreq.update(poem)

    wordPairs = sorted(wordFreq.items(), key = lambda x: -x[1])
    words, _ = zip(*wordPairs)
    wordNum = len(words)

    word2id = dict(zip(words, range(wordNum))) #word to ID
    poemsVector = [([word for word in poem]) for poem in poems] # poem to vector
    labelVector = [([label for label in poem_label]) for poem_label in poems_label]

    return word2id, poemsVector, labelVector

word2id, X_language, Y_language = preprocess_dataset()


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

print('The size of poetry is {}'.format(len(X_language_mix)))
print(len(word2id))
print(X_language_original[0])
