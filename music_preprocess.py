from copy import deepcopy
import os
import collections
from utils import args
import numpy as np
import random

class Melody:
    def __init__(self):
        self.shortest_rhythm = None # int
        self.notes = [] # each element in this list is a Note object
        self.labels = [] #  int(1) for the begin of a phrase, int(0) for other
        self.features = []

    def get_shortest_rhythm(self, line):
        items = line[4: -1].split(' ')
        items = [x for x in items if x != '' and x != ' ']
        self.shortest_rhythm = int(items[1])

    def get_notes_labels(self, mel_entry):  
        notes_list = []
        labels_list = []
        phrase_list = mel_entry.split('\n')
        for phrase in phrase_list:
            phrase = phrase.replace('/','')
            phrase = phrase.replace('(','')
            phrase = phrase.replace(')','')
            phrase = [x for x in phrase if x != '' and x != ' ']
            notes = []
            n = [] # a note unit
            begin = 0
            first_note = 0
            for i in range(len(phrase)):
                # if it's the begin of a note, begin = 1
                if (phrase[i] == '+' and phrase[i+1] >= '1' and phrase[i+1] <= '7'):
                    begin = 1 
                    first_note += 1
                elif (phrase[i] == '-' and phrase[i+1] >= '1' and phrase[i+1] <= '7'):
                    begin = 1
                    first_note += 1
                elif (phrase[i] >= '0' and phrase[i] <= '7'):
                    if i >= 1:
                        if phrase[i-1] != '+' and phrase[i-1] != '-':
                          begin = 1
                          first_note += 1
                    if i == 0:
                        begin = 1
                        first_note += 1                     

                if begin == 1 and first_note != 1:
                    n = [x for x in n if x != '' and x != ' ']
                    if n:
                        begin = 0
                        note = Note()
                        note.get_pitch_duration(self.shortest_rhythm, n)
                        notes.append(note)
                        n = []
                    n.append(phrase[i])
                    if i == len(phrase) - 1:
                        note = Note()
                        note.get_pitch_duration(self.shortest_rhythm, n)
                        notes.append(note)   
                                             
                elif i == len(phrase) - 1:
                    n.append(phrase[i])
                    n = [x for x in n if x != '' and x != ' ']
                    if n:
                        begin = 0
                        note = Note()
                        note.get_pitch_duration(self.shortest_rhythm, n)
                        notes.append(note)
                        n = []
                else:
                    n.append(phrase[i])

            labels = [None]*len(notes)
            for i in range(len(notes)):
                if i == 0:
                    labels[0] = 1
                else:
                    labels[i] = 0

            notes_list.extend(notes)
            labels_list.extend(labels)
                
        self.notes = notes_list
        self.labels = labels_list

    def get_features(self):
        for note in self.notes:
            feature = str(note.pitch) + ' ' + str(note.duration)
            self.features.append(feature)    

class Note:
    def __init__(self):
        self.pitch = None #str
        self.duration = None #int

    def get_pitch_duration(self, shortest_rhythm, n): 
        self.duration = shortest_rhythm
        # get pitch and duration
        # eg of pitch: '3', '4#', '4b', '0', '-3', '+3'
        if n[0] >= '0' and n[0] <= '7':
            if len(n) >= 2 and (n[1] == 'b' or n[1] == '#'):
                self.pitch = ''.join(n[0:2])
                if len(n) == 2:
                    self.duration = shortest_rhythm
                else:
                    for i in range(len(n)-2):
                        if n[i+2] == '_':
                            self.duration = self.duration * 2
                        elif n[i+2] == '.':
                            self.duration = self.duration * (1 + 0.5)
            else:
                self.pitch = n[0]
                if len(n) == 1:
                    self.duration = shortest_rhythm
                else:
                    for i in range(len(n)-1):
                        if n[i+1] == '_':
                            self.duration = self.duration * 2
                        elif n[i+1] == '.':
                            self.duration = self.duration * (1 + 0.5)               
        elif n[0] == '+' or n[0] == '-':
            if len(n) >= 3 and (n[2] == 'b' or n[2] == '#'):
                self.pitch = ''.join(n[0:3])
                if len(n) == 3:
                    self.duration = shortest_rhythm
                else:
                    for i in range(len(n)-3):
                        if n[i+3] == '_':
                            self.duration = self.duration * 2
                        elif n[i+3] == '.':
                            self.duration = self.duration * (1 + 0.5)  
            else:
                self.pitch = ''.join(n[0:2])
                if len(n) == 2:
                    self.duration = shortest_rhythm
                else:
                    for i in range(len(n)-2):
                        if n[i+2] == '_':
                            self.duration = self.duration * 2
                        elif n[i+2] == '.':
                            self.duration = self.duration * (1 + 0.5) 


def preprocess_dataset(): 
    path = './music_dataset'
    files = os.listdir(path)
    melodies = []
    for file in files:
        with open(path+'/'+file, "r", encoding = 'gbk') as f:
            melody = Melody()
            detect_error = 0
            mel_mutex = 0 # 1 when encounter MEL, otherwise 0
            for line in f:
                line = line.replace('\n','') 
                if line: # if not encounter a blank line
                    if line[0:3] == 'KEY':
                        melody.get_shortest_rhythm(line)
                    elif line[0:3] == 'MEL': # Read the first line of MEL
                        mel_entry_list = []
                        line = line.strip()
                        mel_entry_list.append(line + '\n')
                        mel_mutex = 1
                    elif mel_mutex == 1: # Read the other lines of MEL
                        line = line.strip()
                        mel_entry_list.append(line + '\n')
                        if line[-2:] == ">>":
                            mel_entry = "".join(mel_entry_list)[4:-6]
                            if ("^" in mel_entry) == True:
                                detect_error = 1
                            melody.get_notes_labels(mel_entry)
                            mel_mutex = 0
                   
                elif not line: # if encounter a blank line
                    melody.get_features()
                    if len(melody.features) != 0:
                        if detect_error == 0:
                            melodies.append(melody)
                            melody = Melody()
                        elif detect_error == 1:
                            melody = Melody()
                            detect_error = 0


    return melodies

melodies = preprocess_dataset()

X_music = []
Y_music = []

for i in range(len(melodies)):
    X_music.append(melodies[i].features)
    Y_music.append(melodies[i].labels)

#counting words
noteFreq = collections.Counter()
for melody in X_music:
    noteFreq.update(melody)

notePairs = sorted(noteFreq.items(), key = lambda x: -x[1])
notes, _ = zip(*notePairs)
noteNum = len(notes)

note2id = dict(zip(notes, range(noteNum))) #word to ID

X = X_music # used for creating vocabulary word2id
Y = Y_music

random.seed(0)
random.shuffle(X_music)
random.seed(0)
random.shuffle(Y_music)

X_test = X_music[:400]
Y_test = Y_music[:400]

X_valid = X_music[400:800]
Y_valid = Y_music[400:800]

X_music = X_music[800:]
Y_music = Y_music[800:]

X_music_original = deepcopy(X_music)
Y_music_original = deepcopy(Y_music)

print('The size of music data is {}'.format(len(X_music)))
print(len(note2id))