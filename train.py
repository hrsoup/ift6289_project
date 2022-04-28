import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from bilstm_model import BILSTM
import pickle


class GetLoader(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        return x, y
    def __len__(self):
        return len(self.X)

def str2id(lists, maps):
    final = []
    for i in range(len(lists)):
        a = []
        for j in range(len(lists[i])):
            a.append(maps.get(lists[i][j]))
        final.append(a)
        
    return final     

def compute_loss(scores, labels):
    labels = labels.flatten()
    mask = (labels != 2)  # != 'PAD'
    labels = labels[mask]
    scores = scores.view(-1, scores.shape[2])[mask]
    loss = F.cross_entropy(scores, labels)

    return loss

class evaluation_train():
    def __init__(self, word2id, language_type):
        # Hyper parameters
        self.batch_size = 64
        self.epoches = 1000

        # other parameters
        self.vocab_size = len(word2id) + 1 # Assuming there is 'PAD' in the end of a vocab
        self.word2id = word2id
        self.language_type = language_type

        # init parameters for training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BILSTM(self.vocab_size).to(self.device)
    
    def fun(self, data):
        data.sort(key=lambda x: len(x[0]), reverse=True)
        features = []
        labels = []
        for feature, label in data:
            features.append(torch.Tensor(feature))
            labels.append(torch.Tensor(label))
        features = pad_sequence(features, batch_first=True, padding_value=len(self.word2id)) 
        labels = pad_sequence(labels, batch_first=True, padding_value=2)

        return features, labels


    def train(self, train_word_lists, train_label_lists, valid_word_lists, valid_label_lists, word2id):

        train_word = str2id(train_word_lists, word2id)

        train_word = [torch.Tensor(word_list) for word_list in train_word]
        train_label = [torch.Tensor(label_list) for label_list in train_label_lists]
        torch_data = GetLoader(train_word, train_label)

        data_loader = DataLoader(torch_data, batch_size=self.batch_size, shuffle=True, collate_fn=self.fun)

        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        best_val_loss = float('inf')

        for i in range(self.epoches):
            losses = 0
            for (words, labels) in data_loader:
        
                words = words.to(torch.int64).to(self.device)
                labels = labels.to(torch.int64).to(self.device)
                self.model.train()
                scores = self.model(words)
                
                optimizer.zero_grad()
                loss = compute_loss(scores, labels)
                loss = loss.to(self.device)
                loss.backward()
                optimizer.step()     
                losses += loss.item()

            # valid
            val_loss = self.validate(valid_word_lists, valid_label_lists, word2id)
            print("Epoch: {}, Valid Loss:{:.4f}".format(i+1, val_loss))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), "./models/" + self.language_type + "_music.pkl")


    def validate(self, valid_word_lists, valid_label_lists, word2id):
        valid_word_lists = str2id(valid_word_lists, word2id)

        word_lists = [torch.Tensor(word_list) for word_list in valid_word_lists]
        label_lists = [torch.Tensor(label_list) for label_list in valid_label_lists]
        torch_data = GetLoader(word_lists, label_lists)

        data_loader = DataLoader(torch_data, batch_size=self.batch_size, shuffle=True, collate_fn=self.fun)
        val_losses = 0
        with torch.no_grad():    
            for (words, labels) in data_loader:

                words = words.to(torch.int64).to(self.device)
                labels = labels.to(torch.int64).to(self.device)
                self.model.eval()
                scores = self.model(words)
                loss = compute_loss(scores, labels).to(self.device)
                val_losses += loss.item()

        return val_losses
