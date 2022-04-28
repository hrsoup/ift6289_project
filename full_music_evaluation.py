import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import pickle

from bilstm_model import BILSTM


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
    scores = scores.view(-1, scores.shape[2])
    scores = scores[mask]
    loss = F.cross_entropy(scores, labels)

    return loss

class evaluation():
    def __init__(self, word2id):
        # Hyper parameters
        self.batch_size = 64
        self.epoches = 10

        # solid parameters
        self.vocab_size = len(word2id) + 1
        self.word2id = word2id

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
                with open("./models/full_music.pkl", "wb+") as f:
                    pickle.dump(self.model, f)

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

    def test(self, test_word_lists, test_label_lists, word2id):
        word_lists = str2id(test_word_lists, word2id)

        indices = sorted(range(len(word_lists)), key=lambda k: len(word_lists[k]), reverse=True)
        lengths = [len(word_lists[indices[i]]) for i in range(len(indices))]
        word_lists.sort(key=lambda x: len(x), reverse=True)
        test_label_lists.sort(key=lambda x: len(x), reverse=True)

        features = []
        for i in range(len(word_lists)):
            features.append(torch.Tensor(word_lists[i]))
        features = pad_sequence(features, batch_first=True, padding_value=len(word2id))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        features = features.to(torch.int64).to(device)

        with open("./models/full_music.pkl", "rb") as f:
            self.model = pickle.load(f)          

        with torch.no_grad():
            self.model.eval()
            scores = self.model.forward(features) 
            _, pred_labels = torch.max(scores, dim=2)

        # recover the original length
        pred_label_lists = []
        for i in range(len(pred_labels)):
            label_list = [pred_labels[i][j].item() for j in range(lengths[i])]
            pred_label_lists.append(label_list)

        return pred_label_lists, test_label_lists