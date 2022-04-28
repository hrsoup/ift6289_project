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

class evaluation_test():
    def __init__(self, word2id, language_type):
        # Hyper parameters
        self.batch_size = 64
        self.epoches = 30

        # other parameters
        self.vocab_size = len(word2id) + 1 # Assuming there is 'PAD' in the end of a vocab
        self.out_size = 2
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
        features = features.to(torch.int64).to(self.device)


        pretrained_dict = torch.load("./models/" + self.language_type + "_music.pkl")
        self.model.load_state_dict(pretrained_dict)
  

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
