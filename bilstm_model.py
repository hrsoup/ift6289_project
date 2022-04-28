import torch.nn as nn

class BILSTM(nn.Module):
    def __init__(self, vocab_size):
        super(BILSTM, self).__init__()
        self.emb_size = 512
        self.hidden_size = 512
        self.lstm_layer_number = 5

        self.linear1 = nn.Embedding(vocab_size, self.emb_size)
        self.bilstm = nn.LSTM(self.emb_size, self.hidden_size, num_layers=self.lstm_layer_number, batch_first=True, bidirectional=True)
        self.linear2 = nn.Linear(2*self.hidden_size, 2)

    def forward(self, x):
        self.bilstm.flatten_parameters()
        emb_out = self.linear1(x)  # linear 1
        bilstm_out, _ = self.bilstm(emb_out) # bilstm
        scores = self.linear2(bilstm_out)  # linear 2
        return scores