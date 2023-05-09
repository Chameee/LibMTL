import pandas as pd
import numpy as np 
from create_dataset_v2 import CustomDataset, ToNumpyArray
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# define encoder and decoders
class InfluencerBrandEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(InfluencerBrandEncoder, self).__init__()

        self.hidden_size = hidden_size

        self.inf_brand_encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.float()
        accept_head_output = self.inf_brand_encoder(x)
        accept_head_output = accept_head_output.type(torch.LongTensor)        
        return accept_head_output


class InfluencerBrandDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super(InfluencerBrandDecoder, self).__init__()
        
        # input size 64, hidden size 64, output size 2
        self.input_size = input_size
        self.hidden_size = hidden_size 
        self.output_size = output_size

        self.embedding = nn.Embedding(input_size, hidden_size).to(device)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.init_weights()

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(self.output_size, 32, -1)
        output = embedded
        #print(output.shape)
        output, hidden = self.gru(output)
        output = self.softmax(self.linear(output[0]))
        return output, hidden

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                module.bias.data.zero_()
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():   
                    if 'weight' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
    
#     def initHidden(self, batch_size):
#         return torch.zeros(size=[batch_size, 32, self.hidden_size], dtype=torch.long)


    
class MTLInfluencer(nn.Module):
    def __init__(self, input_size, encoder_hidden_size, encoder_output_size, decoder_hidden_size, output_size, device):
        super().__init__()
        self.encoder = InfluencerBrandEncoder(input_size, encoder_hidden_size, encoder_output_size)
        self.decoder = InfluencerBrandDecoder(encoder_output_size, decoder_hidden_size, output_size, device)
        self.output_size =  output_size
        self.decoder_hidden_size = decoder_hidden_size
    def forward(self, x):
        x = x.float()
        encoded = self.encoder(x)
        #accept_head, hidden = self.decoder(torch.zeros(size=[32,1],dtype=torch.long), encoded)
#         decoder_hidden = self.decoder.initHidden(self.output_size)
        accept_head, hidden = self.decoder(torch.zeros(size=[32,1],dtype=torch.long), encoded)
        revenue_head, hidden = self.decoder(torch.zeros(size=[32,1],dtype=torch.long), encoded)
        reputation_head, hidden = self.decoder(torch.zeros(size=[32,1],dtype=torch.long), encoded)
        return accept_head.float(), revenue_head.float(), reputation_head.float()

def main():
    # Example usage
    train_file = 'acceptance_decision_decisionreputation_postrep_performance_0322.csv'
    test_file = 'acceptance_decision_decisionreputation_postrep_performance_0322.csv'
    batch_size = 64
    input_size = 388
    encoder_hidden_size = 64
    encoder_output_size = 32
    decoder_hidden_size = 16
    decoder_output_size = 8
    n_epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    train_dataset = CustomDataset(train_file)
    test_dataset = CustomDataset(test_file)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print("done loading data!")
    print("device:", device)
    
    model = MTLInfluencer(input_size, encoder_hidden_size, encoder_output_size, decoder_hidden_size, batch_size, device).to(device=device)

    accept_loss = nn.BCELoss()
    revenue_loss = nn.BCELoss()
    reputation_loss = nn.BCELoss()
    sig = nn.Sigmoid()

    optimizer  = optim.Adam(model.parameters(), lr=1e-4)  
    
    for epoch in range(n_epochs):
        model.train()
        total_training_loss = 0

        for i, batch in enumerate(train_loader):
            x_batch, y1_batch, y2_batch, y3_batch = batch
            # if len(x_batch) < batch_size:
            #     continue
            # else:
            inputs = x_batch.to(device=device)
            accept_label = y1_batch.to(device=device)
            revenue_label = y2_batch.to(device=device)
            reputation_label = y3_batch.to(device=device)

            optimizer.zero_grad()
            accept_output, revenue_output, reputation_output = model(inputs)
            loss_1 = accept_loss(sig(accept_output[0]).unsqueeze(1), accept_label)
            loss_2 = revenue_loss(sig(revenue_output[0]).unsqueeze(1), revenue_label)
            loss_3 = reputation_loss(sig(reputation_output[0].unsqueeze(1)), reputation_label)

            loss = loss_1 + loss_2 + loss_3
            loss.backward()
            optimizer.step()

            total_training_loss += loss
                    # Print loss
            if (i + 1) % 1000 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, n_epochs, i+1, len(train_loader), loss))



    with torch.no_grad():
        y1_predict_array = []
        y1_label_array = []

        y2_predict_array = []
        y2_label_array = []

        y3_predict_array = []
        y3_label_array = []
        for batch in test_loader:
            x_batch2, y1_batch, y2_batch, y3_batch = batch
            x_batch2 = x_batch2.to(device=device)
            accept_output, revenue_output, reputation_output = model(x_batch2)
            predicted_accept = accept_output[0].argmax(dim=1)
            predicted_revenue = revenue_output[0].argmax(dim=1)
            predicted_reputation = reputation_output[0].argmax(dim=1)

            y1_predict_array.extend(predicted_accept.detach().cpu().numpy())
            y1_label_array.extend(y1_batch.detach().cpu().numpy())

            y2_predict_array.extend(predicted_revenue.detach().cpu().numpy())
            y2_label_array.extend(y2_batch.detach().cpu().numpy())

            y3_predict_array.extend(predicted_reputation.detach().cpu().numpy())
            y3_label_array.extend(y3_batch.detach().cpu().numpy())

        accuracy_accept = accuracy_score(y1_label_array, y1_predict_array)
        precision_score_accept = precision_score(y1_label_array, y1_predict_array)
        recall_score_accept = recall_score(y1_label_array, y1_predict_array)
        f1_score_accept = f1_score(y1_label_array, y1_predict_array)
        print('Accuracy, precision, recall, f1 of accept: {:.2f} {:.2f} {:.2f} {:.2f}%'.format(accuracy_accept, precision_score_accept, recall_score_accept, f1_score_accept))


        accuracy_revenue = accuracy_score(y2_label_array, y2_predict_array)
        precision_score_revenue = precision_score(y2_label_array, y2_predict_array)
        recall_score_revenue = recall_score(y2_label_array, y2_predict_array)
        f1_score_revenue = f1_score(y2_label_array, y2_predict_array)
        print('Accuracy, precision, recall, f1 of accept: {:.2f} {:.2f} {:.2f} {:.2f}%'.format(accuracy_revenue, precision_score_revenue, recall_score_revenue, f1_score_revenue))


        accuracy_reputation = accuracy_score(y3_label_array, y3_predict_array)
        precision_score_reputation = precision_score(y3_label_array, y3_predict_array)
        recall_score_reputation = recall_score(y3_label_array, y3_predict_array)
        f1_score_reputation = f1_score(y3_label_array, y3_predict_array)
        print('Accuracy, precision, recall, f1 of accept: {:.2f} {:.2f} {:.2f} {:.2f}%'.format(accuracy_reputation, precision_score_reputation, recall_score_reputation, f1_score_reputation))
    
if __name__ == "__main__":
    main()