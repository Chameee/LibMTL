import pandas as pd
import numpy as np 
from create_dataset_v2 import CustomDataset, ToNumpyArray
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")
class MTLInfluencer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        #self.net = models.resnet18(pretrained=True)
        #self.n_features = self.net.fc.in_features
        self.hidden_size = hidden_size
        #self.net.fc = nn.Identity()
    
         
        self.shared_encoder =  nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )
        
        self.accept_decoder = nn.Sequential(
            nn.Linear(output_size, 1),
            nn.ReLU()
        )

        self.revenue_decoder = nn.Sequential(
            nn.Linear(output_size, 1),
            nn.ReLU()
        )
        
        self.reputation_decoder = nn.Sequential(
            nn.Linear(output_size, 1),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = x.float()
        encoded_x = self.shared_encoder(x)
        accept_head = self.accept_decoder(encoded_x)
        revenue_head = self.revenue_decoder(encoded_x)
        reputation_head = self.reputation_decoder(encoded_x)
        return accept_head, revenue_head, reputation_head
    
    

def main():
    # Example usage
    all_file = 'acc_v5.csv'
#     train_file = 'acc_v2.csv'
#     test_file = 'acc_v2.csv'
    batch_size = 64
    input_size = 140
    encoder_hidden_size = 64
    encoder_output_size = 32
    decoder_hidden_size = 32
    n_epochs = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device", device)
    
    all_dataset = CustomDataset(all_file)
#     train_dataset = CustomDataset(train_file)
#     test_dataset = CustomDataset(test_file)
    train_size = int(0.8 * len(all_dataset))
    test_size = len(all_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset=all_dataset, lengths=[train_size, test_size], generator=torch.manual_seed(0))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print("done loading data!")
    
    model = MTLInfluencer(input_size, encoder_hidden_size, encoder_output_size).to(device=device)

    accept_loss = nn.BCELoss()
    revenue_loss = nn.MSELoss()
    reputation_loss = nn.MSELoss()
    sig = nn.Sigmoid()

    optimizer  = optim.Adam(model.parameters(), lr=1e-4)  
    
        # Training loop
    for epoch in range(n_epochs):
        model.train()
        total_training_loss = 0

        for i, batch in enumerate(train_loader):
            x_batch, y1_batch, y2_batch, y3_batch = batch
            inputs = x_batch.to(device=device)

            accept_label = y1_batch.to(device=device)
            revenue_label = y2_batch.to(device=device)
            reputation_label = y3_batch.to(device=device)

            optimizer.zero_grad()
            accept_output, revenue_output, reputation_output = model(inputs)
            loss_1 = accept_loss(sig(accept_output), accept_label)
            loss_2 = revenue_loss(sig(revenue_output), revenue_label)
            loss_3 = reputation_loss(sig(reputation_output), reputation_label)

            loss = loss_1 + loss_2 + loss_3
            loss.backward()
            optimizer.step()

            total_training_loss += loss
                    # Print loss
            if (i + 1) % 1000 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, n_epochs, i+1, len(train_loader), loss))

    # Save ckpt
    torch.save(model.state_dict(), './saved_model.pkl')
    
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
#             predicted_accept = accept_output.argmax(dim=1)
#             predicted_revenue = revenue_output.argmax(dim=1)
#             predicted_reputation = reputation_output.argmax(dim=1)


            accept_output_cut = [1 if y > 0 else 0 for y in accept_output]
            revenue_output_cut = [1 if y > 0 else 0 for y in revenue_output]
            reputation_output_cut = [1 if y > 0 else 0 for y in reputation_output]
            
            y1_predict_array.extend(accept_output_cut)
            y1_label_array.extend(y1_batch.reshape(1,-1).tolist()[0])

            y2_predict_array.extend(revenue_output.cpu())
            y2_label_array.extend(y2_batch.reshape(1,-1).tolist()[0])

            y3_predict_array.extend(reputation_output.cpu())
            y3_label_array.extend(y3_batch.reshape(1,-1).tolist()[0])

        accuracy_accept = accuracy_score(y_true=y1_label_array, y_pred=y1_predict_array)
        precision_score_accept = precision_score(y_true=y1_label_array, y_pred=y1_predict_array)
        recall_score_accept = recall_score(y_true=y1_label_array, y_pred=y1_predict_array)
        f1_score_accept = f1_score(y_true=y1_label_array, y_pred=y1_predict_array)
        print('y1 Accuracy, precision, recall, f1 of accept: {:.2f} {:.2f} {:.2f} {:.2f}'.format(accuracy_accept, precision_score_accept, recall_score_accept, f1_score_accept))


        mse_revenue = mean_squared_error(y_true=y2_label_array, y_pred=y2_predict_array)
        r2_revenue = r2_score(y_true=y2_label_array, y_pred=y2_predict_array)
        print('y2 MSE, R2: {:.2f} {:.2f}'.format(mse_revenue, r2_revenue))


        mse_reputation = mean_squared_error(y_true=y3_label_array, y_pred=y3_predict_array)
        r2_reputation = r2_score(y_true=y3_label_array, y_pred=y3_predict_array)
        print('y3 MSE, R2: {:.2f} {:.2f}'.format(mse_reputation, r2_reputation))

if __name__ == "__main__":
    import os 
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    main()