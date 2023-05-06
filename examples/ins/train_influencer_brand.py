import torch, argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset


from create_dataset import CustomDataset, ToNumpyArray

EPOCHS = 10

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
        x = self.inf_brand_encoder(x)
        return x


class AcceptRejectDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AcceptRejectDecoder, self).__init__()
        
        # input size 64, hidden size 64, output size 2
        self.hidden_size = hidden_size 
        self.output_size = output_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        print('ruming see input', input.shape)
        embedded = self.embedding(input).view(1, 32, -1)
        output = embedded
        print('ruming see output', output.shape)
        print('ruming see hidden', hidden.shape)
        output, hidden = self.gru(output, hidden.view(1,32,-1))
        output = self.softmax(self.out(output[0]))
        print('ruming see final output', output.shape)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
               

class AcceptRejectMultiheadDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_targets):
        super(AcceptRejectMultiheadDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_targets = num_targets
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, num_classes + num_targets*2)
        
        self.classifier = nn.Sequential(
            nn.Linear(input_size, num_classes),
            nn.Softmax(dim=1)
        )
        self.regressor = nn.Sequential(
            nn.Linear(input_size, num_targets*2),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Separate classification and regression outputs
        classes = x[:, :self.num_classes]
        targets = x[:, self.num_classes:].view(-1, self.num_targets, 2)
        
        # Softmax classification
        cls_out = self.classifier(x)
        
        # Regress targets
        tgt_out = self.regressor(x)
        tgt_out = tgt_out.view(-1, self.num_targets, 2)
        
        # Return all outputs
        return cls_out, tgt_out

    
class MTLAcceptRejectModel(nn.Module):
    def __init__(self, input_size, encoder_hidden_size, encoder_output_size, decoder_hidden_size):
        super(MTLAcceptRejectModel, self).__init__()

        self.encoder = InfluencerBrandEncoder(input_size, encoder_hidden_size, encoder_output_size)
        self.decoder = AcceptRejectDecoder(encoder_output_size, decoder_hidden_size, 2)
#         self.category_feature_names = ['status', 'country_x', 'macromicro', 'listedby']
#         self.numerical_feature_names = ['decision_media_gained', 'decision_media_total',
#        'decision_followers_gained', 'decision_followers_total',
#        'decision_following_gained', 'decision_following_total',
#       'aftercamp_media_gained', 'aftercamp_media_total',
#        'aftercamp_followers_gained', 'aftercamp_followers_total',
#        'aftercamp_following_gained', 'aftercamp_following_total',
#       'follower_num']
        
    def forward(self, x, target=None):
#         x = self.encode_input(x)
        print('ruming see x:', x.shape)
        encoded = self.encoder(x)
        print('ruming see encode', encoded.shape)
        output, hidden = self.decoder(torch.zeros(size=[32,1],dtype=torch.long), encoded)

        if target is not None:
            loss = nn.CrossEntropyLoss()
            print('ruming see target', target.shape)
            loss_val = loss(output[:, 0].view(1,-1), target.view(1,-1))
            return output, loss_val
        else:
            return output
    
#     def encode_input(self, x):
#         # Encode categorical strings as one-hot vectors'
#         category_features_list = []
#         for feat in self.category_feature_names:
#             category_features = pd.get_dummies(x[feat]).values
#             category_features_list.append(category_features)

#         # Normalize numerical features
#         numerical_features_list = []
#         for feat in self.numerical_feature_names:
#             numerical_features = x[[feat]].values.astype(np.float32)
#             numerical_features = (numerical_features - numerical_features.mean(axis=0)) / numerical_features.std(axis=0)

#         # Concatenate categorical and numerical features
#         features = np.concatenate(category_features + numerical_features, axis=1)
        
#         print('ruming_see feature size', features.size())
#         return features

    def predict(self, x):
        with torch.no_grad():
            encoded = self.encoder(x)
            output, hidden = self.decoder(torch.tensor([0]), encoded)
            return output.argmax().item()




class MTLAcceptRejectMultiheadModel(nn.Module):
    def __init__(self, input_size, encoder_hidden_size, encoder_output_size, decoder_hidden_size):
        super(MTLAcceptRejectMultiheadModel, self).__init__()

        self.encoder = InfluencerBrandEncoder(input_size, encoder_hidden_size, encoder_output_size)
        self.decoder = AcceptRejectMultiheadDecoder(encoder_output_size, decoder_hidden_size, 2, 2)
#         self.category_feature_names = ['status', 'country_x', 'macromicro', 'listedby']
#         self.numerical_feature_names = ['decision_media_gained', 'decision_media_total',
#        'decision_followers_gained', 'decision_followers_total',
#        'decision_following_gained', 'decision_following_total',
#       'aftercamp_media_gained', 'aftercamp_media_total',
#        'aftercamp_followers_gained', 'aftercamp_followers_total',
#        'aftercamp_following_gained', 'aftercamp_following_total',
#       'follower_num']
        
    def forward(self, x, target=None):
#         x = self.encode_input(x)
        print('ruming see x:', x.shape)
        encoded = self.encoder(x)
        print('ruming see encode', encoded.shape)
        cls_out, tgt_out = self.decoder(torch.zeros(size=[32,1],dtype=torch.long), encoded)

        if target is not None:
            loss = nn.CrossEntropyLoss()
            print('ruming see target', target.shape)
            loss_val = loss(cls_out[:, 0].view(1,-1), target.view(1,-1))
            return cls_out, loss_val
        else:
            return cls_out
    
#     def encode_input(self, x):
#         # Encode categorical strings as one-hot vectors'
#         category_features_list = []
#         for feat in self.category_feature_names:
#             category_features = pd.get_dummies(x[feat]).values
#             category_features_list.append(category_features)

#         # Normalize numerical features
#         numerical_features_list = []
#         for feat in self.numerical_feature_names:
#             numerical_features = x[[feat]].values.astype(np.float32)
#             numerical_features = (numerical_features - numerical_features.mean(axis=0)) / numerical_features.std(axis=0)

#         # Concatenate categorical and numerical features
#         features = np.concatenate(category_features + numerical_features, axis=1)
        
#         print('ruming_see feature size', features.size())
#         return features

    def predict(self, x):
        with torch.no_grad():
            encoded = self.encoder(x)
            cls_out, tgt_out = self.decoder(torch.tensor([0]), encoded)
            return cls_out.argmax().item()


def main():
    # Example usage
    train_file = './acceptance_decision_decisionreputation_postrep_performance_0322.csv'
    test_file = './acceptance_decision_decisionreputation_postrep_performance_0322.csv'
    batch_size = 32
    input_size = 388
    encoder_hidden_size = 128
    encoder_output_size = 64
    decoder_hidden_size = 64

    train_dataset = CustomDataset(train_file)
    test_dataset = CustomDataset(test_file)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = MTLAcceptRejectModel(input_size, encoder_hidden_size, encoder_output_size, decoder_hidden_size)

    optimizer = optim.Adam(model.parameters())

    # Training loop
    for epoch in range(EPOCHS):
        for i, batch in enumerate(train_loader):
            x_batch, y_batch = batch

            optimizer.zero_grad()
            output, loss_val = model(x_batch, y_batch)
            loss_val.backward()
            optimizer.step()

            # Print loss
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, EPOCHS, i+1, len(train_loader), loss_val.item()))

                
    # Testing loop
    with torch.no_grad():
        correct = 0
        total = 0

        for batch in test_loader:
            x_batch, y_batch = batch

            output = model(x_batch)
            predicted = output.argmax(dim=1)

            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        print('Accuracy: {:.2f}%'.format(100 * correct / total))
if __name__ == "__main__":
    main()