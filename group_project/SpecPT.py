# SpecPT.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer
from scipy.ndimage import gaussian_filter1d
import numpy as np
from accelerate import Accelerator
import pandas as pd


# SpecPT Model Definition
class SpecPT(nn.Module):
    def __init__(self, input_size=7781, d_model=512, nhead=8, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=2048, dropout=0.1):
        super(SpecPT, self).__init__()

        # Adjusted initial convolutional blocks for feature extraction with new kernel sizes
        self.conv1 = nn.Conv1d(1, 64, kernel_size=41, stride=2, padding=20)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=21, stride=2, padding=10)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=11, stride=2, padding=5)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Transformer Encoder Layer
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Transformer Decoder Layer
        decoder_layer = TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Dummy input to calculate the output size of the convolutional layers dynamically
        dummy_input = torch.zeros(1, 1, input_size)
        dummy_output = self.forward_conv(dummy_input)
        output_size = dummy_output.numel() // dummy_input.shape[0]  # Calculate total conv output size

        # Projection to d_model size
        self.proj_to_d_model = nn.Linear(output_size, d_model)

        # Final Linear Reconstruction Layers
        self.linear1 = nn.Linear(d_model, output_size)
        self.linear2 = nn.Linear(output_size, input_size)

    def forward_conv(self, x):
        # Pass through Adjusted Convolutional Blocks
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        return x

    def forward(self, x):
        # Input shape (batch_size, input_size)
        x = x.unsqueeze(1)  # Adding channel dimension (batch_size, 1, input_size)

        # Pass through Convolutional Blocks
        x = self.forward_conv(x)
        
        # Flatten and project to transformer dimension
        x = x.flatten(start_dim=1)
        x = self.proj_to_d_model(x)
        
        # Transformer Encoder & Decoder
        x = x.unsqueeze(0)  # Add sequence dimension for transformer (src_len, batch_size, d_model)
        encoded_features = self.transformer_encoder(x)
        decoded_features = self.transformer_decoder(encoded_features, encoded_features)
        
        # Final Linear Layers
        x = decoded_features.squeeze(0)  # Remove sequence dimension
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        
        return x


# Dataset Loader for Autoencoder
class CustomLoadDataset_Autoencoder(Dataset):
    def __init__(self, df):
        x = []
        target_id = []
        for index, row in df.iterrows():
            fl = row['spec']
            if np.median(fl) > 0:
                fl = fl / np.median(fl)
                x.append(fl)
                target_id.append(np.array([row['TARGETID']]))

        self.X = torch.from_numpy(np.stack(x, axis=0))
        self.t_id = torch.from_numpy(np.stack(target_id, axis=0))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].float(), self.X[idx].float(), idx, self.t_id[idx]


# Evaluate Function
def evaluate(net, loader, criterion, accelerator):
    net.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for X, Y, idx, t_id in loader:
            X, Y = accelerator.prepare(X, Y)
            output = net(X)
            loss = criterion(output, Y)
            total_loss += loss.item()
            total_samples += X.size(0)

    average_loss = total_loss / total_samples
    return average_loss


# Custom NMAD Loss Function
class NMADLoss(nn.Module):
    def __init__(self, normalization_factor='mad'):
        super(NMADLoss, self).__init__()
        self.normalization_factor = normalization_factor

    def forward(self, input, target):
        mad = torch.mean(torch.abs(input - target))
        if self.normalization_factor == 'mad':
            normalization = torch.median(torch.abs(target - torch.median(target)))
        elif self.normalization_factor == 'std':
            normalization = torch.std(target)
        else:
            raise ValueError("Invalid normalization factor. Use 'mad' or 'std'.")
        nmad = mad / normalization
        return nmad


# Define the activation functions
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# # Residual MLP Block Definition
# class ResidualMLPBlock(nn.Module):
#     def __init__(self, in_features, out_features, dropout_rate=0.1):
#         super(ResidualMLPBlock, self).__init__()
#         self.fc1 = nn.Linear(in_features, out_features)
#         self.bn1 = nn.BatchNorm1d(out_features)
#         self.swish = Swish()
#         self.dropout = nn.Dropout(dropout_rate)
#         self.fc2 = nn.Linear(out_features, out_features)
#         self.bn2 = nn.BatchNorm1d(out_features)
#         self.residual_connection = nn.Linear(in_features, out_features)
    
#     def forward(self, x):
#         residual = self.residual_connection(x)
#         x = self.fc1(x)
#         x = self.bn1(x)
#         x = self.swish(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.bn2(x)
#         return self.swish(x + residual)


# # SpecPT for Redshift Prediction Model
# class SpecPTForRedshift(nn.Module):
#     def __init__(self, pretrained_model, output_features=1, num_mlp_blocks=3, mlp_dim=256, dropout_rate=0.1):
#         super(SpecPTForRedshift, self).__init__()
#         self.encoder = pretrained_model.transformer_encoder
#         self.proj_to_d_model = pretrained_model.proj_to_d_model
#         self.forward_conv = pretrained_model.forward_conv
        
#         for param in self.encoder.parameters():
#             param.requires_grad = False
#         for param in self.proj_to_d_model.parameters():
#             param.requires_grad = False
#         for layer in [pretrained_model.conv1, pretrained_model.conv2, pretrained_model.conv3, pretrained_model.bn1, pretrained_model.bn2, pretrained_model.bn3]:
#             for param in layer.parameters():
#                 param.requires_grad = False
        
#         self.mlp_blocks = nn.Sequential(
#             *[ResidualMLPBlock(mlp_dim if i > 0 else 512, mlp_dim, dropout_rate) for i in range(num_mlp_blocks)]
#         )
        
#         self.prediction = nn.Sequential(
#             nn.Linear(mlp_dim, output_features),
#             nn.Softplus()
#         )
    
#     def forward(self, x):
#         x = x.unsqueeze(1)
#         x = self.forward_conv(x)
#         x = x.flatten(start_dim=1)
#         x = self.proj_to_d_model(x)
#         x = x.unsqueeze(0)
#         encoded_features = self.encoder(x)
#         encoded_features = encoded_features.squeeze(0)
#         x = self.mlp_blocks(encoded_features)
#         redshift = self.prediction(x)
#         return redshift

class SpecPTForRedshift(nn.Module):
    def __init__(self, pretrained_model, output_features=1, num_mlp_blocks=5, mlp_dim=512, dropout_rate=0.2):
        super(SpecPTForRedshift, self).__init__()
        
        self.encoder = pretrained_model.transformer_encoder
        self.proj_to_d_model = pretrained_model.proj_to_d_model
        self.forward_conv = pretrained_model.forward_conv
        
        # Fine-tune the last few layers of the encoder
        for param in list(self.encoder.parameters())[-4:]:
            param.requires_grad = True
        
        self.mlp_blocks = nn.Sequential(
            *[ResidualMLPBlock(mlp_dim if i > 0 else 512, mlp_dim, dropout_rate) for i in range(num_mlp_blocks)]
        )
        
        self.prediction = nn.Sequential(
            nn.Linear(mlp_dim, mlp_dim // 2),
            Swish(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim // 2, output_features),
            nn.Softplus()
        )
        
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.forward_conv(x)
        x = x.flatten(start_dim=1)
        x = self.proj_to_d_model(x)
        x = x.unsqueeze(0)
        
        encoded_features = self.encoder(x)
        encoded_features = encoded_features.squeeze(0)
        
        # Apply attention mechanism
        attn_output, _ = self.attention(encoded_features, encoded_features, encoded_features)
        x = attn_output + encoded_features  # Residual connection
        
        x = self.mlp_blocks(x)
        redshift = self.prediction(x)
        return redshift

class ResidualMLPBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate):
        super(ResidualMLPBlock, self).__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim)
        self.swish = Swish()
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        residual = x
        x = self.swish(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = x + residual  # Residual connection
        x = self.layer_norm(x)
        return self.swish(x)


# Dataset Loader for Redshift
class CustomLoadDataset_Redshift(Dataset):
    def __init__(self, df):
        x = []
        y = []
        target_id = []
        
        if 'TARGETID' in df.columns:
            tid_column_name = 'TARGETID'
        else:
            tid_column_name = 'target_id'
        
        if 'Z' in df.columns:
            redshift_column_name = 'Z'
        else:
            redshift_column_name = 'z'
            
        for index, row in df.iterrows():
            fl = row['spec']
            if np.median(fl) > 0:
                fl = fl / np.median(fl)
                x.append(fl)
                y.append(np.array([row[redshift_column_name]]))
                target_id.append(np.array([row[tid_column_name]]))

        self.X = torch.from_numpy(np.stack(x, axis=0))
        self.Y = torch.from_numpy(np.stack(y, axis=0))
        self.t_id = torch.from_numpy(np.stack(target_id, axis=0))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].float(), self.Y[idx].float(), idx, self.t_id[idx]

# Dataset Loader for Redshift (HST)
class CustomLoadDataset_Redshift_HST(Dataset):
    def __init__(self, df):
        x = []
        y = []
        target_id = []
            
        for index, row in df.iterrows():
            fl = row['flux']
            if np.median(fl) > 0:
                fl = fl / np.median(fl)
                x.append(fl)
                y.append(np.array([row['z']]))
                target_id.append(row['grism_id'])

        self.X = torch.from_numpy(np.stack(x, axis=0))
        self.Y = torch.from_numpy(np.stack(y, axis=0))
        self.t_id = target_id

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].float(), self.Y[idx].float(), idx, self.t_id[idx]


# Custom Load Dataset for Autoencoder (HST)
class CustomLoadDataset_HST(Dataset):
    def __init__(self, df):
        x = []
        idx = []
        target_id = []
        for index,row in df.iterrows():
            #Normalizing flux
#             fl = (row['spec'] - row['spec'].min())/(row['spec'].max() - row['spec'].min())
#             fl = gaussian_filter1d(row['spec'], sigma=3)
            fl = row['flux']
#             fl = gaussian_filter1d(fl, sigma=3)
            if np.median(fl) > 0:
                fl = fl/np.median(fl)
#                 if np.sqrt(np.sum(np.square(fl)))==0 or np.isnan(fl).any():
#                     continue
                x.append(fl)
                target_id.append(row['grism_id'])
            else:
                pass

        
            
        self.X = torch.from_numpy(np.stack(x,axis=0))
        self.t_id = target_id
        
#         self.Y = torch.from_numpy(np.stack(y,axis=0))
            
#         self.X = torch.from_numpy(np.stack(x,axis=0))
#         self.Y = torch.from_numpy(np.reshape(df['z'].values,(len(df['z'].values),1)))

#         self.idx = torch.from_numpy(np.reshape(df.index,(len(df.index),1)))
        # self.X = torch.from_numpy(np.random.random((size, 2, 8912)))
        # self.Y = torch.from_numpy(np.random.random((size, 1)))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].float(), self.X[idx].float(), idx, self.t_id[idx]