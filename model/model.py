# Hierarchical CNN-RNN model for zooplankton image classification.
#
# This module defines the `HierarchicalCNNRNN` architecture, which combines a ResNet18 image
# encoder with an LSTM decoder to predict taxonomy-aware label sequences.


import torch
import torch.nn as nn
import torchvision.models as models


# Neural network combining a CNN encoder and RNN decoder for hierarchical classification
class HierarchicalCNNRNN(nn.Module):
    def __init__(self, num_classes, embed_size=128, hidden_size=256, num_layers=1):
        super(HierarchicalCNNRNN, self).__init__()

        # Use a pre-trained ResNet18 as the image encoder
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for param in resnet.parameters():
            param.requires_grad = False

        modules = list(resnet.children())[:-1]
        self.cnn = nn.Sequential(*modules)

        # Project CNN features to match RNN hidden size
        self.cnn_to_rnn = nn.Linear(resnet.fc.in_features, hidden_size)

        # Embedding layer for label inputs
        self.embedding = nn.Embedding(num_classes, embed_size)

        # LSTM decoder to generate the hierarchy path
        self.rnn = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Output layer predicting the next node in the hierarchy
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, images, captions=None):
        batch_size = images.size(0)

        # Encode images to feature vectors
        features = self.cnn(images)

        features = features.view(features.size(0), -1)

        # Initialize RNN hidden state with image features
        h_0 = self.cnn_to_rnn(features).unsqueeze(0)
        c_0 = torch.zeros_like(h_0)

        if captions is not None:
            # Training mode: use teacher forcing with provided captions
            embeddings = self.embedding(captions[:, :-1])

            outputs, _ = self.rnn(embeddings, (h_0, c_0))

            predictions = self.fc(outputs)
            return predictions

        else:
            # Inference mode: autoregressively predict the next node in the hierarchy
            start_token_index = 1

            start_token = torch.tensor(
                [start_token_index] * batch_size, device=images.device
            ).unsqueeze(1)

            inputs = self.embedding(start_token)
            hidden = (h_0, c_0)
            outputs = []

            max_depth = 5
            for _ in range(max_depth):
                out, hidden = self.rnn(inputs, hidden)
                prediction = self.fc(out)

                predicted_class = prediction.argmax(2)
                outputs.append(predicted_class)

                inputs = self.embedding(predicted_class)

            return torch.cat(outputs, dim=1)
