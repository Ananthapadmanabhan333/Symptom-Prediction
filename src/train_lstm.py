import torch
import torch.nn as nn
import torch.optim as optim
from preprocess import load_data
from sklearn.preprocessing import StandardScaler
import numpy as np

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.sig(self.fc(out))

def train_lstm():
    df = load_data()
    X = df.drop("mood_score", axis=1).values
    y = df["mood_score"].values

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = X[:, np.newaxis, :]  # shape: (samples, seq_len=1, features)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y.reshape(-1,1), dtype=torch.float32)

    model = LSTMPredictor(input_dim=X.shape[2])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    return model
