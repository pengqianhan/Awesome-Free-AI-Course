import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import numpy as np
import time
# Initialize wandb
train_time = time.strftime('%m%d_%H_%M',time.localtime(time.time()))

wandb.init(project="mlp-training", name=f"{train_time}")

# Generate synthetic dataset
def generate_data(n_samples=1000):
    X = torch.randn(n_samples, 10)  # 10 input features
    # Generate target: sum of first three features plus some noise
    y = X[:, 0] + X[:, 1] + X[:, 2] + torch.randn(n_samples) * 0.1
    return X, y.reshape(-1, 1)

# Define the MLP model
class TwoLayerMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerMLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

# Training parameters
input_size = 10
hidden_size = 64
output_size = 1
learning_rate = 0.01
batch_size = 32
epochs = 10

# Generate data
X_train, y_train = generate_data(1000)
X_val, y_val = generate_data(200)

# Create data loaders
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

# Initialize model, loss function, and optimizer
model = TwoLayerMLP(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Log model architecture and hyperparameters
wandb.config.update({
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "epochs": epochs,
    "hidden_size": hidden_size
})

# Training loop
epochloss = {'train': [], 'val': []}
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            val_loss += criterion(outputs, batch_y).item()
    
    # Calculate average losses
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)

    epochloss['train'].append(avg_train_loss)
    epochloss['val'].append(avg_val_loss)
    # Log metrics to wandb
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss
    })
    
    print(f"Epoch [{epoch+1}/{epochs}] - "
          f"Train Loss: {avg_train_loss:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}")
    # save epoch loss
    np.save('epochloss.npy', epochloss)

# Close wandb run
wandb.finish()
