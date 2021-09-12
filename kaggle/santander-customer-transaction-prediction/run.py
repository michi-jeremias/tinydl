# Imports
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.utils.data import DataLoader

from auxiliary.utils import get_predictions
from data.data import get_data, get_submission
from data.loader import Loader
from model.model import NN2, NN3
from model.utils import init_normal, init_xavier
from trainer.trainer import Trainer

# Device
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyperparameters
BATCH_SIZE = 1024


# Data
train_ds, val_ds, test_ds, test_ids = get_data(
    train="aug_train.csv", test="aug_test.csv", submission=True)
train_loader = DataLoader(
    dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(dataset=test_ds, batch_size=BATCH_SIZE)
loader = Loader(train_loader, val_loader, test_loader)

# Model, Optimizer
# model = NN2(input_size=400, hidden_dim=100).to(DEVICE)
model = NN2(input_size=400, hidden_dim=100)
optimizer = optim.Adam(
    params=model.parameters(),
    lr=2e-3,
    weight_decay=1e-4)

model.apply(init_normal)


# Loss
loss_fn = nn.BCELoss()


trainer = Trainer(
    model=model,
    optimizer=optim.Adam(
        params=model.parameters(),
        lr=2e-3,
        weight_decay=1e-4
    ),
    loader=loader,
    loss_fn=nn.BCELoss(),
    metrics_fn=metrics.roc_auc_score
)

trainer.train(20)

# # Train
# def train(num_epochs=NUM_EPOCHS):
#     print("Start training.")
#     model.train()

#     for epoch in range(num_epochs):
#         predictions, actuals = get_predictions(val_loader, model, DEVICE)
#         print(
#             f"[{epoch + 1}/{num_epochs}] Validation ROC: "
#             f"{metrics.roc_auc_score(actuals, predictions)}")

#         for batch_idx, (data, targets) in enumerate(train_loader):
#             data = data.to(DEVICE)
#             targets = targets.to(DEVICE)

#             # Forward
#             scores = model(data)
#             loss = loss_fn(scores, targets)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             if batch_idx == 0:
#                 print(loss)

#     print("Finished training.")


get_submission(
    model=trainer.model,
    loader=test_loader,
    test_ids=test_ids,
    device="cuda",
    filename="subtoday.csv")
