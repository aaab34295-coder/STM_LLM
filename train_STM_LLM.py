import torch
import numpy as np
import pandas as pd
import argparse
import time
import util
import os
from util import *
import random
from model_STM_LLM import ST_LLM_topk_memory_nog2
from ranger21 import Ranger

#新的测试代码
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:180'
os.makedirs("./logs_new", exist_ok=True)

# =================== Arguments ===================
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--data", type=str, default="bike_pick")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lrate", type=float, default=1e-3)
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--input_dim", type=int, default=3)
parser.add_argument("--num_nodes", type=int, default=250)
parser.add_argument("--input_len", type=int, default=12)
parser.add_argument("--output_len", type=int, default=12)
parser.add_argument("--llm_layer", type=int, default=6)
parser.add_argument("--U", type=int, default=2)
parser.add_argument("--wdecay", type=float, default=1e-4)
parser.add_argument("--es_patience", type=int, default=100)
parser.add_argument("--topk", type=int, default=500)
parser.add_argument("--save", type=str, default="./logs_nog2/" + time.strftime("%Y-%m-%d-%H:%M:%S") + "-")
args = parser.parse_args()

# =================== Seed ===================
def seed_it(seed=6666):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_it()

# =================== Dataset ===================
raw_data = args.data
args.data = f"data/{args.data}"

if "bike" in raw_data:
    args.num_nodes = 250
elif "taxi" in raw_data:
    args.num_nodes = 266

device = torch.device(args.device)

adj_mx = load_graph_data(f"{args.data}/adj_mx.pkl")
dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
scaler = dataloader["scaler"]

save_path = args.save + raw_data + "/"
os.makedirs(save_path, exist_ok=True)

# =================== Trainer ===================
class Trainer:
    def __init__(self):
        self.model = ST_LLM_topk_memory_nog2(
            device, adj_mx, args.input_dim, args.num_nodes, args.input_len,
            args.output_len, args.llm_layer, args.U, topk=args.topk
        ).to(device)

        self.optimizer = Ranger(self.model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
        self.loss = util.MAE_torch
        self.clip = 5

        print("Total params:", self.model.param_num())
        print("Trainable params:", self.model.count_trainable_params())

    def train_step(self, x, y):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(x).transpose(1, 3)
        real = torch.unsqueeze(y, dim=1)
        pred = scaler.inverse_transform(out)
        loss = self.loss(pred, real, 0.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def eval_step(self, x, y):
        self.model.eval()
        out = self.model(x).transpose(1, 3)
        real = torch.unsqueeze(y, dim=1)
        pred = scaler.inverse_transform(out)
        loss = self.loss(pred, real, 0.0)
        return loss.item()

engine = Trainer()

# =================== Training ===================
best_val_loss = float("inf")
epochs_no_improve = 0
best_epoch = 0
history = []

print("===== Start Training =====")

for epoch in range(1, args.epochs + 1):

    train_losses = []
    for x, y in dataloader["train_loader"].get_iterator():
        x = torch.Tensor(x).to(device).transpose(1, 3)
        y = torch.Tensor(y).to(device).transpose(1, 3)[:, 0, :, :]
        train_losses.append(engine.train_step(x, y))

    val_losses = []
    for x, y in dataloader["val_loader"].get_iterator():
        x = torch.Tensor(x).to(device).transpose(1, 3)
        y = torch.Tensor(y).to(device).transpose(1, 3)[:, 0, :, :]
        val_losses.append(engine.eval_step(x, y))

    train_mae = np.mean(train_losses)
    val_mae = np.mean(val_losses)

    print(f"Epoch {epoch:03d} | Train MAE {train_mae:.4f} | Val MAE {val_mae:.4f}")

    history.append({"epoch": epoch, "train_mae": train_mae, "val_mae": val_mae})

    if val_mae < best_val_loss:
        best_val_loss = val_mae
        best_epoch = epoch
        epochs_no_improve = 0
        torch.save(engine.model.state_dict(), save_path + "best_model.pth")
        print("✓ Best model updated.")
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= args.es_patience:
        print("Early stopping triggered.")
        break

pd.DataFrame(history).to_csv(save_path + "train_log.csv", index=False)

# =================== Test ===================
print(f"\nLoading best model from epoch {best_epoch}")
engine.model.load_state_dict(torch.load(save_path + "best_model.pth"))
engine.model.eval()

outputs = []
realy = torch.Tensor(dataloader["y_test"]).to(device).transpose(1, 3)[:, 0, :, :]

with torch.no_grad():
    for x, y in dataloader["test_loader"].get_iterator():
        x = torch.Tensor(x).to(device).transpose(1, 3)
        preds = engine.model(x).transpose(1, 3)
        outputs.append(preds.squeeze())

yhat = torch.cat(outputs, dim=0)[:realy.size(0)]

mae, mape, rmse, wmape = [], [], [], []

for h in range(args.output_len):
    pred = scaler.inverse_transform(yhat[:, :, h])
    real = realy[:, :, h]
    m = util.metric(pred, real)
    print(f"Horizon {h+1:02d} | MAE {m[0]:.4f} RMSE {m[2]:.4f}")
    mae.append(m[0])
    mape.append(m[1])
    rmse.append(m[2])
    wmape.append(m[3])

print("\n===== Final Result =====")
print(f"MAE {np.mean(mae):.4f} | RMSE {np.mean(rmse):.4f} | "
      f"MAPE {np.mean(mape):.4f} | WMAPE {np.mean(wmape):.4f}")
