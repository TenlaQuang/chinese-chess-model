import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
from model import XiangqiNet
from preprocess import fen_to_tensor

# CẤU HÌNH TRAIN
DATA_FILE = "data/xiangqi_dataset.csv"
SAVE_PATH = "weights/xiangqi_model.pth"
EPOCHS = 30
BATCH_SIZE = 64
LR = 0.001

class ChessDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return fen_to_tensor(row['FEN']), torch.tensor([float(row['Score'])], dtype=torch.float32)

def train():
    if not os.path.exists(DATA_FILE):
        print("❌ Chưa có dữ liệu. Hãy chạy generate_data.py trước!")
        return

    os.makedirs("weights", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Thiết bị training: {device}")

    model = XiangqiNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    dataset = ChessDataset(DATA_FILE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("🏋️ Bắt đầu huấn luyện...")
    for epoch in range(EPOCHS):
        total_loss = 0
        model.train()
        for boards, scores in loader:
            boards, scores = boards.to(device), scores.to(device)
            optimizer.zero_grad()
            outputs = model(boards)
            loss = criterion(outputs, scores)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.6f}")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"✅ Train xong! Model đã lưu tại: {SAVE_PATH}")

if __name__ == "__main__":
    train()