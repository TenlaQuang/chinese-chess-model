import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
from model import XiangqiNet
from preprocess import fen_to_tensor

# --- CẤU HÌNH (ĐÃ CHỈNH CHO FINE-TUNING) ---
DATA_FILE = "data/xiangqi_dataset.csv"
SAVE_PATH = "weights/xiangqi_model.pth"
EPOCHS = 20           # Học thêm 20 vòng là đủ
BATCH_SIZE = 64
LR = 0.00001          # <--- QUAN TRỌNG: Tốc độ học cực nhỏ (để ngấm từ từ)

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
    
    # --- [THÊM MỚI] ĐOẠN LOAD MODEL CŨ ---
    if os.path.exists(SAVE_PATH):
        print(f"🔄 Phát hiện model cũ: {SAVE_PATH}")
        print("👉 Đang nạp để học tiếp (Fine-tuning)...")
        try:
            model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
            print("✅ Đã nạp thành công! AI sẽ không phải học lại từ đầu.")
        except Exception as e:
            print(f"⚠️ Lỗi nạp model cũ ({e}). Sẽ train mới từ đầu.")
    else:
        print("🆕 Không thấy model cũ. Sẽ train mới từ đầu.")
    # -------------------------------------

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    # Load dữ liệu (Lần này là file 100k dòng)
    try:
        print("⏳ Đang đọc file dữ liệu lớn...")
        dataset = ChessDataset(DATA_FILE)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        print(f"📊 Tổng cộng: {len(dataset)} mẫu dữ liệu.")
    except Exception as e:
        print(f"❌ Lỗi đọc file CSV: {e}")
        return

    # Khởi tạo mức lỗi kỷ lục (Để so sánh)
    # Mẹo: Nếu load model cũ, ta có thể set best_loss cao một chút để nó dễ lưu cái mới
    best_loss = 0.1 

    print(f"🏋️ Bắt đầu Fine-tuning trong {EPOCHS} Epochs...")
    
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
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.6f}", end="")

        # Logic lưu model xịn nhất
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  🔥 [LƯU] Model khôn hơn rồi!")
        else:
            print(f"     (Chưa tốt hơn)")

    print(f"\n✅ Hoàn tất nâng cấp! Model Level 2 đang ở: {SAVE_PATH}")

if __name__ == "__main__":
    train()