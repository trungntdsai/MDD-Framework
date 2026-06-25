import csv
import json

csv_file = "/home/user14/lamnh/ups/train_ups.csv"
output_file = "ups_vocab.json"

# Dùng set để lưu phoneme không trùng nhau
phonemes = set()

with open(csv_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        # Đọc cả Canonical và Transcript
        for col in ["Canonical", "Transcript"]:
            text = row[col].strip()

            # Tách phoneme theo khoảng trắng
            phones = text.split()

            # Thêm vào set
            phonemes.update(phones)

# Sắp xếp cho ổn định
phonemes = sorted(phonemes)

# Tạo vocab với token rỗng ở index 0
vocab = {"": 0}

for idx, phone in enumerate(phonemes, start=1):
    vocab[phone] = idx

# Ghi ra file json
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)

print(f"Saved vocab with {len(vocab)} tokens to {output_file}")