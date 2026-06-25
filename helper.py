import csv
import glob
import os

# Lấy tất cả file csv trong thư mục hiện tại
csv_files = glob.glob("/home/user14/lamnh/ups/*.csv")

for csv_file in csv_files:
    rows = []

    # Đọc file
    with open(csv_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Đổi "\" -> "/"
            row["Path"] = row["Path"].replace("\\", "/")
            rows.append(row)

        fieldnames = reader.fieldnames

    # Ghi đè lại file
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(rows)

    print(f"Fixed: {csv_file}")