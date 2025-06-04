import os

# 目标目录
target_dir = "/home/wangxi/diffusion-pipe/dataset/20250526医美训练素材整理/20250523医美素材整理后"
insert_token = "0523yimei,"

for root, _, files in os.walk(target_dir):
    for file in files:
        if file.endswith(".txt"):
            file_path = os.path.join(root, file)
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if not lines:
                continue  # 跳过空文件

            line = lines[0]
            first_comma = line.find(",")
            if first_comma != -1:
                # 插入新词在第一个逗号之后
                new_line = line[:first_comma+1] + insert_token + line[first_comma+1:]
                lines[0] = new_line
                with open(file_path, "w", encoding="utf-8") as f:
                    f.writelines(lines)
                print(f"✔ Updated: {file_path}")
            else:
                print(f"⚠ 跳过（没有逗号）: {file_path}")
