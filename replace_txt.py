import os

# 目标目录
target_dir = "/home/wangxi/diffusion-pipe/dataset/20250526医美训练素材整理/20250523医美素材整理后"
# 要替换的关键词
old_phrase = "yimeiqianpifu"
new_phrase = "detailed skin texture"

# 遍历目录下所有 txt 文件
for root, _, files in os.walk(target_dir):
    for file in files:
        if file.endswith(".txt"):
            txt_path = os.path.join(root, file)
            with open(txt_path, "r", encoding="utf-8") as f:
                content = f.read()
            if old_phrase in content:
                new_content = content.replace(old_phrase, new_phrase)
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                print(f"✔ Replaced in: {txt_path}")
