import json

# 文件列表
file_paths = ["ana_low.jsonl", "ana_up.jsonl", "path_low.jsonl", "path_up.jsonl"]
merged_data = []

# 读取每个文件并添加到merged_data中
for file_path in file_paths:
    with open(file_path, 'r') as file:
        for line in file:
            record = json.loads(line)
            merged_data.append(record)

# 更新question_id，使其从0开始并依次增加
for i, record in enumerate(merged_data):
    record["question_id"] = i

# 将合并后的数据写入新的JSONL文件
with open("all.jsonl", 'w') as merged_file:
    for record in merged_data:
        merged_file.write(json.dumps(record) + "\n")

print("文件合并成功，并且question_id已更新。")
