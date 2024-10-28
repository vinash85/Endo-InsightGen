import json

# 输入的 txt 文件路径
txt_file_path = "llavamed_answer_ge.txt"
# 输出的 JSONL 文件路径
jsonl_file_path = "llavamed_answer_general.jsonl"

# 打开 JSONL 文件进行写入
with open(jsonl_file_path, "w") as jsonl_file:
    with open(txt_file_path, "r") as txt_file:
        question_text = ""
        response = ""
        i = 0
        for line in txt_file:
            line = line.strip()
            # 识别问题行
            if line.startswith("Q: "):
                question_text = line[3:].strip("?")  # 去掉问号和多余的空格
            # 识别回答行
            elif line.startswith("A: b'") or line.startswith('A: b"'):
                # 去掉 A: b' 或 A: b" 的前缀，以及末尾的 ' 或 " 引号
                response = line[5:].strip("'\"")
                
                # 将问题和回答写入 JSONL
                output_data = {
                    "question_id": i,
                    "question_text": question_text,
                    "response": response
                }
                i = i + 1
                jsonl_file.write(json.dumps(output_data) + "\n")
