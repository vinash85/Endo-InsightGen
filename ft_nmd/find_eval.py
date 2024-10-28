import json
import random
import json
import random

# 定义映射字典
question_mapping = {
    "What pathology is visible in this image?": "Could you identify the pathology visible in this image as Barrett's esophagus?",
    "Can you identify the abnormality in this image?": "Could you specify the abnormality shown in this image, identified as Barrett's esophagus, where the normal squamous epithelium is replaced by columnar epithelium?",
    "Can you identify the Barrett’s esophagus segment in this image?": "Could you recognize the segment of Barrett’s esophagus in this image as less than 3 cm, indicating short-segment Barrett’s esophagus?",
    "How does short-segment Barrett's esophagus differ from typical Barrett’s esophagus?": "Could you explain how short-segment Barrett's esophagus, involving less than 3 cm, differs from typical Barrett’s esophagus?",
    "What pathology is seen in this image?": "Could you identify the pathology seen in this image as grade A esophagitis?",
    "Can you identify the type of esophagitis present in this image?": "Could you specify the type of esophagitis present, identified as grade A, characterized by small erosions in the esophageal lining?",
    "What type of esophagitis is visible in this image?": "Could you recognize the type of esophagitis visible in this image as grade B-D esophagitis?",
    "How does grade B-D esophagitis differ from milder forms of esophagitis?": "Could you describe how grade B-D esophagitis differs from milder forms, involving larger areas of mucosal damage?"
}


input_file = '/data/yue/LLaVA-Med/data2/pathological_up.json'
output_file = '/data/yue/LLaVA-Med/ft_nmd/path_up.jsonl'

# 用于存储所有符合条件的记录
all_data = []

# 读取整个 JSON 文件
with open(input_file, 'r', encoding='utf-8') as infile:
    data_list = json.load(infile)  # 直接读取整个文件为列表

# 遍历 JSON 列表中的每一项
for data in data_list:
    human_value = data["conversations"][0]["value"]
    
    # 如果 human_value 在映射字典中，进行替换
    if human_value in question_mapping:
        # 使用映射后的问题
        text = question_mapping[human_value]
    else:
        # 如果不在映射字典中，直接使用原始问题
        text = human_value
    
    # 构建符合要求的格式
    new_entry = {
        "image": data["image"],
        "pair_id": data["id"],
        "text": text,
        "gpt4_answer": data["conversations"][1]["value"],
        "type": "conversations"
    }
    all_data.append(new_entry)

# 随机选择 300 个记录
selected_data = random.sample(all_data, 200) if len(all_data) >= 200 else all_data

# 将结果写入新的 JSONL 文件
with open(output_file, 'w', encoding='utf-8') as outfile:
    for question_id, entry in enumerate(selected_data):
        entry["question_id"] = question_id  # 添加 question_id 字段
        json.dump(entry, outfile)
        outfile.write('\n')

print("已成功生成随机选择的300张图片并映射问题的JSONL文件")