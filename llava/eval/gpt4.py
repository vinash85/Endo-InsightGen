import sys
import json
import argparse
from pprint import pprint
from copy import deepcopy
from collections import defaultdict
import openai

sys.path.append("llava")

import time
import os
from openai import AsyncOpenAI
from openai import OpenAI
import base64

client = OpenAI(
api_key=os.environ.get("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")
)

# 定义函数读取图片并编码为 Base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string

# 处理 JSONL 文件并生成回答
def generate_responses_from_jsonl(jsonl_file_path, image_folder_path, output_file_path):
    with open(jsonl_file_path, "r") as file, open(output_file_path, "w") as output_file:
        for line in file:
            # 逐行解析 JSONL 文件
            data = json.loads(line.strip())
            
            # 获取图片路径并转换为 Base64 编码
            image_path = f"{image_folder_path}/{data['image']}"
            base64_image = encode_image_to_base64(image_path)
            prompt = f"{data['text']}\n\n[Base64 Image Data]\n{base64_image}"
            
            # 构建消息内容，包含文本和 Base64 编码的图像
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            # 调用 GPT-4 API 获取回答
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )
            
            # 打印并处理 GPT-4 的回答
            answer = response.choices[0].message.content
            output_data = {
                "question_id": data['question_id'],
                "answer": answer
            }
            print(answer)
            # 写入到 JSONL 文件
            output_file.write(json.dumps(output_data) + "\n")


# 运行函数并生成回答
generate_responses_from_jsonl("/data/yue/LLaVA-Med/data/eval/question.jsonl", "/data/yue/LLaVA-Med/data/eval/all_evaluate_image", "/data/yue/LLaVA-Med/answer/gpt4_general_answer.jsonl")
