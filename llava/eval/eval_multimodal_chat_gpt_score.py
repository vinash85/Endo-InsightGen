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


client = OpenAI(
api_key=os.environ.get("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")
)

async def call_async(samples, message_generator, retries=3, delay=5):
    results = []
    for sample in samples:
        messages = message_generator(sample)
        for attempt in range(retries):
            try:
                # 使用 openai.ChatCompletion.create 直接发起请求
                response = client.chat.completions.create(
                    messages=messages,
                    model="gpt-4o"
                )
                results.append({
                    'question_id': sample.get('question_id', 'unknown'),
                    'content': response.choices[0].message.content
                })
                break  # 如果成功，退出重试循环
            except Exception as e:  # 捕获所有异常，避免版本兼容问题
                print(f"Error in call_async: {e}")
                time.sleep(delay * (2 ** attempt))  # 指数退避策略
                if attempt == retries - 1:
                    print("Exceeded maximum retries, moving to the next sample.")
    return results

class LLMEvalPromptGenerator:
   
    instruct_prompt = """We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above..."""
    role = 'Assistant'

    @staticmethod
    def conv_to_str(question, ans1, ans2):
        return (f'[Question]\n{question}\n\n'
                f'[{LLMEvalPromptGenerator.role} 1]\n{ans1}\n\n[End of {LLMEvalPromptGenerator.role} 1]\n\n'
                f'[{LLMEvalPromptGenerator.role} 2]\n{ans2}\n\n[End of {LLMEvalPromptGenerator.role} 2]\n\n'
                f'[System]\n{LLMEvalPromptGenerator.instruct_prompt}\n\n')

    @staticmethod
    def compare_messages_gen(sample):
        messages = [
            {"role": "system", "content": """'You are a helpful and precise assistant for checking the quality of the answer."""},
        ]
        messages.append({"role": "user", "content": LLMEvalPromptGenerator.conv_to_str(sample['question'], sample['ans1'], sample['ans2'])})
        return messages

class ChatEvaluation:
    # Calculate precision, recall, F1 overall and for each domain.

    @staticmethod
    def get_domain(x):
        for domain in ['chest_xray', 'mri', 'histology', 'gross', 'ct_scan']:
            in_domain = x['domain'][domain]
            if in_domain:
                return domain
    
    @staticmethod
    def get_avg(x):
        return sum([float(y) for y in x]) / len(x)

    @staticmethod
    def eval(samples):
        predictions = [(x['question_id'], x['content'].split('\n')[0].split(' ')) for x in samples]
        result_lengths = [len(result) for _, result in predictions]
        avg_result_length = ChatEvaluation.get_avg(result_lengths)
        data_size = len(predictions)

        # 输出统计信息
        print("Average result length:", avg_result_length)
        print("Total number of samples:", data_size)

import asyncio

async def main(args):
    # 加载数据
    answer_data = []
    with open(args.input_path) as f:
        for line in f:
            answer_data.append(json.loads(line))

    question_data = []
    with open(args.question_input_path) as f:
        for line in f:
            question_data.append(json.loads(line))

    # 合并问题和答案数据
    samples = []
    for question, answer in zip(question_data, answer_data):
        sample = deepcopy(question)
        question['question'] = sample['text'][:-8]
        question['ans1'] = sample.pop('gpt4_answer')
        question['ans2'] = answer['text']
        samples.append(question)
  
    samples_question_ids = set(x['question_id'] for x in samples)

    # 生成并收集异步调用结果
    results = []
    BATCH_SIZE = 3
    for i in range(30):
        result_question_ids = set(result['question_id'] for result in results)

        batch = []
        for sample in samples:
            if sample['question_id'] in result_question_ids:
                continue
            batch.append(sample)
            if len(batch) >= BATCH_SIZE:
                async_results = await call_async(batch, lambda x: LLMEvalPromptGenerator.compare_messages_gen(x))
                results.extend(async_results)
                print(f"Result Size: {len(results)}")
                batch = []
        async_results = await call_async(batch, lambda x: LLMEvalPromptGenerator.compare_messages_gen(x))
        results.extend(async_results)
        print(f"Result Size: {len(results)}")

    # 将结果写入输出路径
    with open(args.output_path, 'w') as f:
        for line in results:
            f.write(json.dumps(line) + '\n')

    # 执行评估
    ChatEvaluation().eval(results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--question_input_path', type=str, default='data/eval/llava_med_eval_qa50_qa.jsonl')
    parser.add_argument('--input_path', type=str, default='dbfs:/mnt/hanoverdev/scratch/clwon/llava/test/answers/test50/2023-05-10_med-pretrain-364m-v1-1epoch.jsonl')
    parser.add_argument('--output_path', type=str, default='data/eval/llava_med_eval_qa50_qa_ans.jsonl')
    args = parser.parse_args()
    
    # 使用 asyncio 运行主程序
    asyncio.run(main(args))