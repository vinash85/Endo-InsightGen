import json
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from sentence_transformers import SentenceTransformer, util
nltk.download('punkt_tab')
nltk.download('wordnet')

# 加载 Sentence-BERT 模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 加载预测数据和真实答案
def load_data(predictions_path, ground_truth_path):
    with open(predictions_path, 'r') as pred_file:
        predictions = [json.loads(line) for line in pred_file]

    with open(ground_truth_path, 'r') as gt_file:
        ground_truths = [json.loads(line) for line in gt_file]

    return predictions, ground_truths

# 根据 question_id 将预测与真实答案匹配
def get_matched_data(predictions, ground_truths):
    ground_truth_dict = {gt['question_id']: gt['gpt4_answer'] for gt in ground_truths}
    matched_data = []

    for pred in predictions:
        question_id = pred['question_id']
        if question_id in ground_truth_dict:
            ground_truth_answer = ground_truth_dict[question_id]
            model_answer = pred['text']
            matched_data.append((model_answer, ground_truth_answer))
    
    return matched_data

# 计算 BLEU、METEOR、ROUGE、Cosine Similarity 和单独的 n-gram BLEU 分数
def calculate_scores(matched_data):
    chencherry = SmoothingFunction()
    rouge = Rouge()
    
    bleu_scores = []
    meteor_scores = []
    rouge_scores = []
    cosine_similarities = []
    ngram_bleu_scores = {'1-gram': [], '2-gram': [], '3-gram': [], '4-gram': []}

    for model_answer, ground_truth_answer in matched_data:
        # 分词
        reference = [nltk.word_tokenize(ground_truth_answer)]
        candidate = nltk.word_tokenize(model_answer)
        
        # 计算 BLEU 分数
        bleu = sentence_bleu(reference, candidate, smoothing_function=chencherry.method1)
        bleu_scores.append(bleu)
        
        # 计算 METEOR 分数（分词后再传入）
        meteor = meteor_score(reference, candidate)  
        meteor_scores.append(meteor)
        
        # 计算 ROUGE-L 分数
        rouge_score = rouge.get_scores(" ".join(candidate), " ".join(reference[0]), avg=True)
        rouge_scores.append(rouge_score["rouge-l"]["f"])

        # 计算 Cosine Similarity 分数
        embeddings = model.encode([model_answer, ground_truth_answer])
        cosine_similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        cosine_similarities.append(cosine_similarity)
        
        # 计算 1-gram 到 4-gram 的单独 BLEU 分数
        ngram_bleu_scores['1-gram'].append(sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
        ngram_bleu_scores['2-gram'].append(sentence_bleu(reference, candidate, weights=(0, 1, 0, 0)))
        ngram_bleu_scores['3-gram'].append(sentence_bleu(reference, candidate, weights=(0, 0, 1, 0)))
        ngram_bleu_scores['4-gram'].append(sentence_bleu(reference, candidate, weights=(0, 0, 0, 1)))
    
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_meteor = sum(meteor_scores) / len(meteor_scores)
    avg_rouge = sum(rouge_scores) / len(rouge_scores)
    avg_cosine_similarity = sum(cosine_similarities) / len(cosine_similarities)

    avg_ngram_bleu = {key: sum(scores) / len(scores) for key, scores in ngram_bleu_scores.items()}
    
    return avg_bleu, avg_meteor, avg_rouge, avg_cosine_similarity, avg_ngram_bleu


# 主函数
def main():
    predictions_path = '/data/yue/LLaVA-Med/ft_nmd/answer.jsonl'  # 预测结果文件路径
    ground_truth_path = '/data/yue/LLaVA-Med/ft_nmd/all.jsonl'  # 真实答案文件路径

    predictions, ground_truths = load_data(predictions_path, ground_truth_path)
    matched_data = get_matched_data(predictions, ground_truths)
    avg_bleu, avg_meteor, avg_rouge, avg_cosine_similarity, avg_ngram_bleu = calculate_scores(matched_data)

    print(f'Average BLEU score: {avg_bleu}')
    print(f'Average METEOR score: {avg_meteor}')
    print(f'Average ROUGE-L score: {avg_rouge}')
    print(f'Average Cosine Similarity score: {avg_cosine_similarity}')
    
    # 打印每个 n-gram 的平均 BLEU 分数
    for ngram, score in avg_ngram_bleu.items():
        print(f'Average {ngram} BLEU score: {score}')

if __name__ == "__main__":
    main()
