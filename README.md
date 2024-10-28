# Endo-InsightGen: A Multimodal AI Model for Pathological and Anatomical Descriptions of Endoscopic Images

## Contents

- [Install](#install)
- [Serving](#serving)
- [Evaluation](#evaluation)


## Install

1. Clone this repository and navigate to LLaVA-Med folder
```bash
https://github.com/vinash85/Endo-InsightGen
cd Endo-InsightGen
```

2. Install Package: Create conda environment

```Shell
conda create -n newenvironment python=3.10 -y
conda activate newenvironment
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```


## Training
```Shell
torchrun --nnodes=1 --nproc_per_node=2 --master_port=25001 \
    llava/train/train_mem.py \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --data_path /data/test.json \
    --image_folder /path/to/images \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --bf16 True \
    --output_dir /data/yue/LLaVA-Med/mytest \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb
```

## Evaluation

```Shell

python llava/eval/model_vqa.py \
    --model-name /path/to/model \
    --question-file /data/eval/question.jsonl  \
    --image-folder /dataset/all_images  \ 
    --answers-file answer/answer.jsonl

```

