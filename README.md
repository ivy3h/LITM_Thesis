# Research and Optimization of _Lost-in-the-Middle_ in Long-Context Processing of Large Language Models 🧚‍♀️

This repository contains the materials and code for my graduation thesis, titled *"Research and Optimization of _Lost-in-the-Middle_ in Long-Context Processing of Large Language Models"*.

[[Dataset]](https://huggingface.co/datasets/JiayiHe/IN2_Training) • [[FILM-Qwen-7B]](https://www.modelscope.cn/models/Jiayihe/FILM-Qwen-7B) • [[FILM-Mistral-v0.2-7B]](https://www.modelscope.cn/models/Jiayihe/FILM-Mistral-v0.2-7B) • [[FILM-Mistral-v0.3-7B]](https://www.modelscope.cn/models/Jiayihe/FILM-Mistral-v0.3-7B) 


## Setup 🚀
```bash
git clone https://github.com/ivy3h/LITM_Thesis.git
cd LITM_Thesis
pip install -r requirements.txt
```



## Model Training 🔥

### Data Construction
To construct the training data, run the following command:
```bash
python data_construction.py
```

### Fine-tuning
To fine-tune the model using the constructed data, set up [SWIFT](https://swift.readthedocs.io/en/latest/) and run the following commands:
```bash
# Setup
pip install 'ms-swift'

# Example for fine-tuning with Qwen2.5-7B-Instruct
CUDA_VISIBLE_DEVICES=0,1 \
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --dataset '/root/in2_training.jsonl' \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --attn_impl flash_attn \
    --gradient_accumulation_steps 8 \
    --max_length 32768 \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --model_author swift \
    --model_name swift-robot
```


## Explicit Attention Adjustment 🎯


### U-shaped Attention Visualization

To visualize the U-shaped attention distribution on the Natural Questions-based multi-document question answering task, run the following command:

```bash
python attention_visual.py
```

### FocusICL

To run the FocusICL evaluation on `GSM8K` and `MMLU` datasets, run the following commands:

```bash
# Example for GSM8K with Qwen2.5-7B-Instruct
python attention_eval.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --dataset gsm8k \
    --num_eval_samples 500 \
    --k_values 20 40 60 80 \
    --p_values 0.1 0.2 0.3 \
    --run_focus \
    --max_new_tokens 256

# Example for MMLU with Qwen2.5-7B-Instruct
python attention_eval.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --dataset mmlu \
    --mmlu_subset all \
    --num_eval_samples 500 \
    --k_values 20 40 60 80 \
    --p_values 0.1 0.2 0.3 \
    --run_focus \
    --max_new_tokens 20
```

### FITM

To run the FITM evaluation on `GSM8K` and `MMLU` datasets, run the following commands:

```bash
# Example for GSM8K with Qwen2.5-7B-Instruct
python attention_eval.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --dataset gsm8k \
    --num_eval_samples 500 \
    --k_values 20 40 60 80\
    --run_fitm \
    --fitm_temperatures 1e-4 5e-5 \
    --max_new_tokens 256

# Example for MMLU with Qwen2.5-7B-Instruct
python attention_eval.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --dataset mmlu \
    --mmlu_subset all \
    --num_eval_samples 500 \
    --k_values 20 40 60 80\
    --run_fitm \
    --fitm_temperatures 1e-4 5e-5 \
    --max_new_tokens 20 
```

## Evaluation 💫

To run the **VAL** evaluation, run the following command:
```bash
# Example for document retrieval with Qwen2.5-7B-Instruct
export VLLM_SKIP_SHARED_MEMORY_CHECK=1
python ./vllm_inference/vllm_inference.py \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --testdata_folder ./ValProbing-32K/ \
    --testdata_file document_bi_32k.jsonl \
    --output_folder ./VaLProbing/VaLProbing-32K/results/Qwen2.5-7B-Instruct/ \
    --tensor_parallel_size 2 \
    --max_length 128
```

To run the **long-context** evaluation, run the following command:
```bash
# Example for tasks with max_length = 32 using Qwen2.5-7B-Instruct
export VLLM_SKIP_SHARED_MEMORY_CHECK=1
python ./vllm_inference/vllm_inference.py \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --testdata_file LongBench_output_32.jsonl \
    --testdata_folder ./real_world_long/prompts/ \
    --output_folder ./real_world_long/results/Qwen2.5-7B-Instruct/ \
    --max_length 32 \
    --tensor_parallel_size 2
```

To run the **short-context** evaluation, run the following command:
```bash
# Example for CommonsenceQA with Qwen2.5-7B-Instruct
export VLLM_SKIP_SHARED_MEMORY_CHECK=1
python ./vllm_inference/vllm_inference.py \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --testdata_file csqa_0shot.jsonl \
    --testdata_folder ./short_tasks/prompts/ \
    --output_folder ./short_tasks/results/Qwen2.5-7B-Instruct/ \
    --max_length 128 \
    --tensor_parallel_size 2 \
    --trust_remote_code True
```

## Acknowledgement ✨
Our code is built on [FILM](https://github.com/microsoft/FILM) and [SWIFT](https://swift.readthedocs.io/en/latest/). We extend our gratitude to the authors for their work!
