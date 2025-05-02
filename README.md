# Research and Optimization of 'Lost-in-the-Middle' in Long-Context Processing of Large Language Models 🧚

This repository contains the materials and code for my graduation thesis, titled *"Research and Optimization of 'Lost-in-the-Middle' in Long-Context Processing of Large Language Models"*.

## Setup 🚀
```bash
git clone https://github.com/ivy3h/LITM_Thesis.git
cd LITM_Thesis
pip install -r requirements.txt
```



## Model Training 🔥





## Explicit Attention Adjustment 🎯


### U-shaped Attention Visualization

To visualize the U-shaped attention distribution on the Natural Questions-based multi-document question answering task, execute the following commands:

```bash
python attention_visual.py
```

### FocusICL

To run the FocusICL evaluation on GSM8K and MMLU datasets, execute the following commands:

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

To run the FITM evaluation on GSM8K and MMLU datasets, execute the following commands:

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


## Acknowledgement ✨
Our code is built on [FILM](https://github.com/microsoft/FILM) and [SWIFT](https://swift.readthedocs.io/en/latest/). We extend our gratitude to the authors for their work!
