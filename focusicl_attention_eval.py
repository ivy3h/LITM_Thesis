# -*- coding: utf-8 -*-

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import random
from tqdm.auto import tqdm
import numpy as np
import math
import re
import warnings
import matplotlib.pyplot as plt
import os
import datetime
import argparse
import traceback # For detailed error printing

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# os.environ["HF_TOKEN"] = 'REPLACE_WITH_YOUR_TOKEN'

# --- Default Generation Params ---
DEFAULT_MAX_NEW_TOKENS_GSM8K = 300
DEFAULT_MAX_NEW_TOKENS_MMLU = 20

# --- Load GSM8K Dataset ---
def load_gsm8k_dataset():
    """Loads the GSM8K dataset."""
    cache_dir = "./cache_gsm8k"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    try:
        dataset = load_dataset("gsm8k", "main", cache_dir=cache_dir)
        print("Loaded GSM8K dataset from cache or disk.")
    except Exception as e:
        print(f"Could not load GSM8K from cache ({e}), downloading...")
        dataset = load_dataset("gsm8k", "main")

    print(f"Loaded GSM8K dataset: {len(dataset['train'])} train, {len(dataset['test'])} test examples.")

    if 'id' not in dataset['train'].column_names:
        print("Mapping 'train' split for GSM8K...")
        dataset = dataset.map(lambda example, idx: {'id': f"train_gsm8k_{idx}_{hash(example['question'])}"}, with_indices=True, load_from_cache_file=False)
    if 'id' not in dataset['test'].column_names:
        print("Mapping 'test' split for GSM8K...")
        dataset = dataset.map(lambda example, idx: {'id': f"test_gsm8k_{idx}_{hash(example['question'])}"}, with_indices=True, load_from_cache_file=False)

    return dataset

# --- Load MMLU Dataset ---
def load_mmlu_dataset(subset_name):
    """Loads the MMLU dataset for a specific subset."""
    if not subset_name or subset_name == "all":
         raise ValueError("MMLU requires a specific subset name (e.g., 'high_school_mathematics'). Loading 'all' is not supported by this script.")

    print(f"Loading MMLU dataset, subset: {subset_name}...")
    safe_subset_name = subset_name.replace('/', '_')
    cache_dir = f"./cache_mmlu_{safe_subset_name}"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    try:
        dataset = load_dataset("cais/mmlu", subset_name, cache_dir=cache_dir)
        print(f"Loaded MMLU subset '{subset_name}' from cache or disk.")

        required_splits = ['dev', 'test']
        missing_splits = [s for s in required_splits if s not in dataset]
        if missing_splits:
            raise ValueError(f"MMLU subset '{subset_name}' loaded, but missing required split(s): {', '.join(missing_splits)}")

        dev_size = len(dataset['dev'])
        test_size = len(dataset['test'])
        print(f"MMLU subset '{subset_name}' loaded: {dev_size} dev (demos), {test_size} test examples.")

        if 'id' not in dataset['dev'].column_names:
            print("Mapping 'dev' split for MMLU...")
            dataset['dev'] = dataset['dev'].map(
                lambda example, idx: {'id': f"dev_{safe_subset_name}_{idx}_{hash(example['question'])}"},
                with_indices=True, load_from_cache_file=False
            )
        if 'id' not in dataset['test'].column_names:
            print("Mapping 'test' split for MMLU...")
            dataset['test'] = dataset['test'].map(
                lambda example, idx: {'id': f"test_{safe_subset_name}_{idx}_{hash(example['question'])}"},
                with_indices=True, load_from_cache_file=False
            )
        return dataset

    except Exception as e:
        print(f"Could not load MMLU subset '{subset_name}'. Error: {e}")
        traceback.print_exc()
        exit(1)

# --- Formatting Functions ---

# GSM8K Formatting
def format_gsm8k_icl_input(tokenizer, demonstrations, query, include_answer_prompt=True):
    """Formats demonstrations and query into chat template for GSM8K."""
    messages = [{"role": "system", "content": "You are a helpful assistant capable of solving math word problems step-by-step."}]
    for demo in demonstrations:
        messages.append({"role": "user", "content": demo['question']})
        messages.append({"role": "assistant", "content": demo['answer']}) # GSM8K answer includes reasoning
    messages.append({"role": "user", "content": query})

    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=include_answer_prompt)
    except Exception as e:
        warnings.warn(f"GSM8K Format: Error applying chat template: {e}. Returning empty prompt.")
        prompt = ""
    return prompt

# MMLU Formatting
def format_mmlu_icl_input(tokenizer, demonstrations, query_text, choices, include_answer_prompt=True):
    """Formats demonstrations and query into chat template for MMLU."""
    messages = [{"role": "system", "content": "You are an expert answering multiple choice questions. Respond with the letter corresponding to the correct choice."}]
    letter_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

    for demo in demonstrations:
        demo_question = demo['question']
        demo_choices = demo['choices']
        demo_answer_idx = demo['answer']
        demo_answer_letter = letter_map.get(demo_answer_idx, "?")
        formatted_demo_choices = "\n".join([f"{letter_map.get(i)}. {choice}" for i, choice in enumerate(demo_choices)])
        demo_input_text = f"Question: {demo_question}\nChoices:\n{formatted_demo_choices}\nAnswer:"
        messages.append({"role": "user", "content": demo_input_text})
        messages.append({"role": "assistant", "content": demo_answer_letter})

    formatted_choices = "\n".join([f"{letter_map.get(i)}. {choice}" for i, choice in enumerate(choices)])
    query_input_text = f"Question: {query_text}\nChoices:\n{formatted_choices}\nAnswer:"
    messages.append({"role": "user", "content": query_input_text})

    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=include_answer_prompt)
    except Exception as e:
        warnings.warn(f"MMLU Format: Error applying chat template: {e}. Returning empty prompt.")
        prompt = ""
    return prompt

# --- Answer Extraction Functions ---

# GSM8K Answer Extraction
def extract_gsm8k_answer(generated_text):
    """Extracts the final numerical answer from GSM8K model output."""
    boxed_match = re.search(r"\\boxed\{([\d,.-]+)\}", generated_text)
    if boxed_match:
        answer_str = boxed_match.group(1)
    else:
        numbers = re.findall(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+\.\d+|\.\d+|[-+]?\d+", generated_text)
        if numbers:
            answer_str = numbers[-1]
        else:
            return None
    try:
        cleaned_answer = answer_str.replace(",", "").strip()
        if cleaned_answer.endswith('.'): cleaned_answer = cleaned_answer[:-1]
        if not cleaned_answer: return None
        return float(cleaned_answer) if '.' in cleaned_answer else int(cleaned_answer)
    except ValueError:
        return None

# MMLU Answer Extraction
def extract_mmlu_answer(generated_text):
    """Extracts the final letter answer (A, B, C, D...) from MMLU model output."""
    match = re.search(r"Answer:?\s*([A-E])", generated_text, re.IGNORECASE)
    if match: return match.group(1).upper()
    match = re.search(r"^\s*([A-E])", generated_text, re.IGNORECASE)
    if match: return match.group(1).upper()
    match = re.search(r"\(([A-E])\)", generated_text, re.IGNORECASE)
    if match: return match.group(1).upper()
    match = re.search(r"([A-E])", generated_text, re.IGNORECASE) # Fallback: first letter found
    if match: return match.group(1).upper()
    return None

# --- Core ICL Logic (Now dataset-aware) ---

def run_standard_icl(model, tokenizer, dataset_name, demonstrations, query, choices=None, max_new_tokens=None):
    """
    Runs standard In-Context Learning using the model's chat template.
    Dispatches formatting and extraction based on dataset_name.
    """
    if dataset_name == 'gsm8k':
        prompt = format_gsm8k_icl_input(tokenizer, demonstrations, query, include_answer_prompt=True)
        extractor_func = extract_gsm8k_answer
    elif dataset_name == 'mmlu':
        if choices is None: raise ValueError("MMLU requires 'choices' to be provided.")
        prompt = format_mmlu_icl_input(tokenizer, demonstrations, query, choices, include_answer_prompt=True)
        extractor_func = extract_mmlu_answer
    else:
        raise ValueError(f"Unsupported dataset_name for standard ICL: {dataset_name}")

    if not prompt:
        warnings.warn(f"Standard ICL ({dataset_name}): Empty prompt generated, skipping.")
        return None

    try:
        if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length:
             max_model_len = tokenizer.model_max_length
        elif hasattr(model.config, 'max_position_embeddings') and model.config.max_position_embeddings:
             max_model_len = model.config.max_position_embeddings
        else:
             warnings.warn("Cannot determine model_max_length from tokenizer or config. Using fallback 4096.")
             max_model_len = 4096

        if max_new_tokens is None: 
            raise ValueError("max_new_tokens was not set.")

        inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
        prompt_tokens = inputs["input_ids"].shape[1]

        if prompt_tokens >= max_model_len - max_new_tokens:
             warnings.warn(f"Standard ICL ({dataset_name}): Prompt tokens ({prompt_tokens}) + max_new_tokens ({max_new_tokens}) might exceed max length ({max_model_len}). Trying truncation.")
             inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_model_len - max_new_tokens, truncation_side='left').to(DEVICE)
             prompt_tokens = inputs["input_ids"].shape[1]
             if prompt_tokens >= max_model_len - max_new_tokens:
                  warnings.warn(f"Standard ICL ({dataset_name}): Prompt still too long ({prompt_tokens}) after truncation. Skipping.")
                  return None
        else:
            inputs = inputs.to(DEVICE)

    except Exception as e:
        warnings.warn(f"Standard ICL ({dataset_name}): Error tokenizing prompt: {e}")
        return None

    generated_text = ""
    try:
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                do_sample=False,
                temperature=0.0,
            )
        generated_ids = outputs[0, inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    except torch.cuda.OutOfMemoryError:
         warnings.warn(f"CUDA OOM during Standard ICL generation ({dataset_name}). Skipping.")
         return None
    except Exception as e:
         warnings.warn(f"Error during Standard ICL generation ({dataset_name}): {e}")
         return None

    return extractor_func(generated_text)


def run_focus_icl(model, tokenizer, dataset_name, demonstrations, query, p_threshold, choices=None, max_new_tokens=None):
    """
    Runs FOCUS ICL with Triviality Filtering using the model's chat template.
    Dispatches formatting and extraction based on dataset_name.
    """
    if not (0 < p_threshold < 1):
         warnings.warn(f"FOCUS ({dataset_name}): Invalid p_threshold {p_threshold}. Skipping.")
         return None

    # 1. Prepare Initial Input
    if dataset_name == 'gsm8k':
        prompt_full = format_gsm8k_icl_input(tokenizer, demonstrations, query, include_answer_prompt=True)
        extractor_func = extract_gsm8k_answer
        query_content_for_search = query # Used for finding location
    elif dataset_name == 'mmlu':
        if choices is None: raise ValueError("MMLU requires 'choices' to be provided.")
        prompt_full = format_mmlu_icl_input(tokenizer, demonstrations, query, choices, include_answer_prompt=True)
        extractor_func = extract_mmlu_answer
        letter_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
        formatted_choices = "\n".join([f"{letter_map.get(i)}. {choice}" for i, choice in enumerate(choices)])
        query_content_for_search = f"Question: {query}\nChoices:\n{formatted_choices}\nAnswer:"
    else:
        raise ValueError(f"Unsupported dataset_name for FOCUS ICL: {dataset_name}")

    if not prompt_full:
        warnings.warn(f"FOCUS ICL ({dataset_name}): Empty prompt generated, skipping.")
        return None

    if max_new_tokens is None:
        raise ValueError("max_new_tokens was not set.")

    try:
        if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length:
             max_model_len = tokenizer.model_max_length
        elif hasattr(model.config, 'max_position_embeddings') and model.config.max_position_embeddings:
             max_model_len = model.config.max_position_embeddings
        else:
             warnings.warn("Cannot determine model_max_length from tokenizer or config. Using fallback 4096.")
             max_model_len = 4096

        inputs_full = tokenizer(prompt_full, return_tensors="pt", truncation=False)
        input_ids_full = inputs_full["input_ids"]
        prompt_tokens = input_ids_full.shape[1]

        # Fallback check
        fallback_needed = False
        if prompt_tokens >= max_model_len:
             warnings.warn(f"FOCUS ICL ({dataset_name}): Prompt tokens ({prompt_tokens}) >= max length ({max_model_len}). Running standard ICL fallback.")
             fallback_needed = True
        elif prompt_tokens >= max_model_len - 10: # Need some room for attention calc stability?
             warnings.warn(f"FOCUS ICL ({dataset_name}): Prompt tokens ({prompt_tokens}) very close to max length ({max_model_len}). Running standard ICL fallback.")
             fallback_needed = True

        if fallback_needed:
            return run_standard_icl(model, tokenizer, dataset_name, demonstrations, query, choices=choices, max_new_tokens=max_new_tokens)
        else:
             inputs_full = {k: v.to(DEVICE) for k, v in inputs_full.items()}
             input_ids_full = inputs_full["input_ids"]

    except Exception as e:
        warnings.warn(f"FOCUS ICL ({dataset_name}): Error tokenizing prompt: {e}. Running standard ICL fallback.")
        return run_standard_icl(model, tokenizer, dataset_name, demonstrations, query, choices=choices, max_new_tokens=max_new_tokens)

    # 2. Identify Demonstration Token Indices (Approximate)
    all_demo_indices = []
    try:
        query_tokens = tokenizer.encode(query_content_for_search, add_special_tokens=False)
        input_ids_list = input_ids_full[0].tolist()

        query_start_index = -1
        search_window = min(len(input_ids_list), len(query_tokens) + 500) 
        found = False
        start_scan_idx = len(input_ids_list) - len(query_tokens)
        end_scan_idx = max(-1, len(input_ids_list) - search_window)

        for i in range(start_scan_idx, end_scan_idx, -1):
            if input_ids_list[i : i + len(query_tokens)] == query_tokens:
                query_start_index = i
                found = True
                break

        if found:
            potential_demo_end_idx = query_start_index
            all_demo_indices = list(range(potential_demo_end_idx))
            if not all_demo_indices:
                 warnings.warn(f"FOCUS ({dataset_name}): Demo indices empty. Running standard ICL.")
                 return run_standard_icl(model, tokenizer, dataset_name, demonstrations, query, choices=choices, max_new_tokens=max_new_tokens)
        else:
            warnings.warn(f"FOCUS ({dataset_name}): Could not find query tokens. Demo index estimation failed. Running standard ICL.")
            return run_standard_icl(model, tokenizer, dataset_name, demonstrations, query, choices=choices, max_new_tokens=max_new_tokens)

    except Exception as e:
         warnings.warn(f"FOCUS ({dataset_name}): Error during demo token identification: {e}. Running standard ICL.")
         return run_standard_icl(model, tokenizer, dataset_name, demonstrations, query, choices=choices, max_new_tokens=max_new_tokens)

    # 3. Run Forward Pass to Get Attentions
    last_token_position = input_ids_full.shape[1] - 1
    aggregated_attention = None
    attentions = None
    try:
        with torch.no_grad():
            outputs_attn = model(input_ids=input_ids_full, attention_mask=inputs_full["attention_mask"], output_attentions=True)
            attentions = outputs_attn.attentions
            del outputs_attn
            if attentions is None:
                 if not model.config.output_attentions:
                     warnings.warn(f"FOCUS ({dataset_name}): Model config output_attentions=False. Running standard ICL.")
                     return run_standard_icl(model, tokenizer, dataset_name, demonstrations, query, choices=choices, max_new_tokens=max_new_tokens)
                 else: raise ValueError("Attentions are None despite config.")
    except torch.cuda.OutOfMemoryError:
         warnings.warn(f"CUDA OOM during FOCUS attention ({dataset_name}). Running standard ICL.")
         del attentions
         return run_standard_icl(model, tokenizer, dataset_name, demonstrations, query, choices=choices, max_new_tokens=max_new_tokens)
    except Exception as e:
         warnings.warn(f"Could not get attention weights ({dataset_name}, {e}). Running standard ICL.")
         del attentions
         return run_standard_icl(model, tokenizer, dataset_name, demonstrations, query, choices=choices, max_new_tokens=max_new_tokens)

    # 4. Aggregate Attention Scores
    try:
        num_layers = len(attentions)
        seq_len = input_ids_full.shape[1]
        aggregated_attention = torch.zeros(seq_len, device='cpu')
        for layer in range(num_layers):
            attn_layer = attentions[layer].squeeze(0).to('cpu')
            attn_from_last_token = attn_layer[:, last_token_position, :]
            aggregated_attention += attn_from_last_token.mean(dim=0)
            del attn_layer, attn_from_last_token
        aggregated_attention /= num_layers
        aggregated_attention = aggregated_attention.to(DEVICE)
        del attentions

    except torch.cuda.OutOfMemoryError:
        warnings.warn(f"CUDA OOM during attention aggregation ({dataset_name}). Running standard ICL.")
        del attentions
        return run_standard_icl(model, tokenizer, dataset_name, demonstrations, query, choices=choices, max_new_tokens=max_new_tokens)
    except Exception as e:
        warnings.warn(f"Error during attention aggregation ({dataset_name}, {e}). Running standard ICL.")
        del attentions
        return run_standard_icl(model, tokenizer, dataset_name, demonstrations, query, choices=choices, max_new_tokens=max_new_tokens)

    # 5. Identify Trivial Demonstration Tokens
    valid_demo_indices = [idx for idx in all_demo_indices if 0 <= idx < aggregated_attention.shape[0]]
    if not valid_demo_indices:
         warnings.warn(f"FOCUS ({dataset_name}): No valid demo indices. Running standard ICL.")
         return run_standard_icl(model, tokenizer, dataset_name, demonstrations, query, choices=choices, max_new_tokens=max_new_tokens)

    trivial_token_indices = []
    try:
        valid_indices_tensor = torch.tensor(valid_demo_indices, device=DEVICE, dtype=torch.long)
        demo_attentions = aggregated_attention[valid_indices_tensor]

        if demo_attentions.numel() == 0:
             warnings.warn(f"FOCUS ({dataset_name}): No demo attentions extracted. Running standard ICL.")
             return run_standard_icl(model, tokenizer, dataset_name, demonstrations, query, choices=choices, max_new_tokens=max_new_tokens)
        if torch.isnan(demo_attentions).any() or torch.isinf(demo_attentions).any():
             warnings.warn(f"FOCUS ({dataset_name}): Invalid demo attentions (NaN/Inf). Running standard ICL.")
             return run_standard_icl(model, tokenizer, dataset_name, demonstrations, query, choices=choices, max_new_tokens=max_new_tokens)
        if (demo_attentions == 0).all():
             warnings.warn(f"FOCUS ({dataset_name}): All demo attentions zero. Running standard ICL.")
             return run_standard_icl(model, tokenizer, dataset_name, demonstrations, query, choices=choices, max_new_tokens=max_new_tokens)

        if demo_attentions.numel() > 0:
            threshold_value = torch.quantile(demo_attentions.float().cpu(), p_threshold).to(DEVICE)
            trivial_token_indices = [idx for i, idx in enumerate(valid_demo_indices) if demo_attentions[i] < threshold_value]
        else: # Should be caught above, but belt and braces
             warnings.warn(f"FOCUS ({dataset_name}): Demo attentions became empty. Running standard ICL.")
             return run_standard_icl(model, tokenizer, dataset_name, demonstrations, query, choices=choices, max_new_tokens=max_new_tokens)

    except IndexError as e:
         warnings.warn(f"FOCUS ({dataset_name}): IndexError accessing demo attentions. Running standard ICL. Error: {e}")
         return run_standard_icl(model, tokenizer, dataset_name, demonstrations, query, choices=choices, max_new_tokens=max_new_tokens)
    except RuntimeError as e:
        warnings.warn(f"FOCUS ({dataset_name}): RuntimeError during quantile calc: {e}. Running standard ICL.")
        return run_standard_icl(model, tokenizer, dataset_name, demonstrations, query, choices=choices, max_new_tokens=max_new_tokens)
    except Exception as e:
        warnings.warn(f"FOCUS ({dataset_name}): Generic error during trivial token ID: {e}. Running standard ICL.")
        return run_standard_icl(model, tokenizer, dataset_name, demonstrations, query, choices=choices, max_new_tokens=max_new_tokens)

    # 6. Create Masked Input
    input_ids_masked = input_ids_full
    attention_mask_masked = inputs_full["attention_mask"].clone()
    if trivial_token_indices:
        trivial_indices_tensor = torch.tensor(trivial_token_indices, device=DEVICE, dtype=torch.long)
        valid_mask_indices = trivial_indices_tensor[trivial_indices_tensor < attention_mask_masked.shape[1]]
        if len(valid_mask_indices) > 0:
             attention_mask_masked[0, valid_mask_indices] = 0

    # 7. Generate Response with Masked Attention
    if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length: max_model_len = tokenizer.model_max_length
    elif hasattr(model.config, 'max_position_embeddings') and model.config.max_position_embeddings: max_model_len = model.config.max_position_embeddings
    else: max_model_len = 4096
    max_len_for_gen = max_model_len - max_new_tokens

    if input_ids_masked.shape[1] > max_len_for_gen:
        warnings.warn(f"FOCUS ICL ({dataset_name}): Input for masked gen ({input_ids_masked.shape[1]}) too long. Truncating from left.")
        input_ids_masked = input_ids_masked[:, -max_len_for_gen:]
        attention_mask_masked = attention_mask_masked[:, -max_len_for_gen:]
    gen_start_index = input_ids_masked.shape[1]

    generated_text_masked = ""
    try:
        with torch.no_grad():
            outputs_masked = model.generate(
                input_ids=input_ids_masked,
                attention_mask=attention_mask_masked,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                do_sample=False,
                temperature=0.0,
            )
        generated_ids_masked = outputs_masked[0, gen_start_index:]
        generated_text_masked = tokenizer.decode(generated_ids_masked, skip_special_tokens=True)

    except torch.cuda.OutOfMemoryError:
         warnings.warn(f"CUDA OOM during FOCUS masked generation ({dataset_name}, p={p_threshold}). Skipping.")
         return None
    except Exception as e:
         warnings.warn(f"Error during FOCUS masked generation ({dataset_name}, p={p_threshold}): {e}. Skipping.")
         return None

    # Clean up memory
    del input_ids_masked, attention_mask_masked, outputs_masked, generated_ids_masked
    if 'aggregated_attention' in locals() and aggregated_attention is not None: del aggregated_attention
    if 'demo_attentions' in locals() and demo_attentions is not None: del demo_attentions
    torch.cuda.empty_cache()

    return extractor_func(generated_text_masked)


# --- Evaluation Loop (GSM8K) ---
def evaluate_gsm8k(model, tokenizer, dataset, method_func, n_demos, p_threshold=None, num_test_samples=None, max_new_tokens=None):
    """Evaluates a given ICL method on the GSM8K test set."""
    correct = 0
    total = 0
    errors = 0
    demonstration_pool = list(dataset['train'])
    test_set = dataset['test']
    dataset_name = "gsm8k" # Explicitly set for clarity within function

    if max_new_tokens is None: raise ValueError("evaluate_gsm8k requires max_new_tokens")

    if num_test_samples is None: num_test_samples = len(test_set)
    else: num_test_samples = min(num_test_samples, len(test_set))
    test_samples = test_set.select(range(num_test_samples))

    actual_n_demos_pool = min(n_demos, len(demonstration_pool)) if demonstration_pool else 0
    if len(demonstration_pool) < n_demos:
         warnings.warn(f"GSM8K: Demo pool ({len(demonstration_pool)}) < required ({n_demos}). Using {actual_n_demos_pool}.")

    desc = f"Eval GSM8K {method_func.__name__} (k={n_demos}"
    if method_func == run_focus_icl and p_threshold is not None: desc += f", p={p_threshold}"
    desc += f", N={num_test_samples})"
    pbar = tqdm(test_samples, desc=desc)

    for example in pbar:
        query = example['question']
        true_answer_text = example['answer']
        true_answer_val = extract_gsm8k_answer(true_answer_text)
        example_id = example.get('id', 'N/A')

        if true_answer_val is None:
            errors += 1
            continue

        potential_demos = [d for d in demonstration_pool if d.get('id') != example_id]
        current_n_demos = min(actual_n_demos_pool, len(potential_demos))
        if current_n_demos == 0 and actual_n_demos_pool > 0:
             errors += 1
             continue
        demos = random.sample(potential_demos, current_n_demos) if current_n_demos > 0 else []

        predicted_answer_val = None
        try:
            if method_func == run_focus_icl:
                 if p_threshold is None: raise ValueError("p_threshold needed for FOCUS")
                 predicted_answer_val = method_func(model, tokenizer, dataset_name, demos, query, p_threshold, max_new_tokens=max_new_tokens)
            else:
                 predicted_answer_val = method_func(model, tokenizer, dataset_name, demos, query, max_new_tokens=max_new_tokens)

            if predicted_answer_val is None: errors += 1

        except Exception as e:
             print(f"\n!!! GSM8K Eval Error ({method_func.__name__}, {example_id}): {e}")
             traceback.print_exc()
             errors += 1
             predicted_answer_val = None

        if predicted_answer_val is not None:
            is_correct = False
            try:
                if math.isclose(float(predicted_answer_val), float(true_answer_val), rel_tol=1e-4):
                    is_correct = True
            except (ValueError, TypeError):
                 if str(predicted_answer_val).strip() == str(true_answer_val).strip():
                     is_correct = True
            if is_correct: correct += 1

        total += 1
        accuracy = correct / total if total > 0 else 0
        pbar.set_postfix({"Acc": f"{accuracy:.3f}", "Correct": correct, "Attempted": total, "Errors": errors})

    accuracy = correct / total if total > 0 else 0
    print(f"\n--- GSM8K Eval Complete ({method_func.__name__}, k={n_demos}{f', p={p_threshold}' if p_threshold is not None else ''}) ---")
    print(f"Final Accuracy: {accuracy:.4f} ({correct}/{total}) | Errors: {errors}")
    print("-" * 30)
    return accuracy

# --- Evaluation Loop (MMLU) ---
def evaluate_mmlu(model, tokenizer, dataset, mmlu_subset_name, method_func, n_demos, p_threshold=None, num_test_samples=None, max_new_tokens=None):
    """Evaluates a given ICL method on the MMLU test set for a specific subset."""
    correct = 0
    total = 0
    errors = 0
    demonstration_pool = list(dataset['dev'])
    test_set = dataset['test']
    dataset_name = "mmlu" # Explicitly set

    if max_new_tokens is None: raise ValueError("evaluate_mmlu requires max_new_tokens")

    if num_test_samples is None: num_test_samples = len(test_set)
    else: num_test_samples = min(num_test_samples, len(test_set))
    test_samples = test_set.select(range(num_test_samples))

    actual_n_demos_pool = min(n_demos, len(demonstration_pool)) if demonstration_pool else 0
    if len(demonstration_pool) < n_demos:
        warnings.warn(f"MMLU ({mmlu_subset_name}): Demo pool ({len(demonstration_pool)}) < required ({n_demos}). Using {actual_n_demos_pool}.")

    desc = f"Eval MMLU/{mmlu_subset_name} {method_func.__name__} (k={n_demos}"
    if method_func == run_focus_icl and p_threshold is not None: desc += f", p={p_threshold}"
    desc += f", N={num_test_samples})"
    letter_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
    pbar = tqdm(test_samples, desc=desc)

    for example in pbar:
        query = example['question']
        choices = example['choices']
        true_answer_idx = example['answer']
        true_answer_letter = letter_map.get(true_answer_idx)
        example_id = example.get('id', 'N/A')

        if true_answer_letter is None:
            errors += 1
            continue

        potential_demos = [d for d in demonstration_pool if d.get('id') != example_id]
        current_n_demos = min(actual_n_demos_pool, len(potential_demos))
        if current_n_demos == 0 and actual_n_demos_pool > 0:
             errors += 1
             continue
        demos = random.sample(potential_demos, current_n_demos) if current_n_demos > 0 else []

        predicted_answer_letter = None
        try:
            if method_func == run_focus_icl:
                 if p_threshold is None: raise ValueError("p_threshold needed for FOCUS")
                 predicted_answer_letter = method_func(model, tokenizer, dataset_name, demos, query, p_threshold, choices=choices, max_new_tokens=max_new_tokens)
            else:
                 predicted_answer_letter = method_func(model, tokenizer, dataset_name, demos, query, choices=choices, max_new_tokens=max_new_tokens)

            if predicted_answer_letter is None: errors += 1

        except Exception as e:
             print(f"\n!!! MMLU Eval Error ({method_func.__name__}, {example_id}): {e}")
             traceback.print_exc()
             errors += 1
             predicted_answer_letter = None

        if predicted_answer_letter is not None:
            if predicted_answer_letter == true_answer_letter:
                correct += 1

        total += 1
        accuracy = correct / total if total > 0 else 0
        pbar.set_postfix({"Acc": f"{accuracy:.3f}", "Correct": correct, "Attempted": total, "Errors": errors})

    accuracy = correct / total if total > 0 else 0
    print(f"\n--- MMLU ({mmlu_subset_name}) Eval Complete ({method_func.__name__}, k={n_demos}{f', p={p_threshold}' if p_threshold is not None else ''}) ---")
    print(f"Final Accuracy: {accuracy:.4f} ({correct}/{total}) | Errors: {errors}")
    print("-" * 30)
    return accuracy

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Standard ICL and FOCUS ICL on GSM8K or MMLU dataset.")
    parser.add_argument(
        "--model_name", type=str, required=True,
        help="Required. HF model name/path (e.g., 'Qwen/Qwen2.5-1.5B-Instruct')."
    )
    parser.add_argument(
        "--dataset", type=str, required=True, choices=['gsm8k', 'mmlu'],
        help="Required. Dataset to evaluate on."
    )
    parser.add_argument(
        "--mmlu_subset", type=str, default=None,
        help="Required for MMLU. The specific MMLU subset (e.g., 'high_school_mathematics')."
    )
    parser.add_argument(
        "--num_eval_samples", type=int, default=250,
        help="Number of test samples per evaluation run. Default: 250."
    )
    parser.add_argument(
        "--k_values", nargs='+', type=int, default=[20, 40, 60, 80],
        help="List of k values (demos) to test. Example: --k_values 0 5 20"
    )
    parser.add_argument(
        "--p_values", nargs='+', type=float, default=[0.1, 0.2, 0.3],
        help="List of p thresholds for FOCUS ICL. Empty list skips FOCUS. Example: --p_values 0.1 0.25"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed. Default: 42."
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=None, # Default is None, set dynamically later
        help=f"Max new tokens for generation. If unset, defaults to {DEFAULT_MAX_NEW_TOKENS_GSM8K} for GSM8K and {DEFAULT_MAX_NEW_TOKENS_MMLU} for MMLU."
    )
    parser.add_argument(
        "--force_cpu", action="store_true",
        help="Force CPU evaluation."
    )

    args = parser.parse_args()

    # --- Validate Arguments ---
    if args.dataset == 'mmlu' and not args.mmlu_subset:
        parser.error("--mmlu_subset is required when --dataset=mmlu")
    if args.dataset == 'gsm8k' and args.mmlu_subset:
        warnings.warn("--mmlu_subset is ignored when --dataset=gsm8k")
        args.mmlu_subset = None # Clear it

    # --- Set Dynamic Defaults ---
    MAX_NEW_TOKENS = args.max_new_tokens
    if MAX_NEW_TOKENS is None:
        if args.dataset == 'gsm8k':
            MAX_NEW_TOKENS = DEFAULT_MAX_NEW_TOKENS_GSM8K
        elif args.dataset == 'mmlu':
            MAX_NEW_TOKENS = DEFAULT_MAX_NEW_TOKENS_MMLU
        print(f"Using default max_new_tokens for {args.dataset}: {MAX_NEW_TOKENS}")
    else:
        print(f"Using user-specified max_new_tokens: {MAX_NEW_TOKENS}")


    MODEL_NAME = args.model_name
    DATASET_NAME = args.dataset
    MMLU_SUBSET = args.mmlu_subset
    NUM_EVAL_SAMPLES = args.num_eval_samples
    N_DEMOS_LIST = sorted(list(set(args.k_values)))
    P_THRESHOLDS = sorted(list(set(args.p_values)))
    SEED = args.seed
    if args.force_cpu: DEVICE = "cpu"

    # --- Set Seed ---
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available() and DEVICE == "cuda":
        torch.cuda.manual_seed_all(SEED)

    # --- Print Configuration ---
    print(f"\n--- Configuration ---")
    print(f"Model Name: {MODEL_NAME}")
    print(f"Dataset: {DATASET_NAME}" + (f" / Subset: {MMLU_SUBSET}" if MMLU_SUBSET else ""))
    print(f"Device: {DEVICE}")
    print(f"Evaluation Samples: {NUM_EVAL_SAMPLES}")
    print(f"K values (demos): {N_DEMOS_LIST}")
    print(f"P values (FOCUS): {P_THRESHOLDS if P_THRESHOLDS else 'Not Run'}")
    print(f"Max New Tokens: {MAX_NEW_TOKENS}")
    print(f"Seed: {SEED}")
    print(f"--------------------\n")

    # --- Setup Directories ---
    RESULTS_DIR = "results"
    FIGURES_DIR = "figures"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # --- Load Model & Tokenizer ---
    print(f"Loading model: {MODEL_NAME}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype="auto",
            device_map="auto" if DEVICE == "cuda" else None,
            trust_remote_code=True,
            output_attentions=True if P_THRESHOLDS else False, # Only load attentions if needed
            attn_implementation="flash_attention_2" if DEVICE == "cuda" and torch.cuda.is_available() and hasattr(torch.nn.functional, 'scaled_dot_product_attention') else None
        )
        if DEVICE == "cpu" and ("auto" not in str(model.device) if hasattr(model, 'device') else True):
             model.to(DEVICE) # Ensure CPU placement if forced and device_map didn't handle it
        model.eval()
    except Exception as e:
        print(f"FATAL ERROR: Could not load model or tokenizer '{MODEL_NAME}'.")
        print(f"Error details: {e}")
        traceback.print_exc()
        exit(1)

    # Handle PAD token ID
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            print("Setting pad_token_id to eos_token_id")
            tokenizer.pad_token_id = tokenizer.eos_token_id
            if model.config.pad_token_id is None: model.config.pad_token_id = tokenizer.eos_token_id
        else:
            warnings.warn("Tokenizer has no pad_token_id or eos_token_id. Generation might fail.")
    elif model.config.pad_token_id is None:
         model.config.pad_token_id = tokenizer.pad_token_id

    # Ensure model_max_length is set
    if not hasattr(tokenizer, 'model_max_length') or not tokenizer.model_max_length:
        if hasattr(model.config, 'max_position_embeddings') and model.config.max_position_embeddings:
            tokenizer.model_max_length = model.config.max_position_embeddings
            print(f"Set tokenizer.model_max_length from model config: {tokenizer.model_max_length}")
        else:
            tokenizer.model_max_length = 4096 # Fallback
            print(f"Warning: Tokenizer model_max_length not found, set to default: {tokenizer.model_max_length}")
    else:
        print(f"Tokenizer model_max_length found: {tokenizer.model_max_length}")


    # --- Load Dataset ---
    print(f"\nLoading dataset: {DATASET_NAME}" + (f" ({MMLU_SUBSET})" if MMLU_SUBSET else ""))
    if DATASET_NAME == 'gsm8k':
        eval_dataset = load_gsm8k_dataset()
        evaluate_func = evaluate_gsm8k
    elif DATASET_NAME == 'mmlu':
        eval_dataset = load_mmlu_dataset(subset_name=MMLU_SUBSET)
        evaluate_func = evaluate_mmlu
    else:
        raise ValueError(f"Invalid dataset name: {DATASET_NAME}")

    # --- Run Experiments ---
    results = {k: {} for k in N_DEMOS_LIST}

    for n_demos in N_DEMOS_LIST:
        print(f"\n===== Running Experiments for k={n_demos} on {DATASET_NAME}" + (f"/{MMLU_SUBSET}" if MMLU_SUBSET else "") + " =====")

        if DEVICE == "cuda": torch.cuda.empty_cache(); print("Cleared CUDA Cache")

        # --- Evaluate Standard ICL ---
        print(f"\n--- Evaluating Standard ICL (k={n_demos}) ---")
        try:
            eval_args = {
                "model": model, "tokenizer": tokenizer, "dataset": eval_dataset,
                "method_func": run_standard_icl, "n_demos": n_demos,
                "num_test_samples": NUM_EVAL_SAMPLES, "max_new_tokens": MAX_NEW_TOKENS
            }
            if DATASET_NAME == 'mmlu': eval_args["mmlu_subset_name"] = MMLU_SUBSET

            icl_accuracy = evaluate_func(**eval_args)
            results[n_demos]['ICL'] = icl_accuracy if isinstance(icl_accuracy, (float, int)) else np.nan
        except Exception as e:
            print(f"!! FATAL ERROR during Standard ICL eval (k={n_demos}): {e}")
            results[n_demos]['ICL'] = np.nan
            traceback.print_exc()

        if DEVICE == "cuda": torch.cuda.empty_cache(); print("Cleared CUDA Cache before FOCUS runs")

        # --- Evaluate FOCUS ICL ---
        if P_THRESHOLDS:
            if not model.config.output_attentions:
                print(f"\n--- Skipping FOCUS ICL for k={n_demos} (model config output_attentions=False) ---")
                for p_val in P_THRESHOLDS: results[n_demos][f"FOCUSICL_p{p_val}"] = np.nan
            else:
                for p_val in P_THRESHOLDS:
                    focus_key = f"FOCUSICL_p{p_val}"
                    print(f"\n--- Evaluating FOCUS ICL (k={n_demos}, p={p_val}) ---")
                    try:
                        eval_args = {
                           "model": model, "tokenizer": tokenizer, "dataset": eval_dataset,
                           "method_func": run_focus_icl, "n_demos": n_demos, "p_threshold": p_val,
                           "num_test_samples": NUM_EVAL_SAMPLES, "max_new_tokens": MAX_NEW_TOKENS
                        }
                        if DATASET_NAME == 'mmlu': eval_args["mmlu_subset_name"] = MMLU_SUBSET

                        focus_icl_accuracy = evaluate_func(**eval_args)
                        results[n_demos][focus_key] = focus_icl_accuracy if isinstance(focus_icl_accuracy, (float, int)) else np.nan
                    except Exception as e:
                        print(f"!! FATAL ERROR during FOCUS ICL eval (k={n_demos}, p={p_val}): {e}")
                        results[n_demos][focus_key] = np.nan
                        traceback.print_exc()
        else:
             print(f"\n--- Skipping FOCUS ICL for k={n_demos} (No p_thresholds specified) ---")

        print(f"===== Finished Experiments for k={n_demos} =====")


    # --- Print Summary ---
    dataset_print_name = f"{DATASET_NAME}" + (f" - {MMLU_SUBSET}" if MMLU_SUBSET else "")
    print(f"\n\n--- Overall Evaluation Summary ({dataset_print_name}) ---")
    print(f"Model: {MODEL_NAME}")
    print(f"Test Samples per run: {NUM_EVAL_SAMPLES}")
    if P_THRESHOLDS: print(f"FOCUS p_thresholds: {P_THRESHOLDS}")
    else: print("FOCUS ICL not run.")

    table_width = 30 + len(P_THRESHOLDS) * 19 if P_THRESHOLDS else 30
    print("-" * table_width)
    header = "| k (n_demos) | Standard ICL Acc |"
    for p_val in P_THRESHOLDS: header += f" FOCUS ICL (p={p_val}) Acc |"
    print(header)
    separator = "|-------------|------------------|"
    for _ in P_THRESHOLDS: separator += "--------------------|"
    print(separator)

    summary_lines = [ 
        f"--- Overall Evaluation Summary ({dataset_print_name}) ---",
        f"Model: {MODEL_NAME}",
        f"Device: {DEVICE}",
        f"Test Samples per run: {NUM_EVAL_SAMPLES}",
        f"K values tested: {N_DEMOS_LIST}",
        f"FOCUS p_thresholds: {P_THRESHOLDS if P_THRESHOLDS else 'Not Run'}",
        f"Max New Tokens: {MAX_NEW_TOKENS}",
        f"Seed: {SEED}",
        f"Evaluation Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n",
        "-" * table_width, header, separator
    ]
    sorted_k = sorted(results.keys())
    for n_demos in sorted_k:
        scores = results[n_demos]
        icl_score_str = f"{scores.get('ICL', np.nan):.4f}" if not np.isnan(scores.get('ICL', np.nan)) else "Error/NaN"
        row = f"| {n_demos:<11} | {icl_score_str:<16} |"
        for p_val in P_THRESHOLDS:
            focus_key = f"FOCUSICL_p{p_val}"
            focus_score = scores.get(focus_key, np.nan)
            focus_score_str = f"{focus_score:.4f}" if not np.isnan(focus_score) else "Error/NaN"
            row += f" {focus_score_str:<18} |"
        print(row)
        summary_lines.append(row)
    print("-" * table_width)
    summary_lines.append("-" * table_width)
    summary_lines.append("\n--- End of Summary ---")

    # --- Save Summary to Text File ---
    print("\n--- Saving Summary ---")
    try:
        safe_model_name = MODEL_NAME.replace('/', '_').replace(':', '-')
        dataset_file_part = f"{DATASET_NAME}" + (f"_{MMLU_SUBSET.replace('/','_')}" if MMLU_SUBSET else "")
        p_values_str = "-".join(map(str, P_THRESHOLDS)) if P_THRESHOLDS else "none"
        k_range_str = f"k{min(N_DEMOS_LIST)}-{max(N_DEMOS_LIST)}" if len(N_DEMOS_LIST)>1 else f"k{N_DEMOS_LIST[0]}" if N_DEMOS_LIST else "k_none"

        results_filename_base = f"results_{dataset_file_part}_{safe_model_name}_{k_range_str}_p{p_values_str}_n{NUM_EVAL_SAMPLES}"
        results_filename = os.path.join(RESULTS_DIR, results_filename_base + ".txt")

        with open(results_filename, 'w', encoding='utf-8') as f:
            for line in summary_lines:
                f.write(line + "\n")
        print(f"Results summary saved to: {results_filename}")

    except Exception as e:
        print(f"Error saving results summary: {e}")
        traceback.print_exc()

    # --- Plotting ---
    print("\n--- Generating Plot ---")
    plot_generated = False
    has_data = any(not np.isnan(results[k].get('ICL', np.nan)) or \
                   any(not np.isnan(results[k].get(f"FOCUSICL_p{p}", np.nan)) for p in P_THRESHOLDS)
                   for k in N_DEMOS_LIST if k in results)

    if has_data:
        try:
            x_values = sorted(results.keys())
            plt.figure(figsize=(12, 7))

            # Plot Standard ICL
            icl_accuracies = [results[k].get('ICL', np.nan) for k in x_values]
            valid_icl = [(x, y) for x, y in zip(x_values, icl_accuracies) if not np.isnan(y)]
            if valid_icl: plt.plot([p[0] for p in valid_icl], [p[1] for p in valid_icl], marker='o', label='Standard ICL')

            # Plot FOCUS ICL
            if P_THRESHOLDS:
                markers = ['s', '^', 'D', 'v', '<', '>']
                linestyles = ['--', ':', '-.', '--', ':', '-.']
                for i, p_val in enumerate(P_THRESHOLDS):
                    focus_key = f"FOCUSICL_p{p_val}"
                    focus_acc = [results[k].get(focus_key, np.nan) for k in x_values]
                    valid_focus = [(x, y) for x, y in zip(x_values, focus_acc) if not np.isnan(y)]
                    if valid_focus: plt.plot([p[0] for p in valid_focus], [p[1] for p in valid_focus], marker=markers[i % len(markers)], linestyle=linestyles[i % len(linestyles)], label=f'FOCUS ICL (p={p_val})')

            plt.xlabel("Number of Demonstrations (k)")
            plt.ylabel(f"Accuracy on {dataset_print_name}")
            plt.title(f"Performance vs. Demonstrations ({dataset_print_name})\nModel: {MODEL_NAME}, Samples/Run: {NUM_EVAL_SAMPLES}")
            plt.xticks(N_DEMOS_LIST if N_DEMOS_LIST else [0])
            plt.ylim(bottom=0)
            all_acc = [acc for k in x_values for acc in results[k].values() if isinstance(acc, (float, int)) and not np.isnan(acc)]
            max_y = max(all_acc) if all_acc else 1.0
            plt.ylim(top=min(max_y * 1.1, 1.05))
            plt.legend()
            plt.grid(True, linestyle='--', linewidth=0.5)
            plt.tight_layout()

            # Save Plot
            plot_filename_base = f"performance_{dataset_file_part}_{safe_model_name}_{k_range_str}_p{p_values_str}_n{NUM_EVAL_SAMPLES}"
            pdf_filename = os.path.join(FIGURES_DIR, plot_filename_base + ".pdf")
            jpg_filename = os.path.join(FIGURES_DIR, plot_filename_base + ".jpg")
            plt.savefig(pdf_filename); plt.savefig(jpg_filename)
            print(f"Plot saved as '{pdf_filename}' and '{jpg_filename}'")
            plot_generated = True
            plt.close()

        except Exception as e:
            print(f"Error during plotting: {e}")
            traceback.print_exc()
    else:
         print("\n--- Skipping Plot Generation (No valid results to plot) ---")

    print("\n--- Script Finished ---")
