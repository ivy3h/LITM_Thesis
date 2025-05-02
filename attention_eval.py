# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F # Added for softmax
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
DUMMY_CONTENT = "This is placeholder text." # Default dummy content for FITM calibration

# --- Default Generation Params ---
DEFAULT_MAX_NEW_TOKENS_GSM8K = 300
DEFAULT_MAX_NEW_TOKENS_MMLU = 20

# --- Load GSM8K Dataset ---
# (Keep the load_gsm8k_dataset function as is)
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

    # Add unique IDs if missing
    if 'id' not in dataset['train'].column_names:
        print("Mapping 'train' split for GSM8K...")
        dataset['train'] = dataset['train'].map(lambda example, idx: {'id': f"train_gsm8k_{idx}_{hash(example['question'])}"}, with_indices=True, load_from_cache_file=False)
    if 'id' not in dataset['test'].column_names:
        print("Mapping 'test' split for GSM8K...")
        dataset['test'] = dataset['test'].map(lambda example, idx: {'id': f"test_gsm8k_{idx}_{hash(example['question'])}"}, with_indices=True, load_from_cache_file=False)

    return dataset

# --- Load MMLU Dataset ---
# (Keep the load_mmlu_dataset function as is)
def load_mmlu_dataset(subset_name):
    """Loads the MMLU dataset for a specific subset."""
    if not subset_name:
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

        # Add unique IDs if missing
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
    # Ensure demonstrations is a list of dicts
    if not isinstance(demonstrations, list) or not all(isinstance(d, dict) for d in demonstrations):
        raise TypeError("Demonstrations must be a list of dictionaries.")

    messages = [{"role": "system", "content": "You are a helpful assistant capable of solving math word problems step-by-step."}]
    for demo in demonstrations:
        # Check required keys
        if 'question' not in demo or 'answer' not in demo:
            warnings.warn(f"Skipping invalid demo (missing keys): {demo}")
            continue
        messages.append({"role": "user", "content": str(demo['question'])})
        messages.append({"role": "assistant", "content": str(demo['answer'])}) # GSM8K answer includes reasoning
    messages.append({"role": "user", "content": str(query)})

    try:
        # Handle potential tokenizer errors more gracefully
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=include_answer_prompt)
        if not isinstance(prompt, str):
             warnings.warn(f"GSM8K Format: apply_chat_template did not return a string. Result: {prompt}")
             return ""
    except Exception as e:
        warnings.warn(f"GSM8K Format: Error applying chat template: {e}. Returning empty prompt.")
        prompt = ""
    return prompt

# MMLU Formatting
def format_mmlu_icl_input(tokenizer, demonstrations, query_text, choices, include_answer_prompt=True):
    """Formats demonstrations and query into chat template for MMLU."""
    if not isinstance(demonstrations, list) or not all(isinstance(d, dict) for d in demonstrations):
        raise TypeError("Demonstrations must be a list of dictionaries.")

    messages = [{"role": "system", "content": "You are an expert answering multiple choice questions. Respond with the letter corresponding to the correct choice."}]
    letter_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

    for demo in demonstrations:
        if 'question' not in demo or 'choices' not in demo or 'answer' not in demo:
            warnings.warn(f"Skipping invalid MMLU demo (missing keys): {demo}")
            continue
        demo_question = demo['question']
        demo_choices = demo['choices']
        demo_answer_idx = demo['answer']
        demo_answer_letter = letter_map.get(demo_answer_idx, "?")
        formatted_demo_choices = "\n".join([f"{letter_map.get(i)}. {choice}" for i, choice in enumerate(demo_choices)])
        demo_input_text = f"Question: {demo_question}\nChoices:\n{formatted_demo_choices}\nAnswer:"
        messages.append({"role": "user", "content": str(demo_input_text)})
        messages.append({"role": "assistant", "content": str(demo_answer_letter)})

    formatted_choices = "\n".join([f"{letter_map.get(i)}. {choice}" for i, choice in enumerate(choices)])
    query_input_text = f"Question: {query_text}\nChoices:\n{formatted_choices}\nAnswer:"
    messages.append({"role": "user", "content": str(query_input_text)})

    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=include_answer_prompt)
        if not isinstance(prompt, str):
             warnings.warn(f"MMLU Format: apply_chat_template did not return a string. Result: {prompt}")
             return ""
    except Exception as e:
        warnings.warn(f"MMLU Format: Error applying chat template: {e}. Returning empty prompt.")
        prompt = ""
    return prompt

# --- Answer Extraction Functions ---
# (Keep extract_gsm8k_answer and extract_mmlu_answer as is)
def extract_gsm8k_answer(generated_text):
    """Extracts the final numerical answer from GSM8K model output."""
    # Prefer the first boxed answer
    boxed_match = re.search(r"\\boxed\{([\d,.-]+)\}", generated_text)
    if boxed_match:
        answer_str = boxed_match.group(1)
    else:
        # Fallback: Look for 'Final Answer: The final answer is $\boxed{<answer>}$'
        final_ans_match = re.search(r"(?:final|Final Answer:)\s*(?:the final answer is)?\s*\$?\s*(?:\\boxed\{)?\s*([\d,.-]+)\s*(?:\\boxed\})?\$?\s*$", generated_text)
        if final_ans_match:
            answer_str = final_ans_match.group(1)
        else:
            # Fallback: Last number in the string
            # Improved regex to handle negative numbers and commas better
            numbers = re.findall(r"[-+]?\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+\.\d+|\.\d+|[-+]?\d+", generated_text)
            if numbers:
                answer_str = numbers[-1]
            else:
                return None # No number found
    try:
        # Clean the extracted string
        cleaned_answer = answer_str.replace(",", "").replace("$","").replace(" ", "").strip()
        # Remove trailing punctuation like periods
        if cleaned_answer.endswith('.'): cleaned_answer = cleaned_answer[:-1]
        if not cleaned_answer: return None
        # Convert to float if it contains a decimal, otherwise int
        return float(cleaned_answer) if '.' in cleaned_answer else int(cleaned_answer)
    except (ValueError, TypeError):
        return None # Conversion failed


def extract_mmlu_answer(generated_text):
    """Extracts the final letter answer (A, B, C, D...) from MMLU model output."""
    # Prioritize "Answer: <Letter>" format or variations
    match = re.search(r"(?:Answer|answer|ANSWER):?\s*([A-E])", generated_text)
    if match: return match.group(1).upper()

    # Prioritize letter within parentheses like (A)
    match = re.search(r"\(([A-E])\)", generated_text)
    if match: return match.group(1).upper()

    # Look for letter at the very end of the string, possibly surrounded by spaces
    match = re.search(r"([A-E])\s*$", generated_text, re.IGNORECASE)
    if match: return match.group(1).upper()

    # Fallback: First single letter A-E found in the *last line*
    lines = generated_text.strip().split('\n')
    if lines:
        last_line = lines[-1].strip()
        match = re.search(r"^([A-E])$", last_line, re.IGNORECASE) # Exact match A-E on last line
        if match: return match.group(1).upper()
        match = re.search(r"([A-E])", last_line, re.IGNORECASE) # First A-E on last line
        if match: return match.group(1).upper()

    # Absolute Fallback: First single letter A-E found anywhere
    match = re.search(r"([A-E])", generated_text, re.IGNORECASE)
    if match: return match.group(1).upper()

    return None


# --- Helper Functions for FITM ---

def find_token_indices(tokenizer, full_input_ids, content_to_find):
    """Finds the indices of tokens corresponding to specific content within the full input."""
    try:
        content_ids = tokenizer.encode(content_to_find, add_special_tokens=False)
        if not content_ids: return [] # Cannot find empty content

        full_ids_list = full_input_ids[0].tolist()
        content_len = len(content_ids)
        indices = []

        for i in range(len(full_ids_list) - content_len + 1):
            if full_ids_list[i : i + content_len] == content_ids:
                indices = list(range(i, i + content_len))
                break # Find the first occurrence

        # If not found, try encoding with adding prefix space (common tokenization artifact)
        if not indices:
             content_ids_space = tokenizer.encode(" " + content_to_find, add_special_tokens=False)
             if content_ids_space and content_ids_space != content_ids:
                 content_len_space = len(content_ids_space)
                 for i in range(len(full_ids_list) - content_len_space + 1):
                     if full_ids_list[i : i + content_len_space] == content_ids_space:
                         indices = list(range(i, i + content_len_space))
                         break

        return indices

    except Exception as e:
        warnings.warn(f"Error finding token indices for '{content_to_find[:50]}...': {e}")
        return []


def aggregate_attention_to_tokens(attentions, target_indices, source_pos):
    """
    Aggregates attention paid *from* source_pos *to* target_indices.
    Averages over layers, heads, and target tokens.
    """
    if not target_indices or attentions is None or not attentions: return 0.0

    num_layers = len(attentions)
    total_attention = 0.0
    valid_layers = 0

    # Ensure target_indices are within bounds
    max_seq_len = attentions[0].shape[-1] # Get sequence length from attention matrix
    valid_target_indices = [idx for idx in target_indices if 0 <= idx < max_seq_len]
    if not valid_target_indices: return 0.0
    if not (0 <= source_pos < max_seq_len): return 0.0 # Source position out of bounds

    try:
        target_indices_tensor = torch.tensor(valid_target_indices, device=DEVICE, dtype=torch.long)

        for layer_attn in attentions: # [batch, heads, seq_len, seq_len]
            # Get attention from source_pos to the target indices
            # layer_attn shape: [1, num_heads, seq_len, seq_len]
            attn_from_source = layer_attn[0, :, source_pos, :] # [num_heads, seq_len]
            attn_to_targets = attn_from_source[:, target_indices_tensor] # [num_heads, num_targets]

            # Average over heads and target tokens for this layer
            layer_avg_attention = attn_to_targets.mean()
            if not torch.isnan(layer_avg_attention) and not torch.isinf(layer_avg_attention):
                total_attention += layer_avg_attention.item()
                valid_layers += 1
            del layer_attn, attn_from_source, attn_to_targets, layer_avg_attention # Memory saving

        del target_indices_tensor
        if valid_layers == 0: return 0.0
        return total_attention / valid_layers

    except IndexError as e:
         warnings.warn(f"IndexError during attention aggregation (source: {source_pos}, targets: {valid_target_indices[:5]}...): {e}")
         return 0.0
    except Exception as e:
        warnings.warn(f"Error during attention aggregation: {e}")
        return 0.0


def get_demo_content_for_token_finding(demo, dataset_name):
    """Gets the representative text content of a demonstration for token index finding."""
    if dataset_name == 'gsm8k':
        # Use the question as the primary identifier
        return demo.get('question', '')
    elif dataset_name == 'mmlu':
        # Use the question and choices combined, similar to how it appears in the prompt
        q = demo.get('question', '')
        choices = demo.get('choices', [])
        letter_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
        formatted_choices = "\n".join([f"{letter_map.get(i)}. {choice}" for i, choice in enumerate(choices)])
        # Return a significant part, perhaps just the question is enough if unique
        return f"Question: {q}" # Simplified for robustness
    else:
        # Fallback: try to get 'question' or 'text'
        return demo.get('question', demo.get('text', ''))


# --- Core ICL Logic ---

# (Keep run_standard_icl function as is)
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

        # Ensure prompt+generation doesn't exceed max length
        max_len_for_gen = max_model_len - max_new_tokens - 5 # Add small buffer

        if prompt_tokens > max_len_for_gen:
             warnings.warn(f"Standard ICL ({dataset_name}): Prompt tokens ({prompt_tokens}) too long for max_new_tokens ({max_new_tokens}) within max_len ({max_model_len}). Truncating from left.")
             inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len_for_gen, truncation_side='left').to(DEVICE)
             prompt_tokens = inputs["input_ids"].shape[1]
             if prompt_tokens >= max_len_for_gen: # Check again after truncation
                  warnings.warn(f"Standard ICL ({dataset_name}): Prompt still too long ({prompt_tokens}) after truncation. Skipping.")
                  return None
        else:
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}


    except Exception as e:
        warnings.warn(f"Standard ICL ({dataset_name}): Error tokenizing prompt: {e}")
        return None

    generated_text = ""
    gen_start_index = inputs["input_ids"].shape[1]
    try:
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                do_sample=False, # Use greedy decoding for consistency
                temperature=0.0,
            )
        # Ensure correct decoding slice
        if outputs.shape[1] > gen_start_index:
             generated_ids = outputs[0, gen_start_index:]
             generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        else:
             generated_text = "" # No new tokens generated

    except torch.cuda.OutOfMemoryError:
         warnings.warn(f"CUDA OOM during Standard ICL generation ({dataset_name}). Skipping.")
         return None
    except Exception as e:
         warnings.warn(f"Error during Standard ICL generation ({dataset_name}): {e}")
         traceback.print_exc()
         return None
    finally:
        # Clean up memory safely
        if 'inputs' in locals():
            del inputs
        if 'outputs' in locals():
            del outputs
        if DEVICE == "cuda": torch.cuda.empty_cache()


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

# --- NEW: FITM ICL Implementation ---
def run_fitm_icl(model, tokenizer, dataset_name, demonstrations, query,
                 fitm_temperature, dummy_content,
                 choices=None, max_new_tokens=None):
    """
    Runs Found-in-the-Middle (FITM) ICL using attention calibration.
    """
    if not demonstrations:
        warnings.warn("FITM requires demonstrations for calibration. Running standard ICL.")
        return run_standard_icl(model, tokenizer, dataset_name, demonstrations, query, choices=choices, max_new_tokens=max_new_tokens)

    if fitm_temperature <= 0:
        warnings.warn(f"FITM temperature must be > 0 (got {fitm_temperature}). Using 1e-5.")
        fitm_temperature = 1e-5

    num_demos = len(demonstrations)
    baseline_attentions = {} # {pos_idx: attention_score}
    doc_token_indices = {} # {pos_idx: [token_indices]}
    fallback_to_standard = False

    # --- Step 1: Calibration Pass (Measure baseline attention with dummy docs) ---
    # print("FITM: Starting Calibration Pass") # DEBUG
    dummy_demo_template = {'question': dummy_content, 'answer': 'OK.'} if dataset_name == 'gsm8k' else \
                          {'question': dummy_content, 'choices': ['A','B'], 'answer': 0} # MMLU needs choices/answer keys

    for k in range(num_demos):
        calib_demos = list(demonstrations) # Create a copy
        original_demo_content = get_demo_content_for_token_finding(calib_demos[k], dataset_name) # Store original content
        calib_demos[k] = dummy_demo_template # Replace with dummy

        if dataset_name == 'gsm8k':
            calib_prompt = format_gsm8k_icl_input(tokenizer, calib_demos, query, include_answer_prompt=True)
        elif dataset_name == 'mmlu':
            calib_prompt = format_mmlu_icl_input(tokenizer, calib_demos, query, choices, include_answer_prompt=True)
        else: raise ValueError(f"Unsupported dataset: {dataset_name}")

        if not calib_prompt:
            warnings.warn(f"FITM Calibration (k={k}): Empty prompt. Skipping position.")
            baseline_attentions[k] = 0.0 # Assign default? Or fallback?
            continue

        try:
            inputs_calib = tokenizer(calib_prompt, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length - 10).to(DEVICE) # Truncate if needed
            input_ids_calib = inputs_calib["input_ids"]
            if input_ids_calib.shape[1] >= tokenizer.model_max_length - 5: # Check length after truncation
                 warnings.warn(f"FITM Calibration (k={k}): Prompt too long even after truncation. Skipping position.")
                 baseline_attentions[k] = 0.0
                 continue

            # Find dummy token indices *in this specific prompt*
            dummy_indices = find_token_indices(tokenizer, input_ids_calib, dummy_content)
            if not dummy_indices:
                warnings.warn(f"FITM Calibration (k={k}): Could not find dummy tokens. Skipping position.")
                baseline_attentions[k] = 0.0
                continue

            with torch.no_grad():
                outputs_calib = model(input_ids=input_ids_calib, attention_mask=inputs_calib["attention_mask"], output_attentions=True)
                attentions_calib = outputs_calib.attentions

            last_token_pos_calib = input_ids_calib.shape[1] - 1
            baseline_attn = aggregate_attention_to_tokens(attentions_calib, dummy_indices, last_token_pos_calib)
            baseline_attentions[k] = baseline_attn if not (np.isnan(baseline_attn) or np.isinf(baseline_attn)) else 0.0

            del inputs_calib, outputs_calib, attentions_calib, dummy_indices # Memory clean
            if DEVICE == "cuda": torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            warnings.warn(f"CUDA OOM during FITM Calibration (k={k}). Falling back to standard ICL.")
            fallback_to_standard = True; break
        except Exception as e:
            warnings.warn(f"Error during FITM Calibration (k={k}): {e}. Assigning 0 baseline.")
            traceback.print_exc()
            baseline_attentions[k] = 0.0 # Assign 0 on error for this pos

    if fallback_to_standard:
         if DEVICE == "cuda": torch.cuda.empty_cache()
         return run_standard_icl(model, tokenizer, dataset_name, demonstrations, query, choices=choices, max_new_tokens=max_new_tokens)

    # --- Step 2: Main Prompt Attention Pass (Measure attention to real demos) ---
    # print("FITM: Starting Main Attention Pass") # DEBUG
    doc_attentions = {} # {pos_idx: attention_score}
    if dataset_name == 'gsm8k':
        main_prompt = format_gsm8k_icl_input(tokenizer, demonstrations, query, include_answer_prompt=True)
    elif dataset_name == 'mmlu':
        main_prompt = format_mmlu_icl_input(tokenizer, demonstrations, query, choices, include_answer_prompt=True)
    else: raise ValueError(f"Unsupported dataset: {dataset_name}")

    if not main_prompt:
         warnings.warn("FITM Main Pass: Empty prompt generated. Falling back to standard ICL.")
         return run_standard_icl(model, tokenizer, dataset_name, demonstrations, query, choices=choices, max_new_tokens=max_new_tokens)

    try:
        inputs_main = tokenizer(main_prompt, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length - 10).to(DEVICE)
        input_ids_main = inputs_main["input_ids"]
        attention_mask_main = inputs_main["attention_mask"] # Keep original mask for generation
        if input_ids_main.shape[1] >= tokenizer.model_max_length - 5:
             warnings.warn("FITM Main Pass: Prompt too long after truncation. Falling back.")
             del inputs_main # Clean up before fallback
             return run_standard_icl(model, tokenizer, dataset_name, demonstrations, query, choices=choices, max_new_tokens=max_new_tokens)

        last_token_pos_main = input_ids_main.shape[1] - 1

        # Find token indices for all *real* demonstrations in the main prompt
        all_demo_indices_found = True
        for k in range(num_demos):
            demo_content = get_demo_content_for_token_finding(demonstrations[k], dataset_name)
            indices = find_token_indices(tokenizer, input_ids_main, demo_content)
            if not indices:
                 warnings.warn(f"FITM Main Pass: Could not find tokens for demo {k}. Assigning 0 attention.")
                 doc_token_indices[k] = []
                 doc_attentions[k] = 0.0
                 # If even one demo isn't found, calibration might be unreliable
                 # Consider falling back completely? For now, assign 0.
                 # all_demo_indices_found = False
            else:
                 doc_token_indices[k] = indices

        # Get attentions for the main prompt
        with torch.no_grad():
            outputs_main = model(input_ids=input_ids_main, attention_mask=attention_mask_main, output_attentions=True)
            attentions_main = outputs_main.attentions

        # Calculate attention to each real demo
        for k in range(num_demos):
            if doc_token_indices.get(k): # Only if indices were found
                attn = aggregate_attention_to_tokens(attentions_main, doc_token_indices[k], last_token_pos_main)
                doc_attentions[k] = attn if not (np.isnan(attn) or np.isinf(attn)) else 0.0
            # else: doc_attentions[k] is already 0.0

        del outputs_main, attentions_main # Memory clean

    except torch.cuda.OutOfMemoryError:
        warnings.warn("CUDA OOM during FITM Main Attention Pass. Falling back to standard ICL.")
        fallback_to_standard = True
    except Exception as e:
        warnings.warn(f"Error during FITM Main Attention Pass: {e}. Falling back.")
        traceback.print_exc()
        fallback_to_standard = True

    if fallback_to_standard:
         del inputs_main # Clean up
         if DEVICE == "cuda": torch.cuda.empty_cache()
         return run_standard_icl(model, tokenizer, dataset_name, demonstrations, query, choices=choices, max_new_tokens=max_new_tokens)

    # --- Step 3: Calculate Calibrated Relevance & Scaling Factors ---
    # print("FITM: Calculating Calibrated Relevance") # DEBUG
    calibrated_relevance = []
    min_baseline = min(baseline_attentions.values()) if baseline_attentions else 0
    max_doc_attn = max(doc_attentions.values()) if doc_attentions else 0

    for k in range(num_demos):
        doc_attn = doc_attentions.get(k, 0.0)
        base_attn = baseline_attentions.get(k, 0.0) # Use 0 if calibration failed for this pos
        # Simple subtraction as per paper's implication
        relevance = doc_attn - base_attn
        calibrated_relevance.append(relevance)

    if not calibrated_relevance: # Should not happen if demos exist, but check
         warnings.warn("FITM: No calibrated relevance scores. Falling back.")
         del inputs_main
         return run_standard_icl(model, tokenizer, dataset_name, demonstrations, query, choices=choices, max_new_tokens=max_new_tokens)

    # Apply Softmax with temperature to get scaling factors
    try:
        relevance_tensor = torch.tensor(calibrated_relevance, dtype=torch.float32, device=DEVICE)
        # Add small epsilon to prevent log(0) if temp is very small? Softmax handles 0 well.
        scale_factors = F.softmax(relevance_tensor / fitm_temperature, dim=0)
        # print(f"FITM Relevance: {relevance_tensor.cpu().numpy()}") # DEBUG
        # print(f"FITM Scale Factors (temp={fitm_temperature}): {scale_factors.cpu().numpy()}") # DEBUG

    except Exception as e:
         warnings.warn(f"FITM: Error calculating scale factors: {e}. Falling back.")
         del inputs_main, relevance_tensor
         return run_standard_icl(model, tokenizer, dataset_name, demonstrations, query, choices=choices, max_new_tokens=max_new_tokens)


    # --- Step 4: Create Calibrated Attention Mask ---
    # print("FITM: Creating Calibrated Mask") # DEBUG
    attention_mask_calibrated = attention_mask_main.clone().float() # Use float mask for scaling

    for k in range(num_demos):
        indices = doc_token_indices.get(k)
        if indices:
            scale_k = scale_factors[k].item()
            indices_tensor = torch.tensor(indices, device=DEVICE, dtype=torch.long)
            # Clamp indices to be within mask bounds (shouldn't be needed with truncation checks)
            valid_indices = indices_tensor[indices_tensor < attention_mask_calibrated.shape[1]]
            if valid_indices.numel() > 0:
                 attention_mask_calibrated[0, valid_indices] *= scale_k # Scale existing mask values

    # Ensure query part remains 1.0? Assume it's already 1.0 and not part of demos.
    # Ensure mask values are valid (e.g., clamp between 0 and 1 if needed, though softmax ensures >=0)
    # attention_mask_calibrated = torch.clamp(attention_mask_calibrated, 0.0, 1.0) # Optional clamp

    # --- Step 5: Generate with Calibrated Mask ---
    # print("FITM: Starting Generation") # DEBUG
    generated_text_fitm = ""
    gen_start_index = input_ids_main.shape[1]
    try:
        with torch.no_grad():
            outputs_fitm = model.generate(
                input_ids=input_ids_main,
                attention_mask=attention_mask_calibrated, # Use the scaled mask
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                do_sample=False,
                temperature=0.0,
            )

        if outputs_fitm.shape[1] > gen_start_index:
            generated_ids_fitm = outputs_fitm[0, gen_start_index:]
            generated_text_fitm = tokenizer.decode(generated_ids_fitm, skip_special_tokens=True)
        else:
            generated_text_fitm = ""

    except torch.cuda.OutOfMemoryError:
         warnings.warn(f"CUDA OOM during FITM generation. Skipping.")
         generated_text_fitm = None # Indicate error
    except Exception as e:
         warnings.warn(f"Error during FITM generation: {e}. Skipping.")
         traceback.print_exc()
         generated_text_fitm = None # Indicate error
        # ... (inside run_fitm_icl, after generation)
    finally:
        # Extensive cleanup
        # Check existence before deleting
        if 'inputs_main' in locals(): del inputs_main
        if 'input_ids_main' in locals(): del input_ids_main
        if 'attention_mask_main' in locals(): del attention_mask_main
        if 'attention_mask_calibrated' in locals(): del attention_mask_calibrated
        if 'doc_attentions' in locals(): del doc_attentions
        if 'baseline_attentions' in locals(): del baseline_attentions
        if 'doc_token_indices' in locals(): del doc_token_indices
        if 'calibrated_relevance' in locals(): del calibrated_relevance
        if 'relevance_tensor' in locals(): del relevance_tensor
        if 'scale_factors' in locals(): del scale_factors
        if 'outputs_fitm' in locals(): del outputs_fitm
        if 'generated_ids_fitm' in locals(): del generated_ids_fitm
        # Also clean up calibration loop variables if they might persist on error
        if 'inputs_calib' in locals(): del inputs_calib
        if 'outputs_calib' in locals(): del outputs_calib
        if 'attentions_calib' in locals(): del attentions_calib

        if DEVICE == "cuda": torch.cuda.empty_cache()

    # --- Step 6: Extract Answer ---
    if generated_text_fitm is None: return None # Generation failed

    if dataset_name == 'gsm8k':
        extractor_func = extract_gsm8k_answer
    elif dataset_name == 'mmlu':
        extractor_func = extract_mmlu_answer
    else: raise ValueError(f"Unsupported dataset: {dataset_name}")

    return extractor_func(generated_text_fitm)



# --- Evaluation Loop (GSM8K) ---
# (Modify evaluate_gsm8k to handle FITM arguments)
def evaluate_gsm8k(model, tokenizer, dataset, method_func, n_demos,
                   p_threshold=None, # For FOCUS
                   fitm_temperature=None, dummy_content=None, # For FITM
                   num_test_samples=None, max_new_tokens=None):
    """Evaluates a given ICL method on the GSM8K test set."""
    correct = 0
    total = 0
    errors = 0
    skipped_long = 0
    demonstration_pool = list(dataset['train'])
    test_set = dataset['test']
    dataset_name = "gsm8k"

    if max_new_tokens is None: raise ValueError("evaluate_gsm8k requires max_new_tokens")

    if num_test_samples is None: num_test_samples = len(test_set)
    else: num_test_samples = min(num_test_samples, len(test_set))
    test_samples = test_set.select(range(num_test_samples))

    actual_n_demos_pool = min(n_demos, len(demonstration_pool)) if demonstration_pool else 0
    if len(demonstration_pool) < n_demos:
         warnings.warn(f"GSM8K: Demo pool ({len(demonstration_pool)}) < required ({n_demos}). Using {actual_n_demos_pool}.")

    method_name = method_func.__name__
    desc = f"Eval GSM8K {method_name} (k={n_demos}"
    if method_name == 'run_focus_icl' and p_threshold is not None: desc += f", p={p_threshold}"
    if method_name == 'run_fitm_icl' and fitm_temperature is not None: desc += f", temp={fitm_temperature}"
    desc += f", N={num_test_samples})"
    pbar = tqdm(test_samples, desc=desc)

    results_log = []

    for example in pbar:
        query = example['question']
        true_answer_text = example['answer']
        true_answer_val = extract_gsm8k_answer(true_answer_text)
        example_id = example.get('id', 'N/A')

        if true_answer_val is None:
            # warnings.warn(f"Could not parse true answer for {example_id}. Skipping.")
            errors += 1
            results_log.append({'id': example_id, 'status': 'error_parsing_truth', 'pred': None, 'true': true_answer_text})
            continue

        # Ensure demos are sampled *without* the current test example if it exists in train
        potential_demos = [d for d in demonstration_pool if d.get('id') != example_id]
        current_n_demos = min(actual_n_demos_pool, len(potential_demos))

        # Need demos for FOCUS and FITM
        if current_n_demos < actual_n_demos_pool and method_name != 'run_standard_icl' and n_demos > 0:
             warnings.warn(f"Skipping {example_id} for {method_name} k={n_demos} - Not enough unique demos ({current_n_demos}/{actual_n_demos_pool})")
             errors += 1
             results_log.append({'id': example_id, 'status': 'error_insufficient_demos', 'pred': None, 'true': true_answer_val})
             continue

        demos = random.sample(potential_demos, current_n_demos) if current_n_demos > 0 else []

        predicted_answer_val = None
        status = 'ok'
        try:
            kwargs = {"model": model, "tokenizer": tokenizer, "dataset_name": dataset_name,
                      "demonstrations": demos, "query": query, "max_new_tokens": max_new_tokens}

            if method_name == 'run_focus_icl':
                 if p_threshold is None: raise ValueError("p_threshold needed for FOCUS")
                 kwargs["p_threshold"] = p_threshold
            elif method_name == 'run_fitm_icl':
                 if fitm_temperature is None: raise ValueError("fitm_temperature needed for FITM")
                 if dummy_content is None: raise ValueError("dummy_content needed for FITM")
                 kwargs["fitm_temperature"] = fitm_temperature
                 kwargs["dummy_content"] = dummy_content
            # Standard ICL takes no extra args

            predicted_answer_val = method_func(**kwargs)

            if predicted_answer_val is None:
                 errors += 1
                 status = 'error_no_pred'
                 # Check for specific warnings like prompt too long?
                 # This requires capturing warnings, which is complex here. Assume None means error.

        except (torch.cuda.OutOfMemoryError, RuntimeError) as mem_e:
             print(f"\n!!! OOM/Runtime Error ({method_name}, {example_id}): {mem_e}")
             status = 'error_oom_runtime'
             errors += 1
             predicted_answer_val = None
             # Try to clear cache aggressively
             if DEVICE == "cuda": torch.cuda.empty_cache()
        except Exception as e:
             print(f"\n!!! GSM8K Eval Error ({method_name}, {example_id}): {e}")
             traceback.print_exc()
             status = 'error_exception'
             errors += 1
             predicted_answer_val = None

        is_correct = False
        if predicted_answer_val is not None:
            try:
                # Use a slightly more robust comparison for floats
                if isinstance(true_answer_val, float) or isinstance(predicted_answer_val, float):
                    if math.isclose(float(predicted_answer_val), float(true_answer_val), rel_tol=1e-4, abs_tol=1e-6):
                        is_correct = True
                # Exact match for integers
                elif int(predicted_answer_val) == int(true_answer_val):
                     is_correct = True
            except (ValueError, TypeError):
                 # Fallback string comparison if numeric fails
                 if str(predicted_answer_val).strip() == str(true_answer_val).strip():
                     is_correct = True

        if status == 'ok':
             total += 1
             if is_correct:
                 correct += 1
             results_log.append({'id': example_id, 'status': status, 'pred': predicted_answer_val, 'true': true_answer_val, 'correct': is_correct})
        else:
             # Log errors without incrementing total attempts count
             results_log.append({'id': example_id, 'status': status, 'pred': None, 'true': true_answer_val})


        # Update progress bar
        accuracy = correct / total if total > 0 else 0
        pbar.set_postfix({"Acc": f"{accuracy:.3f}", "Correct": correct, "Attempted": total, "Errors": errors})
        # Clean cache periodically inside loop? Maybe too slow.
        # if (pbar.n + 1) % 20 == 0 and DEVICE == "cuda": torch.cuda.empty_cache()


    accuracy = correct / total if total > 0 else 0
    print(f"\n--- GSM8K Eval Complete ({method_name}, k={n_demos}"
          f"{f', p={p_threshold}' if method_name=='run_focus_icl' and p_threshold is not None else ''}"
          f"{f', temp={fitm_temperature}' if method_name=='run_fitm_icl' and fitm_temperature is not None else ''}) ---")
    print(f"Final Accuracy: {accuracy:.4f} ({correct}/{total}) | Errors/Skipped: {errors}")
    print("-" * 30)
    # Optionally save detailed log
    # with open(f"results_log_{dataset_name}_{method_name}_k{n_demos}.json", "w") as f:
    #    json.dump(results_log, f, indent=2)
    return accuracy


# --- Evaluation Loop (MMLU) ---
def evaluate_mmlu(model, tokenizer, dataset, mmlu_subset_name, method_func, n_demos,
                  p_threshold=None, # For FOCUS
                  fitm_temperature=None, dummy_content=None, # For FITM
                  num_test_samples=None, max_new_tokens=None):
    """Evaluates a given ICL method on the MMLU test set for a specific subset."""
    correct = 0
    total = 0
    errors = 0
    demonstration_pool = list(dataset['dev'])
    test_set = dataset['test']
    dataset_name = "mmlu"

    if max_new_tokens is None: raise ValueError("evaluate_mmlu requires max_new_tokens")

    if num_test_samples is None: num_test_samples = len(test_set)
    else: num_test_samples = min(num_test_samples, len(test_set))
    test_samples = test_set.select(range(num_test_samples))

    actual_n_demos_pool = min(n_demos, len(demonstration_pool)) if demonstration_pool else 0
    if len(demonstration_pool) < n_demos:
        warnings.warn(f"MMLU ({mmlu_subset_name}): Demo pool ({len(demonstration_pool)}) < required ({n_demos}). Using {actual_n_demos_pool}.")

    method_name = method_func.__name__
    desc = f"Eval MMLU/{mmlu_subset_name} {method_name} (k={n_demos}"
    if method_name == 'run_focus_icl' and p_threshold is not None: desc += f", p={p_threshold}"
    if method_name == 'run_fitm_icl' and fitm_temperature is not None: desc += f", temp={fitm_temperature}"
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
            # warnings.warn(f"Could not parse true answer index for {example_id}. Skipping.")
            errors += 1
            continue

        potential_demos = [d for d in demonstration_pool if d.get('id') != example_id]
        current_n_demos = min(actual_n_demos_pool, len(potential_demos))

        if current_n_demos < actual_n_demos_pool and method_name != 'run_standard_icl' and n_demos > 0:
             warnings.warn(f"Skipping {example_id} for {method_name} k={n_demos} - Not enough unique demos ({current_n_demos}/{actual_n_demos_pool})")
             errors += 1
             continue

        demos = random.sample(potential_demos, current_n_demos) if current_n_demos > 0 else []

        predicted_answer_letter = None
        status = 'ok'
        try:
            kwargs = {"model": model, "tokenizer": tokenizer, "dataset_name": dataset_name,
                      "demonstrations": demos, "query": query, "choices": choices,
                      "max_new_tokens": max_new_tokens}

            if method_name == 'run_focus_icl':
                 if p_threshold is None: raise ValueError("p_threshold needed for FOCUS")
                 kwargs["p_threshold"] = p_threshold
            elif method_name == 'run_fitm_icl':
                 if fitm_temperature is None: raise ValueError("fitm_temperature needed for FITM")
                 if dummy_content is None: raise ValueError("dummy_content needed for FITM")
                 kwargs["fitm_temperature"] = fitm_temperature
                 kwargs["dummy_content"] = dummy_content
            # Standard ICL takes no extra args

            predicted_answer_letter = method_func(**kwargs)

            if predicted_answer_letter is None:
                 errors += 1
                 status = 'error_no_pred'

        except (torch.cuda.OutOfMemoryError, RuntimeError) as mem_e:
             print(f"\n!!! OOM/Runtime Error ({method_name}, {example_id}): {mem_e}")
             status = 'error_oom_runtime'
             errors += 1
             predicted_answer_letter = None
             if DEVICE == "cuda": torch.cuda.empty_cache()
        except Exception as e:
             print(f"\n!!! MMLU Eval Error ({method_name}, {example_id}): {e}")
             traceback.print_exc()
             status = 'error_exception'
             errors += 1
             predicted_answer_letter = None

        is_correct = False
        if predicted_answer_letter is not None:
            if predicted_answer_letter == true_answer_letter:
                is_correct = True

        if status == 'ok':
             total += 1
             if is_correct:
                 correct += 1
        # else: Error already incremented

        # Update progress bar
        accuracy = correct / total if total > 0 else 0
        pbar.set_postfix({"Acc": f"{accuracy:.3f}", "Correct": correct, "Attempted": total, "Errors": errors})

    accuracy = correct / total if total > 0 else 0
    print(f"\n--- MMLU ({mmlu_subset_name}) Eval Complete ({method_name}, k={n_demos}"
          f"{f', p={p_threshold}' if method_name=='run_focus_icl' and p_threshold is not None else ''}"
          f"{f', temp={fitm_temperature}' if method_name=='run_fitm_icl' and fitm_temperature is not None else ''}) ---")
    print(f"Final Accuracy: {accuracy:.4f} ({correct}/{total}) | Errors/Skipped: {errors}")
    print("-" * 30)
    return accuracy


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Standard ICL and FITM ICL on GSM8K or MMLU dataset.")
    parser.add_argument(
        "--model_name", type=str, required=True,
        help="Required. HF model name/path (e.g., 'google/gemma-2b-it')."
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
        "--num_eval_samples", type=int, default=100, # Reduced default for faster runs, esp. with FITM
        help="Number of test samples per evaluation run. Default: 100."
    )
    parser.add_argument(
        "--k_values", nargs='+', type=int, default=[5, 10], # Smaller K for faster FITM runs
        help="List of k values (demos) to test. Example: --k_values 0 5 10"
    )
    # FOCUS arguments (keep if comparing)
    parser.add_argument(
        "--run_focus", action="store_true",
        help="Run FOCUS ICL evaluation (requires model with output_attentions=True)."
    )
    parser.add_argument(
        "--p_values", nargs='+', type=float, default=[0.2],
        help="List of p thresholds for FOCUS ICL. Used only if --run_focus is set. Example: --p_values 0.1 0.25"
    )
    # FITM arguments
    parser.add_argument(
        "--run_fitm", action="store_true",
        help="Run FITM ICL evaluation (requires model with output_attentions=True)."
    )
    parser.add_argument(
        "--fitm_temperatures", nargs='+', type=float, default=[1e-4, 5e-5], # Example temperatures from paper/common sense
        help="List of temperatures for FITM calibration softmax. Used only if --run_fitm is set. Example: --fitm_temperatures 1e-4 5e-5"
    )
    parser.add_argument(
        "--fitm_dummy_content", type=str, default=DUMMY_CONTENT,
        help=f"Dummy content string for FITM calibration. Default: '{DUMMY_CONTENT}'"
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
    # Add HF token argument if needed for gated models
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face Hub token for gated models.")


    args = parser.parse_args()

    # --- Validate Arguments ---
    if args.dataset == 'mmlu' and not args.mmlu_subset:
        parser.error("--mmlu_subset is required when --dataset=mmlu")
    if args.dataset == 'gsm8k' and args.mmlu_subset:
        warnings.warn("--mmlu_subset is ignored when --dataset=gsm8k")
        args.mmlu_subset = None # Clear it
    if (args.run_focus or args.run_fitm) and args.force_cpu:
        warnings.warn("Running FOCUS or FITM on CPU will be extremely slow due to attention calculations.")
    if not args.run_focus:
        args.p_values = [] # Clear p_values if not running FOCUS
    if not args.run_fitm:
        args.fitm_temperatures = [] # Clear fitm_temperatures if not running FITM

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
    P_THRESHOLDS = sorted(list(set(args.p_values))) if args.run_focus else []
    FITM_TEMPERATURES = sorted(list(set(args.fitm_temperatures))) if args.run_fitm else []
    DUMMY_CONTENT = args.fitm_dummy_content
    SEED = args.seed
    if args.force_cpu: DEVICE = "cpu"
    HF_TOKEN = args.hf_token # Store HF token


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
    print(f"Run Standard ICL: True")
    print(f"Run FOCUS ICL: {args.run_focus}" + (f" (p={P_THRESHOLDS})" if args.run_focus else ""))
    print(f"Run FITM ICL: {args.run_fitm}" + (f" (temps={FITM_TEMPERATURES})" if args.run_fitm else ""))
    if args.run_fitm: print(f"FITM Dummy Content: '{DUMMY_CONTENT}'")
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
    # Determine if attention output is needed
    output_attentions_required = args.run_focus or args.run_fitm
    if output_attentions_required:
        print("Loading model with output_attentions=True")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype="auto",
            device_map="auto" if DEVICE == "cuda" else None,
            trust_remote_code=True,
            output_attentions=output_attentions_required, # Set dynamically
            token=HF_TOKEN # Pass token here
            # attn_implementation="flash_attention_2" # FA2 might interfere with getting attentions, test carefully
        )
        # Verify attention output setting if required
        if output_attentions_required and not model.config.output_attentions:
             warnings.warn(f"Model {MODEL_NAME} loaded BUT model.config.output_attentions is False. FOCUS/FITM will likely fail or fall back.")
             # Attempt to force it? Might not be supported by all models.
             # model.config.output_attentions = True
             # print("Attempted to force model.config.output_attentions=True")


        if DEVICE == "cpu" and ("auto" not in str(model.device) if hasattr(model, 'device') else True):
             model.to(DEVICE) # Ensure CPU placement if forced and device_map didn't handle it
        model.eval()
        print(f"Model loaded successfully to device: {model.device}")
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
            # Add a pad token if none exists - important for batching/masking
            print("Adding pad token")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
            if model.config.pad_token_id is None: model.config.pad_token_id = tokenizer.pad_token_id

    elif model.config.pad_token_id is None:
         model.config.pad_token_id = tokenizer.pad_token_id

    # Ensure model_max_length is set
    if not hasattr(tokenizer, 'model_max_length') or not tokenizer.model_max_length:
        if hasattr(model.config, 'max_position_embeddings') and model.config.max_position_embeddings:
            tokenizer.model_max_length = model.config.max_position_embeddings
            print(f"Set tokenizer.model_max_length from model config: {tokenizer.model_max_length}")
        else:
            tokenizer.model_max_length = 2048 # More conservative fallback
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

        if DEVICE == "cuda": torch.cuda.empty_cache(); print("Cleared CUDA Cache")

        # --- Evaluate FOCUS ICL ---
        if args.run_focus:
            if not model.config.output_attentions:
                print(f"\n--- Skipping FOCUS ICL for k={n_demos} (model config output_attentions=False) ---")
                for p_val in P_THRESHOLDS: results[n_demos][f"FOCUS_p{p_val}"] = np.nan
            else:
                for p_val in P_THRESHOLDS:
                    focus_key = f"FOCUS_p{p_val}"
                    print(f"\n--- Evaluating FOCUS ICL (k={n_demos}, p={p_val}) ---")
                    # !!! Make sure run_focus_icl exists if --run_focus is used !!!
                    if 'run_focus_icl' not in globals():
                         print("!!! run_focus_icl function not defined. Skipping FOCUS.")
                         results[n_demos][focus_key] = np.nan
                         continue
                    try:
                        eval_args = {
                           "model": model, "tokenizer": tokenizer, "dataset": eval_dataset,
                           "method_func": run_focus_icl, "n_demos": n_demos, "p_threshold": p_val,
                           "num_test_samples": NUM_EVAL_SAMPLES, "max_new_tokens": MAX_NEW_TOKENS
                        }
                        if DATASET_NAME == 'mmlu':
                            eval_args["mmlu_subset_name"] = MMLU_SUBSET
                            # Need to pass choices for MMLU FOCUS eval - modify evaluate_mmlu if needed
                            # Currently evaluate_mmlu does not explicitly pass choices to method_func
                            warnings.warn("MMLU FOCUS evaluation might need adjustment in evaluate_mmlu to pass choices.")


                        focus_icl_accuracy = evaluate_func(**eval_args)
                        results[n_demos][focus_key] = focus_icl_accuracy if isinstance(focus_icl_accuracy, (float, int)) else np.nan
                    except Exception as e:
                        print(f"!! FATAL ERROR during FOCUS ICL eval (k={n_demos}, p={p_val}): {e}")
                        results[n_demos][focus_key] = np.nan
                        traceback.print_exc()
        else:
             print(f"\n--- Skipping FOCUS ICL for k={n_demos} (--run_focus not specified) ---")

        # --- Evaluate FITM ICL ---
        if args.run_fitm:
            if not model.config.output_attentions:
                 print(f"\n--- Skipping FITM ICL for k={n_demos} (model config output_attentions=False) ---")
                 for temp_val in FITM_TEMPERATURES: results[n_demos][f"FITM_t{temp_val:.0e}"] = np.nan
            else:
                 for temp_val in FITM_TEMPERATURES:
                    fitm_key = f"FITM_t{temp_val:.0e}" # Use scientific notation for key
                    print(f"\n--- Evaluating FITM ICL (k={n_demos}, temp={temp_val:.0e}) ---")
                    try:
                        eval_args = {
                            "model": model, "tokenizer": tokenizer, "dataset": eval_dataset,
                            "method_func": run_fitm_icl, "n_demos": n_demos,
                            "fitm_temperature": temp_val, "dummy_content": DUMMY_CONTENT,
                            "num_test_samples": NUM_EVAL_SAMPLES, "max_new_tokens": MAX_NEW_TOKENS
                        }
                        if DATASET_NAME == 'mmlu': eval_args["mmlu_subset_name"] = MMLU_SUBSET

                        fitm_icl_accuracy = evaluate_func(**eval_args)
                        results[n_demos][fitm_key] = fitm_icl_accuracy if isinstance(fitm_icl_accuracy, (float, int)) else np.nan
                    except Exception as e:
                        print(f"!! FATAL ERROR during FITM ICL eval (k={n_demos}, temp={temp_val}): {e}")
                        results[n_demos][fitm_key] = np.nan
                        traceback.print_exc()
        else:
            print(f"\n--- Skipping FITM ICL for k={n_demos} (--run_fitm not specified) ---")


        print(f"===== Finished Experiments for k={n_demos} =====")


    # --- Print Summary ---
    dataset_print_name = f"{DATASET_NAME}" + (f" - {MMLU_SUBSET}" if MMLU_SUBSET else "")
    print(f"\n\n--- Overall Evaluation Summary ({dataset_print_name}) ---")
    # ... (rest of summary printing and plotting code - needs modification to include FITM) ...

    # --- MODIFIED Print Summary & Save ---
    print(f"\n\n--- Overall Evaluation Summary ({dataset_print_name}) ---")
    print(f"Model: {MODEL_NAME}")
    print(f"Test Samples per run: {NUM_EVAL_SAMPLES}")
    if args.run_focus: print(f"FOCUS p_thresholds: {P_THRESHOLDS}")
    if args.run_fitm: print(f"FITM temperatures: {FITM_TEMPERATURES}")

    # Build header dynamically
    header_parts = ["| k (n_demos) | Standard ICL Acc |"]
    col_width_k = 13
    col_width_std = 18
    col_width_focus = 19 # e.g., " FOCUS (p=0.2) Acc |"
    col_width_fitm = 20 # e.g., " FITM (t=1e-04) Acc |"

    if args.run_focus:
        for p_val in P_THRESHOLDS: header_parts.append(f" FOCUS (p={p_val}) Acc |")
    if args.run_fitm:
        for temp_val in FITM_TEMPERATURES: header_parts.append(f" FITM (t={temp_val:.0e}) Acc |")
    header = "".join(header_parts)

    # Build separator dynamically
    separator_parts = [f"|{'-'*col_width_k}|{'-'*col_width_std}|"]
    if args.run_focus:
        for _ in P_THRESHOLDS: separator_parts.append(f"{'-'*col_width_focus}|")
    if args.run_fitm:
        for _ in FITM_TEMPERATURES: separator_parts.append(f"{'-'*col_width_fitm}|")
    separator = "".join(separator_parts)
    table_width = len(separator)

    print("-" * table_width)
    print(header)
    print(separator)

    summary_lines = [
        f"--- Overall Evaluation Summary ({dataset_print_name}) ---",
        f"Model: {MODEL_NAME}",
        f"Device: {DEVICE}",
        f"Test Samples per run: {NUM_EVAL_SAMPLES}",
        f"K values tested: {N_DEMOS_LIST}",
        f"Run Standard ICL: True",
        f"Run FOCUS ICL: {args.run_focus}" + (f" (p={P_THRESHOLDS})" if args.run_focus else ""),
        f"Run FITM ICL: {args.run_fitm}" + (f" (temps={FITM_TEMPERATURES})" if args.run_fitm else ""),
        f"Max New Tokens: {MAX_NEW_TOKENS}",
        f"Seed: {SEED}",
        f"Evaluation Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n",
        "-" * table_width, header, separator
    ]
    sorted_k = sorted(results.keys())
    for n_demos in sorted_k:
        scores = results[n_demos]
        icl_score = scores.get('ICL', np.nan)
        icl_score_str = f"{icl_score:.4f}" if not np.isnan(icl_score) else "Error/NaN"
        row_parts = [f"| {n_demos:<{col_width_k-2}} | {icl_score_str:<{col_width_std-2}} |"]

        if args.run_focus:
            for p_val in P_THRESHOLDS:
                focus_key = f"FOCUS_p{p_val}"
                focus_score = scores.get(focus_key, np.nan)
                focus_score_str = f"{focus_score:.4f}" if not np.isnan(focus_score) else "Error/NaN"
                row_parts.append(f" {focus_score_str:<{col_width_focus-2}} |")
        if args.run_fitm:
            for temp_val in FITM_TEMPERATURES:
                fitm_key = f"FITM_t{temp_val:.0e}"
                fitm_score = scores.get(fitm_key, np.nan)
                fitm_score_str = f"{fitm_score:.4f}" if not np.isnan(fitm_score) else "Error/NaN"
                row_parts.append(f" {fitm_score_str:<{col_width_fitm-2}} |")

        row = "".join(row_parts)
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
        methods_str = "ICL"
        if args.run_focus: methods_str += f"_FOCUSp{''.join(map(str, P_THRESHOLDS))}"
        if args.run_fitm: methods_str += f"_FITMt{''.join([f'{t:.0e}' for t in FITM_TEMPERATURES])}"
        k_range_str = f"k{min(N_DEMOS_LIST)}-{max(N_DEMOS_LIST)}" if len(N_DEMOS_LIST)>1 else f"k{N_DEMOS_LIST[0]}" if N_DEMOS_LIST else "k_none"

        results_filename_base = f"results_{dataset_file_part}_{safe_model_name}_{methods_str}_{k_range_str}_n{NUM_EVAL_SAMPLES}"
        results_filename = os.path.join(RESULTS_DIR, results_filename_base + ".txt")

        with open(results_filename, 'w', encoding='utf-8') as f:
            for line in summary_lines:
                f.write(line + "\n")
        print(f"Results summary saved to: {results_filename}")

    except Exception as e:
        print(f"Error saving results summary: {e}")
        traceback.print_exc()


    # --- MODIFIED Plotting ---
    print("\n--- Generating Plot ---")
    plot_generated = False
    has_data = any(not np.isnan(results[k].get('ICL', np.nan)) or \
                   (args.run_focus and any(not np.isnan(results[k].get(f"FOCUS_p{p}", np.nan)) for p in P_THRESHOLDS)) or \
                   (args.run_fitm and any(not np.isnan(results[k].get(f"FITM_t{t:.0e}", np.nan)) for t in FITM_TEMPERATURES))
                   for k in N_DEMOS_LIST if k in results)


    if has_data:
        try:
            x_values = sorted(results.keys())
            plt.figure(figsize=(12, 7))

            # Plot Standard ICL
            icl_accuracies = [results[k].get('ICL', np.nan) for k in x_values]
            valid_icl = [(x, y) for x, y in zip(x_values, icl_accuracies) if not np.isnan(y)]
            if valid_icl: plt.plot([p[0] for p in valid_icl], [p[1] for p in valid_icl], marker='o', linestyle='-', label='Standard ICL', linewidth=2)

            markers = ['s', '^', 'D', 'v', '<', '>']
            linestyles = ['--', ':', '-.', '--', ':', '-.']

            # Plot FOCUS ICL
            if args.run_focus:
                for i, p_val in enumerate(P_THRESHOLDS):
                    focus_key = f"FOCUS_p{p_val}"
                    focus_acc = [results[k].get(focus_key, np.nan) for k in x_values]
                    valid_focus = [(x, y) for x, y in zip(x_values, focus_acc) if not np.isnan(y)]
                    if valid_focus: plt.plot([p[0] for p in valid_focus], [p[1] for p in valid_focus], marker=markers[i % len(markers)], linestyle=linestyles[i % len(linestyles)], label=f'FOCUS (p={p_val})')

            # Plot FITM ICL
            if args.run_fitm:
                # Use different markers/styles for FITM
                fitm_markers = ['X', 'P', '*', 'h', '+', 'd']
                fitm_linestyles = ['-', '-', '-', '-', '-', '-'] # Maybe solid lines for FITM
                fitm_colors = plt.cm.viridis(np.linspace(0, 0.8, len(FITM_TEMPERATURES))) # Use a colormap

                for i, temp_val in enumerate(FITM_TEMPERATURES):
                    fitm_key = f"FITM_t{temp_val:.0e}"
                    fitm_acc = [results[k].get(fitm_key, np.nan) for k in x_values]
                    valid_fitm = [(x, y) for x, y in zip(x_values, fitm_acc) if not np.isnan(y)]
                    if valid_fitm: plt.plot([p[0] for p in valid_fitm], [p[1] for p in valid_fitm], marker=fitm_markers[i % len(fitm_markers)], linestyle=fitm_linestyles[i % len(fitm_linestyles)], color=fitm_colors[i], label=f'FITM (t={temp_val:.1e})')


            plt.xlabel("Number of Demonstrations (k)")
            plt.ylabel(f"Accuracy on {dataset_print_name}")
            plt.title(f"Performance vs. Demonstrations ({dataset_print_name})\nModel: {MODEL_NAME.split('/')[-1]}, Samples/Run: {NUM_EVAL_SAMPLES}")
            plt.xticks(N_DEMOS_LIST if N_DEMOS_LIST else [0])
            # Calculate y-limits based on valid data
            all_acc = []
            for k in x_values:
                 if k in results:
                     all_acc.extend([v for v in results[k].values() if isinstance(v, (float, int)) and not np.isnan(v)])
            min_y = min(all_acc) if all_acc else 0.0
            max_y = max(all_acc) if all_acc else 1.0
            plt.ylim(bottom=max(0.0, min_y - 0.05), top=min(1.0, max_y + 0.05))

            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left") # Place legend outside
            plt.grid(True, linestyle='--', linewidth=0.5)
            plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend

            # Save Plot
            plot_filename_base = f"performance_{dataset_file_part}_{safe_model_name}_{methods_str}_{k_range_str}_n{NUM_EVAL_SAMPLES}"
            pdf_filename = os.path.join(FIGURES_DIR, plot_filename_base + ".pdf")
            jpg_filename = os.path.join(FIGURES_DIR, plot_filename_base + ".jpg")
            plt.savefig(pdf_filename); plt.savefig(jpg_filename, dpi=300)
            print(f"Plot saved as '{pdf_filename}' and '{jpg_filename}'")
            plot_generated = True
            plt.close()

        except Exception as e:
            print(f"Error during plotting: {e}")
            traceback.print_exc()
    else:
         print("\n--- Skipping Plot Generation (No valid results to plot) ---")

    print("\n--- Script Finished ---")