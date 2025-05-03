import random
import json
import time
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset 
from tqdm import tqdm
import gc
import traceback
import os

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"

# <<< Dataset Configurations >>>
SOURCE_DATASET_1_NAME = "allenai/c4"
SOURCE_DATASET_1_CONFIG = "en"
SOURCE_DATASET_1_SPLIT = "train"

SOURCE_DATASET_2_NAME = "wikimedia/wikipedia"
SOURCE_DATASET_2_CONFIG = "20231101.en"
SOURCE_DATASET_2_SPLIT = "train"
# <<<------------------------>>>

SEGMENT_TARGET_TOKENS = 128
MIN_CONTEXT_TOKENS = 4000
MAX_CONTEXT_TOKENS = 32000
OUTPUT_FILE = "in2_training.jsonl" 
SAVE_INTERVAL = 50

# Data Mixture Ratios
RATIO_FINE_GRAINED = 0.6
RATIO_INTEGRATION = 0.2
RATIO_SHORT_CONTEXT = 0.1
RATIO_GENERAL_INSTRUCTION = 0.1

# Generation Parameters
MAX_NEW_TOKENS_QA = 350
TEMPERATURE = 0.7
TOP_P = 0.9

MAX_RETRIES = 3
RETRY_DELAY = 5

# --- Helper Functions ---

def load_source_corpus(total_limit=None, shuffle_seed=None, shuffle_buffer_size=10000):
    corpus_combined = []

    if total_limit is None:
        print("Warning: No total_limit specified for combined dataset loading. This might consume excessive RAM. Please set a 'total_limit'.")
        total_limit = 10000 
        print(f"Using safety limit: {total_limit}")

    limit_ds1 = total_limit // 2
    limit_ds2 = total_limit - limit_ds1 

    print(f"Attempting to load approx {limit_ds1} from {SOURCE_DATASET_1_NAME} and {limit_ds2} from {SOURCE_DATASET_2_NAME}")
    print(f"Using stream shuffle buffer size: {shuffle_buffer_size}. This may take a moment.")

    seed1 = shuffle_seed if shuffle_seed is not None else random.randint(0, 2**32 - 1)
    seed2 = (shuffle_seed + 1 if shuffle_seed is not None else random.randint(0, 2**32 - 1))
    print(f"Using shuffle seed {seed1} for {SOURCE_DATASET_1_NAME}")
    print(f"Using shuffle seed {seed2} for {SOURCE_DATASET_2_NAME}")


    # --- Load Dataset 1 (C4) ---
    print(f"Loading and shuffling dataset 1: {SOURCE_DATASET_1_NAME} ({SOURCE_DATASET_1_CONFIG})")
    corpus_ds1 = []
    try:
        dataset1 = load_dataset(
            SOURCE_DATASET_1_NAME,
            SOURCE_DATASET_1_CONFIG,
            split=SOURCE_DATASET_1_SPLIT,
            streaming=True
        )
        # <<< Shuffle the stream BEFORE taking elements >>>
        shuffled_dataset1 = dataset1.shuffle(seed=seed1, buffer_size=shuffle_buffer_size)
        dataset1_limited = shuffled_dataset1.take(limit_ds1)
        # <<< ------------------------------------------ >>>
        print(f"Collecting {limit_ds1} text data from shuffled {SOURCE_DATASET_1_NAME} into memory list...")
        for item in tqdm(dataset1_limited, desc=f"Loading {SOURCE_DATASET_1_NAME}", total=limit_ds1):
            if 'text' in item and item['text'].strip():
                corpus_ds1.append(item['text'])
        print(f"Collected {len(corpus_ds1)} documents from {SOURCE_DATASET_1_NAME}.")
        corpus_combined.extend(corpus_ds1)
        # Make sure to delete the shuffled dataset reference too
        del dataset1, shuffled_dataset1, dataset1_limited, corpus_ds1
        gc.collect()
    except Exception as e:
        print(f"Error loading dataset '{SOURCE_DATASET_1_NAME}': {e}")
        traceback.print_exc() # Print traceback for debugging

    # --- Load Dataset 2 (Wikipedia) ---
    actual_limit_ds2 = total_limit - len(corpus_combined)
    if actual_limit_ds2 <= 0 :
         print("Warning: Reached total limit after loading dataset 1. Skipping dataset 2.")
    else:
        print(f"Loading and shuffling dataset 2: {SOURCE_DATASET_2_NAME} ({SOURCE_DATASET_2_CONFIG})")
        corpus_ds2 = []
        try:
            dataset2 = load_dataset(
                SOURCE_DATASET_2_NAME,
                SOURCE_DATASET_2_CONFIG,
                split=SOURCE_DATASET_2_SPLIT,
                streaming=True
            )
            # <<< Shuffle the stream BEFORE taking elements >>>
            shuffled_dataset2 = dataset2.shuffle(seed=seed2, buffer_size=shuffle_buffer_size)
            dataset2_limited = shuffled_dataset2.take(actual_limit_ds2)
            # <<< ------------------------------------------ >>>
            print(f"Collecting {actual_limit_ds2} text data from shuffled {SOURCE_DATASET_2_NAME} into memory list...")
            for item in tqdm(dataset2_limited, desc=f"Loading {SOURCE_DATASET_2_NAME}", total=actual_limit_ds2):
                if 'text' in item and item['text'].strip():
                    text_content = item['text']
                    corpus_ds2.append(text_content)

            print(f"Collected {len(corpus_ds2)} documents from {SOURCE_DATASET_2_NAME}.")
            corpus_combined.extend(corpus_ds2)
            del dataset2, shuffled_dataset2, dataset2_limited, corpus_ds2
            gc.collect()
        except Exception as e:
            print(f"Error loading dataset '{SOURCE_DATASET_2_NAME}': {e}")
            traceback.print_exc() 

    # --- Shuffle the combined *in-memory* corpus ---
    if corpus_combined:
        print(f"Total documents collected: {len(corpus_combined)}. Shuffling in-memory list...")
        random.shuffle(corpus_combined)
        print("Combined in-memory corpus shuffled.")
    else:
        print("Warning: No documents collected from any source.")

    return corpus_combined
# <<<------------------------------------------------------>>>


def load_evaluation_data():
    # Return an empty set for this example, replace with your actual data loading
    print("Warning: Using empty evaluation data.")
    return set()

def check_contamination(text_ci, eval_ngrams_set):
    if not isinstance(text_ci, str): 
        return False
    return False

def load_general_instruction_data(limit=None):
    print("Loading placeholder general instruction data...")
    data = [{"instruction": f"Placeholder instruction {i}", "output": f"Placeholder output {i}"} for i in range(500)]
    if limit is not None and limit < len(data):
        return data[:limit]
    return data

def split_text_into_segments(text, tokenizer, target_tokens=SEGMENT_TARGET_TOKENS):
    if not text or not isinstance(text, str): return [] 
    try:
        tokens = tokenizer.encode(text, add_special_tokens=False)
    except Exception as e:
        print(f"Warning: Error tokenizing text segment, skipping. Error: {e}")
        return []
    segments = []
    start_idx = 0
    while start_idx < len(tokens):
        end_idx = min(start_idx + target_tokens, len(tokens))
        segment_tokens = tokens[start_idx:end_idx]
        if not segment_tokens:
            break
        try:
            segment_text = tokenizer.decode(segment_tokens, skip_special_tokens=True)
            if segment_text.strip():
                segments.append({"text": segment_text, "tokens": segment_tokens})
        except Exception as e:
            print(f"Warning: Error decoding segment tokens, skipping. Error: {e}")
        start_idx = end_idx
    return segments


def get_random_segments(all_segments_flat, num_segments, exclude_indices=None):
    if num_segments <= 0 or not all_segments_flat: return []
    if exclude_indices is None: exclude_indices = set()

    available_indices = [i for i in range(len(all_segments_flat)) if i not in exclude_indices]

    if not available_indices:
        print("Warning: No available indices for get_random_segments after exclusion.")
        return []

    if len(available_indices) < num_segments:
        chosen_indices = random.choices(available_indices, k=num_segments)
    else:
        chosen_indices = random.sample(available_indices, num_segments)

    return [all_segments_flat[i] for i in chosen_indices]


def create_generation_prompt(template_example_num, context_segments):
    # Ensure context_segments is not empty and contains dicts with 'text'
    if not context_segments or not isinstance(context_segments, list):
        print("Warning: create_generation_prompt received invalid context_segments")
        return ""
    if not all(isinstance(seg, dict) and 'text' in seg for seg in context_segments):
         print("Warning: create_generation_prompt received context_segments with invalid items")
         return ""

    base_instruction = """Generate one question and the corresponding answer strictly based on the provided context. The question should require understanding the information within the context, not just simple extraction. The answer should be concise and directly address the question using only information from the context. Do not add any information not present in the context. Output the question prefixed with "Question:" and the answer prefixed with "Answer:", separated by a newline."""

    if template_example_num == 7: # Fine-grained
        if not context_segments: return ""
        context_text = context_segments[0]['text'].strip()
        if not context_text: return ""
        instruction = f"""{base_instruction}

### Context ###:
{context_text}

Question:
Answer:"""
        return instruction.strip() 

    elif template_example_num == 8: 
        if len(context_segments) < 2: return ""
        context_str = ""
        for i, seg in enumerate(context_segments):
            seg_text = seg['text'].strip()
            if seg_text: # Only include non-empty segments
                context_str += f"--- Piece {i+1} ---\n{seg_text}\n\n"
        if not context_str.strip(): return "" 
        instruction = f"""{base_instruction}

### Context Pieces ###:
{context_str.strip()}

Question:
Answer:"""
        return instruction.strip() 
    else:
        print(f"Warning: Unsupported template_example_num: {template_example_num}")
        return ""


@torch.no_grad()
def call_qwen_for_qa(model, tokenizer, instruction_prompt, device, data_type_debug="unknown"):
    if not instruction_prompt or not isinstance(instruction_prompt, str):
        print("Warning: call_qwen_for_qa received empty or invalid instruction_prompt")
        return None

    messages = [{"role": "user", "content": instruction_prompt}]
    try:
        prompt_formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        if not prompt_formatted:
            print("Warning: tokenizer.apply_chat_template returned empty string.")
            return None
        inputs = tokenizer(prompt_formatted, return_tensors="pt").to(device)
    except Exception as e:
        print(f"Error during tokenization or template application: {e}")
        traceback.print_exc()
        return None

    output_sequences = None
    for attempt in range(MAX_RETRIES):
        try:
            output_sequences = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS_QA,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            break
        except torch.cuda.OutOfMemoryError as oom_err:
            print(f"OOM Error during generation (Attempt {attempt + 1}/{MAX_RETRIES}): {oom_err}. Clearing cache and retrying...")
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"Generation Error (Attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            traceback.print_exc()
            time.sleep(RETRY_DELAY)

        if attempt == MAX_RETRIES - 1:
            print("Max retries reached for model generation. Skipping this example.")
            return None 

    if output_sequences is None: 
        return None

    try:
        input_token_len = inputs["input_ids"].shape[1]
        generated_tokens = output_sequences[0][input_token_len:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    except Exception as e:
        print(f"Error decoding generated tokens: {e}")
        traceback.print_exc()
        return None

    try:
        q_marker = "Question:"
        a_marker = "Answer:"

        q_start_idx = generated_text.find(q_marker)
        a_start_idx = generated_text.find(a_marker)

        if q_start_idx == -1 or a_start_idx == -1:
             lines = generated_text.split('\n')
             lines = [line.strip() for line in lines if line.strip()]
             if len(lines) >= 2: 
                 question = lines[0]
                 answer = "\n".join(lines[1:])
                 if question and answer:
                      # print("Debug: Parsed using fallback line splitting.")
                      return {"question": question, "answer": answer}
             return None

        # Ensure Answer comes after Question
        if a_start_idx <= q_start_idx:
            a_start_idx = generated_text.find(a_marker, q_start_idx + len(q_marker))
            if a_start_idx == -1:
                # print(f"Debug: Answer marker not found after question marker. Raw: {generated_text}")
                return None 

        question = generated_text[q_start_idx + len(q_marker):a_start_idx].strip()

        answer_part = generated_text[a_start_idx + len(a_marker):].strip()
        answer_lines = [line.strip() for line in answer_part.split('\n') if line.strip()]

        answer = "\n".join(answer_lines)

        if not question or not answer:
            # print(f"Debug: Parsed empty question or answer. Raw: {generated_text}")
            return None

        if "Question:" in answer or "Answer:" in question:
             # print(f"Debug: Found markers within parsed Q/A, likely parse error. Raw: {generated_text}")
             return None 

        return {"question": question, "answer": answer}

    except Exception as e:
        print(f"Error parsing generated text: {e}")
        print(f"Problematic Raw Text:\n---\n{generated_text}\n---")
        traceback.print_exc()
        return None


# --- Main Data Generation Logic ---

def main():
    # --- Device Setup & Model/Tokenizer Loading ---
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA available. Using device: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        reserved_mem = torch.cuda.memory_reserved(0) / (1024**3)
        allocated_mem = torch.cuda.memory_allocated(0) / (1024**3)
        print(f"GPU Memory: Total={total_mem:.2f} GB, Reserved={reserved_mem:.2f} GB, Allocated={allocated_mem:.2f} GB")
    else:
        device = torch.device("cpu")
        print("Warning: CUDA not available. Using CPU (this will be very slow).")

    print(f"Loading Tokenizer: {MODEL_NAME}")
    try:
        # Ensure trust_remote_code=True is necessary and accepted for Qwen2.5
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=True)
        if tokenizer.pad_token is None:
            print("Tokenizer does not have a pad token. Setting pad_token = eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Fatal Error loading tokenizer: {e}")
        traceback.print_exc()
        return

    print(f"Loading Model: {MODEL_NAME} with 4-bit quantization")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto", 
            trust_remote_code=True,
        )
        model.eval() 
        print("Model loaded successfully.")
        if torch.cuda.is_available():
            reserved_mem = torch.cuda.memory_reserved(0) / (1024**3)
            allocated_mem = torch.cuda.memory_allocated(0) / (1024**3)
            print(f"GPU Memory after model load: Reserved={reserved_mem:.2f} GB, Allocated={allocated_mem:.2f} GB")

    except Exception as e:
        print(f"Fatal Error loading model: {e}")
        traceback.print_exc()
        return


    # --- Load Combined Source Corpus ---
    total_document_limit = 20000 
    current_shuffle_seed = int(time.time())
    print(f"Using shuffle seed for this run: {current_shuffle_seed}")
    source_corpus = load_source_corpus(
        total_limit=total_document_limit,
        shuffle_seed=current_shuffle_seed, 
        shuffle_buffer_size=10000 
    )
    # <<< ------------------------------------ >>>
    if not source_corpus:
        print("Error: Combined source corpus is empty after loading attempt. Exiting.")
        return

    # --- Load Other Data ---
    eval_ngrams = load_evaluation_data() 
    general_instructions = load_general_instruction_data() 
    if not general_instructions:
        print("Warning: No general instructions loaded. Ratio for general instructions will be 0.")
        RATIO_GENERAL_INSTRUCTION = 0

    # --- Pre-process Corpus (Segments all loaded docs) ---
    print(f"Pre-processing: Splitting {len(source_corpus)} loaded source documents into segments...")
    all_segments_flat = []
    original_doc_indices = {} 
    current_flat_index = 0
    doc_iterator = tqdm(enumerate(source_corpus), total=len(source_corpus), desc="Segmenting Corpus")
    for i, doc in doc_iterator:
        if not isinstance(doc, str) or not doc.strip():
            # print(f"Warning: Skipping empty or non-string document at index {i}.")
            continue 

        segments = split_text_into_segments(doc, tokenizer, target_tokens=SEGMENT_TARGET_TOKENS)

        if segments:
             doc_segment_indices = list(range(current_flat_index, current_flat_index + len(segments)))
             original_doc_indices[i] = doc_segment_indices
             all_segments_flat.extend(segments)
             current_flat_index += len(segments)
        # else:
            # print(f"Warning: Document at index {i} produced no segments.")


    print(f"Total segments created: {len(all_segments_flat)}")
    if not all_segments_flat:
        print("Error: No segments were created from the loaded corpus. Cannot proceed.")
        return

    # --- Generate Training Data Loop ---
    generated_data_accumulator = []
    num_target_examples = len(original_doc_indices) 
    print(f"Targeting generation based on {num_target_examples} documents that yielded segments.")

    processed_docs_count = 0 
    valid_doc_indices = list(original_doc_indices.keys())
    random.shuffle(valid_doc_indices) 

    # Initialize counts and calculate targets based on num_target_examples
    counts = {"fine": 0, "integ": 0, "short": 0, "general": 0, "skipped_contam": 0, "skipped_nosegs": 0, "error_gen": 0}
    if num_target_examples > 0:
         total_ratio = RATIO_FINE_GRAINED + RATIO_INTEGRATION + RATIO_SHORT_CONTEXT + RATIO_GENERAL_INSTRUCTION
         if not math.isclose(total_ratio, 1.0):
             print(f"Warning: Sum of ratios ({total_ratio}) is not 1.0. Adjusting proportionally.")
             scale_factor = 1.0 / total_ratio if total_ratio > 0 else 0
             RATIO_FINE_GRAINED *= scale_factor
             RATIO_INTEGRATION *= scale_factor
             RATIO_SHORT_CONTEXT *= scale_factor
             RATIO_GENERAL_INSTRUCTION *= scale_factor

         target_counts = {
             "fine": int(num_target_examples * RATIO_FINE_GRAINED),
             "integ": int(num_target_examples * RATIO_INTEGRATION),
             "short": int(num_target_examples * RATIO_SHORT_CONTEXT),
             "general": int(num_target_examples * RATIO_GENERAL_INSTRUCTION)
         }
         current_total_targets = sum(target_counts.values())
         diff = num_target_examples - current_total_targets
         if diff != 0 and "fine" in target_counts:
             target_counts["fine"] += diff

         target_counts["general"] = min(target_counts["general"], len(general_instructions))
         if target_counts["general"] < int(num_target_examples * RATIO_GENERAL_INSTRUCTION):
              print(f"Warning: Available general instructions ({len(general_instructions)}) is less than target ({int(num_target_examples * RATIO_GENERAL_INSTRUCTION)}). Capping.")

    else: 
         target_counts = {"fine": 0, "integ": 0, "short": 0, "general": 0}

    general_instr_iter = iter(random.sample(general_instructions, len(general_instructions))) # Shuffle general instructions too


    print(f"Target counts: {target_counts}")
    if os.path.exists(OUTPUT_FILE):
         print(f"Warning: Output file '{OUTPUT_FILE}' already exists. Appending data.")
        
    pbar = tqdm(total=num_target_examples, desc="Generating IN2 Data")

    # <<< Main Loop Logic (Identical to V3 - processes the combined source_corpus) >>>
    while processed_docs_count < num_target_examples and valid_doc_indices:
        doc_id = valid_doc_indices.pop(0)

        try:
            ci = source_corpus[doc_id] 
            segment_indices_for_doc = original_doc_indices[doc_id] 
            segments_from_ci = [all_segments_flat[idx] for idx in segment_indices_for_doc]
        except (IndexError, KeyError) as e:
            print(f"Warning: Error retrieving document or segments for doc_id {doc_id}. Skipping. Error: {e}")
            counts["error_gen"] += 1 
            continue 

        if not segments_from_ci:
             print(f"Warning: No segments found for doc_id {doc_id} despite being in original_doc_indices. Skipping.")
             counts["skipped_nosegs"] += 1
             continue

        if check_contamination(ci, eval_ngrams):
            counts["skipped_contam"] += 1
            pbar.update(1) 
            processed_docs_count += 1 
            continue

        # --- Choose data type based on remaining targets ---
        data_type = None
        possible_types = []

        if counts["general"] < target_counts["general"]: possible_types.append("general")
        if counts["integ"] < target_counts["integ"] and len(segments_from_ci) >= 2: possible_types.append("integ")
        if counts["short"] < target_counts["short"]: possible_types.append("short")
        if counts["fine"] < target_counts["fine"]: possible_types.append("fine")

        if possible_types:
            data_type = random.choice(possible_types)
        else:
            if counts["fine"] < target_counts["fine"]:
                 data_type = "fine"
            else:
                 pbar.update(1) 
                 processed_docs_count += 1
                 continue

        # --- Generate QA based on chosen data_type ---
        qa_result = None
        final_context_Li = None 
        instruction_prompt = "" 

        try: 
            if data_type == "fine":
                 target_segment = random.choice(segments_from_ci)
                 instruction_prompt = create_generation_prompt(7, [target_segment]) 

                 if instruction_prompt:
                     qa_result = call_qwen_for_qa(model, tokenizer, instruction_prompt, device, "fine")
                     if qa_result:
                         target_tokens = random.randint(MIN_CONTEXT_TOKENS, MAX_CONTEXT_TOKENS)
                         num_total_segments = max(1, target_tokens // SEGMENT_TARGET_TOKENS)
                         num_distractor_segments = num_total_segments - 1

                         target_seg_global_idx = -1
                         try:
                             target_seg_global_idx = all_segments_flat.index(target_segment)
                         except ValueError:
                              print(f"Warning: Could not find global index for target segment from doc {doc_id}. Cannot guarantee exclusion.")
                              qa_result = None 
                              counts["error_gen"] += 1

                         if qa_result: 
                             exclude_set = {target_seg_global_idx} if target_seg_global_idx != -1 else set()
                             distractor_segments = get_random_segments(all_segments_flat, num_distractor_segments, exclude_indices=exclude_set)

                             context_pool = [target_segment] + distractor_segments
                             random.shuffle(context_pool)
                             final_context_Li = " ".join([s['text'] for s in context_pool]).strip()

                             if final_context_Li:
                                 counts["fine"] += 1
                             else: 
                                 qa_result = None
                                 counts["error_gen"] += 1

            elif data_type == "integ":
                 num_segments_to_integrate = random.randint(2, min(5, len(segments_from_ci))) 
                 target_segments = random.sample(segments_from_ci, num_segments_to_integrate)
                 instruction_prompt = create_generation_prompt(8, target_segments) 

                 if instruction_prompt:
                     qa_result = call_qwen_for_qa(model, tokenizer, instruction_prompt, device, "integ")
                     if qa_result:
                         target_tokens = random.randint(MIN_CONTEXT_TOKENS, MAX_CONTEXT_TOKENS)
                         num_total_segments = max(1, target_tokens // SEGMENT_TARGET_TOKENS)
                         num_distractor_segments = max(0, num_total_segments - len(target_segments))

                         target_seg_global_indices = set()
                         valid_indices = True
                         for ts in target_segments:
                             try:
                                 target_seg_global_indices.add(all_segments_flat.index(ts))
                             except ValueError:
                                  print(f"Warning: Could not find global index for integration target segment from doc {doc_id}. Cannot guarantee exclusion.")
                                  valid_indices = False
                                  break
                         if not valid_indices:
                              qa_result = None # Skip this QA pair
                              counts["error_gen"] += 1

                         if qa_result: 
                             distractor_segments = get_random_segments(all_segments_flat, num_distractor_segments, exclude_indices=target_seg_global_indices)
                             context_pool = target_segments + distractor_segments
                             random.shuffle(context_pool)
                             final_context_Li = " ".join([s['text'] for s in context_pool]).strip()

                             if final_context_Li:
                                 counts["integ"] += 1
                             else:
                                 qa_result = None
                                 counts["error_gen"] += 1

            elif data_type == "short":
                  short_context_text = ci 
                  instruction_prompt = create_generation_prompt(7, [{"text": short_context_text, "tokens": []}]) 

                  if instruction_prompt:
                      qa_result = call_qwen_for_qa(model, tokenizer, instruction_prompt, device, "short")
                      if qa_result:
                          final_context_Li = short_context_text.strip()
                          if final_context_Li:
                              counts["short"] += 1
                          else: 
                              qa_result = None
                              counts["error_gen"] += 1

            elif data_type == "general":
                  try:
                      instr_data = next(general_instr_iter)
                      if 'instruction' in instr_data and 'output' in instr_data and instr_data['instruction'] and instr_data['output']:
                           qa_result = {"question": instr_data['instruction'], "answer": instr_data['output']}
                           final_context_Li = "" 
                           counts["general"] += 1
                      else:
                           print(f"Warning: Invalid general instruction data format: {instr_data}. Skipping.")
                           counts["error_gen"] += 1
                           qa_result = None 
                  except StopIteration:
                      print("Warning: Ran out of general instructions unexpectedly.")
                      target_counts["general"] = counts["general"] 
                      qa_result = None 
                  except Exception as e:
                       print(f"Error processing general instruction: {e}")
                       traceback.print_exc()
                       qa_result = None
                       counts["error_gen"] += 1

        except Exception as gen_err:
             print(f"!! Unhandled error during generation logic for doc {doc_id}, type {data_type}: {gen_err}")
             traceback.print_exc()
             counts["error_gen"] += 1
             qa_result = None 

        # --- Store Successful Results ---
        if qa_result and final_context_Li is not None: 
            if data_type != "general" and not final_context_Li.strip():
                 print(f"Warning: Generated empty or whitespace-only context for non-general type '{data_type}'. Skipping.")
                 counts["error_gen"] += 1
            else:
                 output_record = {
                     "context": final_context_Li,
                     "instruction": qa_result['question'],
                     "output": qa_result['answer'],
                     "data_type": data_type 
                 }
                 generated_data_accumulator.append(output_record)

                 # --- Periodic Save ---
                 if len(generated_data_accumulator) >= SAVE_INTERVAL:
                     try:
                         with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                             for record in generated_data_accumulator:
                                 f.write(json.dumps(record) + '\n')
                         # print(f"Successfully saved {len(generated_data_accumulator)} records.")
                         generated_data_accumulator = [] 
                     except IOError as io_err:
                          print(f"\n!! Error saving intermediate results to {OUTPUT_FILE}: {io_err}")
                     except Exception as write_err:
                          print(f"\n!! Unexpected error during intermediate save: {write_err}")
                          traceback.print_exc()

        elif not qa_result and data_type != "general":
             if data_type in ["fine", "integ", "short"]: 
                 counts["error_gen"] += 1

        # --- Update Progress Bar and Doc Counter ---
        pbar.update(1)
        processed_docs_count += 1

        # --- Periodic Cleanup ---
        if processed_docs_count % 50 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- End of Loop ---
    pbar.close()

    # --- Final Save ---
    if generated_data_accumulator: 
        print(f"\nSaving final {len(generated_data_accumulator)} records...")
        try:
            with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                for record in generated_data_accumulator:
                    f.write(json.dumps(record) + '\n')
            print(f"Successfully saved final {len(generated_data_accumulator)} records.")
        except IOError as io_err:
            print(f"\n!! Error saving final results to {OUTPUT_FILE}: {io_err}")
        except Exception as write_err:
            print(f"\n!! Unexpected error during final save: {write_err}")
            traceback.print_exc()

    # --- Final Summary ---
    print(f"\n--- Generation Summary ---")
    print(f"Target Docs for Generation (based on valid segments): {num_target_examples}")
    print(f"Total Docs Processed in Loop: {processed_docs_count}") # How many docs from valid_doc_indices were handled
    print(f"Source Documents Limit Configured: {total_document_limit}")
    print(f"Actual Source Documents Loaded: {len(source_corpus)}")
    print(f"Documents Yielding Segments: {len(original_doc_indices)}")
    print("-" * 20)
    print(f"Data Type Targets:")
    print(f"  Fine-grained: {target_counts.get('fine', 0)}")
    print(f"  Integration:  {target_counts.get('integ', 0)}")
    print(f"  Short Context:{target_counts.get('short', 0)}")
    print(f"  General Inst:{target_counts.get('general', 0)}")
    print("-" * 20)
    print(f"Actual Generated Counts:")
    print(f"  Fine-grained: {counts['fine']}")
    print(f"  Integration:  {counts['integ']}")
    print(f"  Short Context:{counts['short']}")
    print(f"  General Inst:{counts['general']}")
    total_generated = counts['fine'] + counts['integ'] + counts['short'] + counts['general']
    print(f"  Total Successful Records Generated: {total_generated}")
    print("-" * 20)
    print(f"Skipped / Errors:")
    print(f"  Skipped (Contamination): {counts['skipped_contam']}")
    print(f"  Skipped (No Segments - initial filtering): {len(source_corpus) - len(original_doc_indices)}") 
    print(f"  Errors (Generation/Parsing/Other): {counts['error_gen']}")
    print("-" * 20)

    try:
        if os.path.exists(OUTPUT_FILE):
             with open(OUTPUT_FILE, 'r', encoding='utf-8') as f: lines = len(f.readlines())
             print(f"Total lines now in output file '{OUTPUT_FILE}': {lines}")
        else:
             print(f"Output file '{OUTPUT_FILE}' not found or not created.")
    except Exception as e:
        print(f"Could not verify output file line count: {e}")

    print("Dataset construction script finished.")


if __name__ == "__main__":
    print("Ensure required packages are installed: pip install torch transformers accelerate sentencepiece bitsandbytes tqdm datasets")
    print("--------------------------------------------------------------------------")
    print(f"Output will be written/appended to: {OUTPUT_FILE}")
    main()