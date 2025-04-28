# --- START OF FILE attention_visual_repeated_shuffle_lines.py ---

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm.auto import tqdm
import random
import gc  
import os
import matplotlib.colors as mcolors 

# --- Configuration ---
TARGET_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
NUM_DOCUMENTS = 30  
NUM_REPETITIONS = 10 

DATASET_NAME = "google-research-datasets/nq_open" 
NUM_EXAMPLES_TO_RUN = 100 
MAX_TOKENS_PER_DOC = 3072 
MAX_CONTEXT_LENGTH = 32768 

# --- Plotting Style Configuration ---
model_plot_styles = {
    "Qwen/Qwen2.5-7B-Instruct": ['#252A34', '#252A34', '-', 'o'], 
}
fontsize = 10
output_filename = f"qwen7b_attention_k{NUM_DOCUMENTS}_rep{NUM_REPETITIONS}_shuffled_lines.jpg"

# --- Helper Function ---
def hex_to_rgba(hex_color, alpha=0.2): 
    return mcolors.to_rgba(hex_color, alpha=alpha)

# --- Device Setup ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    dtype = torch.bfloat16
else:
    device = torch.device("cpu")
    print("Using CPU (will be very slow)")
    dtype = torch.float32

# --- Load and Prepare Dataset (Load Once) ---
print(f"Loading dataset: {DATASET_NAME}")
max_k = NUM_DOCUMENTS 
buffer_needed = int(NUM_EXAMPLES_TO_RUN * 1.2 + max_k) 

try:
    nq_dataset = load_dataset(DATASET_NAME, split='train', streaming=True)
    print(f"Buffering {buffer_needed} examples from NQ dataset...")
    nq_dataset_buffered = list(nq_dataset.take(buffer_needed))
    print(f"Buffered {len(nq_dataset_buffered)} examples.")
    if len(nq_dataset_buffered) < NUM_EXAMPLES_TO_RUN + max_k:
        print(f"Error: Dataset buffer ({len(nq_dataset_buffered)}) is smaller than needed ({NUM_EXAMPLES_TO_RUN + max_k}). Cannot guarantee unique distractors. Exiting.")
        exit()

except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Please ensure you have internet connection and the 'datasets' library installed.")
    exit()

def get_document_text(example):
    if not example: return "Invalid Example"
    answer_candidates = example.get('answer', [])
    answer_text = answer_candidates[0] if answer_candidates else "No Answer Found"
    return f"Question: {example.get('question', 'No Question')} Answer: {answer_text}"

# --- Pre-select Examples (Once) ---
print(f"Selecting {NUM_EXAMPLES_TO_RUN} base examples...")
primary_examples_data = []
example_indices_used = set()
available_indices = list(range(len(nq_dataset_buffered)))
random.shuffle(available_indices)
processed_count = 0

while processed_count < NUM_EXAMPLES_TO_RUN and available_indices:
    idx = available_indices.pop()
    example = nq_dataset_buffered[idx]
    if example and example.get('question'):
        primary_examples_data.append({'index': idx, 'example': example})
        example_indices_used.add(idx)
        processed_count += 1

if processed_count < NUM_EXAMPLES_TO_RUN:
    print(f"Error: Only found {processed_count} valid examples, needed {NUM_EXAMPLES_TO_RUN}. Exiting.")
    exit()

print(f"Selected {NUM_EXAMPLES_TO_RUN} examples.")
available_distractor_indices = [i for i in range(len(nq_dataset_buffered)) if i not in example_indices_used]
print(f"Available distractor indices: {len(available_distractor_indices)}")

if len(available_distractor_indices) < NUM_DOCUMENTS - 1:
    print(f"Warning: Not enough unique distractors ({len(available_distractor_indices)}) available for K={NUM_DOCUMENTS}. Will reuse distractors.")


# --- Load Model (Once) ---
print(f"\n--- Loading Model: {TARGET_MODEL_ID} ---")
model_id = TARGET_MODEL_ID
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager" # Force eager attention implementation
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    print("Model and tokenizer loaded.")

except Exception as e:
    print(f"ERROR loading model {model_id}: {e}. Cannot proceed. Exiting.")
    exit()


# --- Experiment Loop (Over Repetitions) ---
# Store the MEAN attention profile (1D array) for each repetition
all_repetition_means = []

for rep in range(NUM_REPETITIONS):
    print(f"\n===== Running Repetition {rep+1}/{NUM_REPETITIONS} for K={NUM_DOCUMENTS} =====")

    model_attention_profiles = [] # Store attention profiles (NxK matrix) for this repetition's examples
    pbar = tqdm(primary_examples_data, desc=f"Rep {rep+1}/{NUM_REPETITIONS}", leave=False)

    for primary_data in pbar:
        example_idx = primary_data['index']
        example = primary_data['example']
        question = example['question']
        gold_doc_text = get_document_text(example)

        # --- Sample Distractor Documents ---
        distractor_docs_text = []
        current_distractor_pool = list(available_distractor_indices) # Use the pre-selected pool
        random.shuffle(current_distractor_pool)
        distractor_count = 0
        
        selected_indices = []
        needed = NUM_DOCUMENTS - 1
        unique_pool = [i for i in current_distractor_pool if i != example_idx]

        if len(unique_pool) >= needed:
            selected_indices = random.sample(unique_pool, needed)
        else:
            # print(f"Warning: Reusing distractors for example {example_idx} in rep {rep+1}")
            selected_indices = unique_pool
            remaining_needed = needed - len(unique_pool)
            reuse_pool = [i for i in available_distractor_indices if i != example_idx] 
            if not reuse_pool: 
                 # print(f"ERROR: No distractors available for reuse for example {example_idx}. Skipping.")
                 continue 
            selected_indices.extend(random.choices(reuse_pool, k=remaining_needed))

        for dist_idx in selected_indices:
             dist_example = nq_dataset_buffered[dist_idx]
             dist_text = get_document_text(dist_example)
             if dist_example and dist_example.get('question') and dist_text:
                  distractor_docs_text.append(dist_text)
                  distractor_count += 1

        if len(distractor_docs_text) != NUM_DOCUMENTS - 1:
             # print(f"Warning: Could only gather {len(distractor_docs_text)} distractors for example {example_idx}. Skipping.")
             continue

        # --- Construct Prompt ---
        documents_text = [gold_doc_text] + distractor_docs_text
        random.shuffle(documents_text) 

        truncated_docs = []
        for doc in documents_text:
            doc_tokens = tokenizer.encode(doc, add_special_tokens=False)
            truncated_tokens = doc_tokens[:MAX_TOKENS_PER_DOC]
            truncated_docs.append(tokenizer.decode(truncated_tokens))

        prompt_parts = [f"Question: {question}\n\nSearch Results:\n"]
        doc_token_spans = []
        current_tokens = tokenizer.encode(prompt_parts[0], add_special_tokens=False)
        actual_num_docs_added = 0

        for i, doc_text in enumerate(truncated_docs):
            doc_header = f"Document [{i+1}]: "
            header_tokens = tokenizer.encode(doc_header, add_special_tokens=False)
            doc_tokens = tokenizer.encode(doc_text + "\n", add_special_tokens=False)
            full_doc_segment_tokens = header_tokens + doc_tokens
            
            start_idx = len(current_tokens)
            end_idx = start_idx + len(full_doc_segment_tokens)

            final_q_part_rough = f"\nBased *only* on the documents provided, answer the question: {question}\nAnswer:"
            approx_final_q_len = len(tokenizer.encode(final_q_part_rough, add_special_tokens=False)) + 10

            if end_idx + approx_final_q_len > MAX_CONTEXT_LENGTH:
                break

            doc_token_spans.append((start_idx, end_idx))
            current_tokens.extend(full_doc_segment_tokens)
            prompt_parts.append(doc_header + doc_text + "\n")
            actual_num_docs_added += 1

        if actual_num_docs_added == 0:
             continue

        final_q_part = f"\nBased *only* on the documents provided, answer the question: {question}\nAnswer:"
        final_q_tokens = tokenizer.encode(final_q_part, add_special_tokens=False)

        if len(current_tokens) + len(final_q_tokens) > MAX_CONTEXT_LENGTH:
             overflow = (len(current_tokens) + len(final_q_tokens)) - MAX_CONTEXT_LENGTH
             current_tokens = current_tokens[:MAX_CONTEXT_LENGTH - len(final_q_tokens)]
             if doc_token_spans:
                 last_start, last_end = doc_token_spans[-1]
                 if last_end > len(current_tokens):
                      new_end = len(current_tokens)
                      if new_end > last_start:
                          doc_token_spans[-1] = (last_start, new_end)
                      else:
                          doc_token_spans.pop()
                          actual_num_docs_added -= 1
                          if actual_num_docs_added == 0: continue

        current_tokens.extend(final_q_tokens)
        input_ids = torch.tensor([current_tokens], device=device)

        # --- Get Attention Weights ---
        try:
            with torch.no_grad():
                outputs = model(input_ids, output_attentions=True)

            if not hasattr(outputs, 'attentions') or outputs.attentions is None or \
               not isinstance(outputs.attentions, (list, tuple)) or \
               any(attn is None for attn in outputs.attentions) or \
               len(outputs.attentions) == 0:
                # print(f"\nWarning: Invalid/missing attentions for example {example_idx}. Skipping.")
                if 'outputs' in locals(): del outputs
                if 'input_ids' in locals(): del input_ids
                gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
                continue

            attentions = outputs.attentions
            stacked_attentions = torch.stack([attn.float() for attn in attentions])
            avg_attention = stacked_attentions.mean(dim=(0, 2))
            last_token_attention = avg_attention[0, -1, :]

            # --- Map Attention to Documents ---
            doc_attentions = np.full(NUM_DOCUMENTS, np.nan)
            seq_len = last_token_attention.shape[0]

            for i, (start_idx, end_idx) in enumerate(doc_token_spans):
                 valid_start = max(0, start_idx)
                 valid_end = min(seq_len, end_idx)
                 if valid_start < valid_end:
                     mean_attn_span = last_token_attention[valid_start:valid_end].mean().item()
                     if np.isfinite(mean_attn_span):
                         doc_attentions[i] = mean_attn_span
                     else:
                         doc_attentions[i] = 0.0 # Or np.nan

            model_attention_profiles.append(doc_attentions) # Add the 1D array for this example

        except RuntimeError as e:
            print(f"\nERROR during model inference/attention processing for example {example_idx}: {e}")
            if "CUDA out of memory" in str(e): print("CUDA OOM Error.")
        except Exception as e:
             print(f"\nUnexpected Error during processing example {example_idx} in Rep {rep+1}: {e}")
        finally:
            # Cleanup
            if 'outputs' in locals(): del outputs
            if 'attentions' in locals(): del attentions
            if 'stacked_attentions' in locals(): del stacked_attentions
            if 'avg_attention' in locals(): del avg_attention
            if 'last_token_attention' in locals(): del last_token_attention
            if 'input_ids' in locals(): del input_ids
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        # --- END: Code from previous version for processing one example ---

    # --- End of example loop for this repetition ---
    pbar.close()

    # --- Calculate and Store Mean Profile for THIS Repetition ---
    if model_attention_profiles:
        rep_matrix = np.array(model_attention_profiles)
        if rep_matrix.ndim == 2 and rep_matrix.shape[1] == NUM_DOCUMENTS:
            mean_rep_attention = np.nanmean(rep_matrix, axis=0) # Mean across examples for this rep
            all_repetition_means.append(mean_rep_attention)
            print(f"Finished Repetition {rep+1}. Stored mean attention profile (length {len(mean_rep_attention)}).")
            print(f" Rep {rep+1} mean profile sample: {mean_rep_attention[:5]}...") # Print start of mean profile
        else:
            print(f"Warning: Invalid data shape in Rep {rep+1}. Matrix shape: {rep_matrix.shape if isinstance(rep_matrix, np.ndarray) else 'N/A'}. Skipping storage for this rep.")
    else:
        print(f"Warning: No successful attention profiles generated for Repetition {rep+1}.")
        all_repetition_means.append(np.full(NUM_DOCUMENTS, np.nan)) # Add placeholder if needed

# --- End of Repetition loop ---

# --- Unload Model ---
print(f"\n--- Unloading Model: {model_id} ---")
del model
del tokenizer
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("-" * 20)


# --- Aggregating and Plotting ---
print("\n===== Plotting Results =====")

if not all_repetition_means or all(np.all(np.isnan(mean_prof)) for mean_prof in all_repetition_means):
    print("No valid mean repetition results were collected. Exiting.")
    exit()

# --- Plotting ---
fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=600) # Single plot

model_short_name = TARGET_MODEL_ID.split('/')[-1]
style_info = model_plot_styles.get(TARGET_MODEL_ID)
base_color, _, _, _ = style_info if style_info else ('#252A34', None, None, None)

doc_positions = np.arange(1, NUM_DOCUMENTS + 1)
num_valid_reps_plotted = 0

# Plot each repetition's mean profile
for rep_idx, mean_rep_attention in enumerate(all_repetition_means):
    valid_plot_mask = ~np.isnan(mean_rep_attention)
    plot_x = doc_positions[valid_plot_mask]
    plot_y = mean_rep_attention[valid_plot_mask]

    if plot_x.size > 0:
        ax.plot(
            plot_x, plot_y,
            color=base_color,
            linewidth=0.8,  
            linestyle="-",
            marker=None,     
            alpha=0.4,       
        )
        num_valid_reps_plotted += 1
    else:
        print(f"Skipping plot for Rep {rep_idx+1} as no valid data points found.")


# Optionally plot the overall average across repetitions
if num_valid_reps_plotted > 0:
     overall_mean_attention = np.nanmean(np.array(all_repetition_means), axis=0)
     valid_overall_mask = ~np.isnan(overall_mean_attention)
     overall_x = doc_positions[valid_overall_mask]
     overall_y = overall_mean_attention[valid_overall_mask]

     if overall_x.size > 0:
         ax.plot(
             overall_x, overall_y,
             color='red', # Use a distinct color for the average
             linewidth=2.0,
             linestyle="--",
             marker='^',
             markersize=4,
             alpha=0.9,
             label=f"Overall Avg ({num_valid_reps_plotted} reps)"
         )


# --- Plot Styling ---
ax.set_xlabel("Document Position", fontsize=fontsize)
ax.set_ylabel("Avg. Attention Weight", fontsize=fontsize)

# Ticks
ax.tick_params(axis='both', which='major', labelsize=fontsize * 0.9)
tick_step = max(1, (NUM_DOCUMENTS -1) // 4 if NUM_DOCUMENTS > 1 else 1)
ticks = np.arange(1, NUM_DOCUMENTS + 1, tick_step)
if NUM_DOCUMENTS > 1 and ticks[-1] != NUM_DOCUMENTS :
     ticks = np.append(ticks, NUM_DOCUMENTS)
ax.set_xticks(ticks)
ax.set_xticklabels(ticks.astype(int))
ax.set_xlim(0.5, NUM_DOCUMENTS + 0.5)

# Auto-adjust y-limits based on *all* plotted data
all_y_data = []
for mean_prof in all_repetition_means:
    all_y_data.extend(mean_prof[~np.isnan(mean_prof)])
if 'overall_y' in locals() and overall_y.size > 0:
     all_y_data.extend(overall_y)

if all_y_data:
     min_y = np.min(all_y_data)
     max_y = np.max(all_y_data)
     padding = (max_y - min_y) * 0.05 if (max_y - min_y) > 1e-6 else 0.1 # Add padding, handle flat case
     ax.set_ylim(min_y - padding, max_y + padding)
else:
     # Fallback if somehow no data was plotted
     ax.set_ylim(0, 0.1)


ax.grid(False)

# Add legend (only for the overall average line if plotted)
handles, labels = ax.get_legend_handles_labels()
if handles: # Only show legend if the overall average line was successfully plotted
    ax.legend(handles=handles, labels=labels, fontsize=fontsize * 0.9, loc='best')

# --- Final Adjustments and Saving ---
fig.tight_layout(rect=[0, 0.03, 1, 0.93]) # Adjust rect to accommodate title

try:
    plt.savefig(output_filename, dpi=600, bbox_inches='tight')
    print(f"\nPlot saved as {output_filename}")
except Exception as e:
     print(f"Error saving figure: {e}")

# plt.show() 

print("\nExperiment finished.")
# --- END OF FILE ---