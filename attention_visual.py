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
# os.environ["HUGGING_FACE_HUB_TOKEN"] = "YOUR_HF_TOKEN" 

# --- Configuration ---

MODEL_IDS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct"
]

NUM_DOCUMENTS_LIST = [10, 20, 30]  
DATASET_NAME = "google-research-datasets/nq_open" 
NUM_EXAMPLES_TO_RUN = 100
MAX_TOKENS_PER_DOC = 4096 
MAX_CONTEXT_LENGTH = 32768 

# --- Plotting Style Configuration ---
model_plot_styles = {
    "Qwen/Qwen2.5-1.5B-Instruct":   ['#FF2E63', '#FF2E63', ':', 'v'],
    "Qwen/Qwen2.5-3B-Instruct":   ['#08D9D6', '#08D9D6', '--', 's'], 
    "Qwen/Qwen2.5-7B-Instruct": ['#252A34', '#252A34', '-', 'o'],
}
fontsize = 15
output_filename = "qwen_attention_comparison_k10_20_30_auto_yaxis.jpg" 

# --- Helper Function ---
def hex_to_rgba(hex_color, alpha=0.5):
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
max_k = max(NUM_DOCUMENTS_LIST)
buffer_needed = NUM_EXAMPLES_TO_RUN * (max_k + 10)

try:
    nq_dataset = load_dataset(DATASET_NAME, split='train', streaming=True)
    print(f"Buffering {buffer_needed} examples from NQ dataset...")
    nq_dataset_buffered = list(nq_dataset.take(buffer_needed))
    print(f"Buffered {len(nq_dataset_buffered)} examples.")
    if len(nq_dataset_buffered) < NUM_EXAMPLES_TO_RUN + max_k:
        print(f"Warning: Dataset buffer ({len(nq_dataset_buffered)}) is smaller than potentially needed ({NUM_EXAMPLES_TO_RUN + max_k}). May reuse examples or fail for larger K.")

except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Please ensure you have internet connection and the 'datasets' library installed.")
    exit()

def get_document_text(example):
    if not example: return "Invalid Example"
    answer_candidates = example.get('answer', [])
    answer_text = answer_candidates[0] if answer_candidates else "No Answer Found"
    return f"Question: {example.get('question', 'No Question')} Answer: {answer_text}"

# --- Pre-select Examples ---
print(f"Selecting {NUM_EXAMPLES_TO_RUN} base examples...")
primary_examples_data = []
example_indices_used = set()
available_indices = list(range(len(nq_dataset_buffered)))
random.shuffle(available_indices)
processed_count = 0

for idx in available_indices:
    if processed_count >= NUM_EXAMPLES_TO_RUN:
        break
    example = nq_dataset_buffered[idx]
    if example and example.get('question'):
        primary_examples_data.append({'index': idx, 'example': example})
        example_indices_used.add(idx)
        processed_count += 1

if processed_count < NUM_EXAMPLES_TO_RUN:
    print(f"Warning: Only found {processed_count} valid examples to run out of {NUM_EXAMPLES_TO_RUN} requested.")
    NUM_EXAMPLES_TO_RUN = processed_count

print(f"Selected {NUM_EXAMPLES_TO_RUN} examples.")
available_distractor_indices = [i for i in range(len(nq_dataset_buffered)) if i not in example_indices_used]
print(f"Available distractor indices: {len(available_distractor_indices)}")


# --- Experiment Loop ---
results = {}

for num_docs in NUM_DOCUMENTS_LIST:
    print(f"\n===== Running for NUM_DOCUMENTS (K) = {num_docs} =====")
    results[num_docs] = {}

    if len(available_distractor_indices) < num_docs -1:
        print(f"Warning: Not enough unique distractors ({len(available_distractor_indices)}) available for K={num_docs}. Will reuse distractors.")

    for model_id in MODEL_IDS:
        print(f"\n--- Loading Model: {model_id} ---")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager"
            )
            model.eval()

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = model.config.eos_token_id
            print("Model and tokenizer loaded.")

        except Exception as e:
            print(f"ERROR loading model {model_id}: {e}. Skipping this model for K={num_docs}.")
            if 'model' in locals(): del model
            if 'tokenizer' in locals(): del tokenizer
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            continue

        print(f"--- Running Inference for {model_id} (K={num_docs}) ---")
        model_attention_profiles = []
        pbar = tqdm(primary_examples_data, desc=f"Processing {model_id} K={num_docs}", leave=False)

        for primary_data in pbar:
            example_idx = primary_data['index']
            example = primary_data['example']
            question = example['question']
            gold_doc_text = get_document_text(example)

            # --- Sample Distractor Documents ---
            distractor_docs_text = []
            current_distractor_pool = list(available_distractor_indices)
            random.shuffle(current_distractor_pool)
            distractor_count = 0
            distractor_indices_used_this_round = set()

            while distractor_count < num_docs - 1 and len(current_distractor_pool) > 0:
                dist_idx = current_distractor_pool.pop()
                if dist_idx == example_idx: continue
                dist_example = nq_dataset_buffered[dist_idx]
                dist_text = get_document_text(dist_example)
                if dist_example and dist_example.get('question') and dist_text and dist_idx not in distractor_indices_used_this_round:
                    distractor_docs_text.append(dist_text)
                    distractor_indices_used_this_round.add(dist_idx)
                    distractor_count += 1

            if distractor_count < num_docs - 1:
                 reuse_pool = [i for i in available_distractor_indices if i != example_idx]
                 random.shuffle(reuse_pool)
                 needed = (num_docs - 1) - distractor_count
                 for i in range(needed):
                     if not reuse_pool: break
                     dist_idx = reuse_pool[i % len(reuse_pool)]
                     dist_example = nq_dataset_buffered[dist_idx]
                     dist_text = get_document_text(dist_example)
                     if dist_example and dist_example.get('question') and dist_text:
                         distractor_docs_text.append(dist_text)
                         distractor_count += 1

            if len(distractor_docs_text) < num_docs - 1:
                # print(f"Warning: Could only gather {len(distractor_docs_text)} distractors for example {example_idx} (K={num_docs}). Skipping this example.")
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
                 current_tokens = current_tokens[:-overflow]
                 if doc_token_spans:
                     last_start, last_end = doc_token_spans[-1]
                     if last_end > len(current_tokens):
                          new_end = len(current_tokens)
                          if new_end > last_start:
                              doc_token_spans[-1] = (last_start, new_end)
                          else:
                              doc_token_spans.pop()
                              actual_num_docs_added -= 1
                              if actual_num_docs_added == 0:
                                 continue

            current_tokens.extend(final_q_tokens)
            input_ids = torch.tensor([current_tokens], device=device)

            # --- Get Attention Weights ---
            try:
                with torch.no_grad():
                    outputs = model(input_ids, output_attentions=True)

                if outputs.attentions is None or not isinstance(outputs.attentions, (list, tuple)) or any(attn is None for attn in outputs.attentions):
                    print(f"\nWarning: Invalid attentions received for example {example_idx}. Skipping attention processing.")
                    if 'outputs' in locals(): del outputs
                    if 'input_ids' in locals(): del input_ids
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    continue

                attentions = outputs.attentions
                stacked_attentions = torch.stack([attn.float() for attn in attentions])
                avg_attention = stacked_attentions.mean(dim=(0, 2))
                last_token_attention = avg_attention[0, -1, :]

                # --- Map Attention to Documents ---
                doc_attentions = np.zeros(actual_num_docs_added)
                seq_len = last_token_attention.shape[0]

                for i, (start_idx, end_idx) in enumerate(doc_token_spans):
                     valid_start = max(0, start_idx)
                     valid_end = min(seq_len, end_idx)

                     if valid_start < valid_end:
                         mean_attn_span = last_token_attention[valid_start:valid_end].mean().item()
                         if np.isfinite(mean_attn_span):
                            doc_attentions[i] = mean_attn_span
                         else:
                            doc_attentions[i] = 0.0
                     else:
                         doc_attentions[i] = 0.0

                padded_attentions = np.pad(doc_attentions,
                                           (0, num_docs - actual_num_docs_added),
                                           'constant',
                                           constant_values=np.nan)
                model_attention_profiles.append(padded_attentions)

            except RuntimeError as e:
                print(f"\nERROR during model inference/attention processing for example {example_idx}: {e}")
                if "CUDA out of memory" in str(e):
                    print("CUDA OOM Error. Try reducing K, MAX_TOKENS_PER_DOC, or NUM_EXAMPLES_TO_RUN.")
            except Exception as e:
                 print(f"\nUnexpected Error during processing example {example_idx}: {e}")
            finally:
                if 'outputs' in locals() and hasattr(outputs, 'attentions'): del outputs.attentions
                if 'outputs' in locals(): del outputs
                if 'attentions' in locals(): del attentions
                if 'stacked_attentions' in locals(): del stacked_attentions
                if 'avg_attention' in locals(): del avg_attention
                if 'last_token_attention' in locals(): del last_token_attention
                if 'input_ids' in locals(): del input_ids
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        # --- End of example loop ---
        pbar.close()

        if model_attention_profiles:
            results[num_docs][model_id] = np.array(model_attention_profiles)
            print(f"Finished {model_id} for K={num_docs}. Stored {len(model_attention_profiles)} attention profiles.")
        else:
            print(f"Warning: No successful attention profiles generated for {model_id} with K={num_docs}.")
            results[num_docs][model_id] = np.array([])

        # --- Unload Model ---
        print(f"--- Unloading Model: {model_id} ---")
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("-" * 20)

# --- End of K loop ---

# --- Plotting ---
print("\n===== Aggregating and Plotting Results =====")
if not results:
    print("No results were generated. Exiting.")
    exit()

num_subplots = len(NUM_DOCUMENTS_LIST)
fig, axes = plt.subplots(1, num_subplots,
                         figsize=(5 * num_subplots, 5),
                         sharey=False, # <<< MODIFICATION: Set to False for independent Y axes
                         dpi=600)

if num_subplots == 1:
    axes = [axes]

plot_handles = {}

for i, num_docs in enumerate(NUM_DOCUMENTS_LIST):
    ax = axes[i]
    print(f"Plotting for K={num_docs}")
    ax_has_data = False

    for model_id in MODEL_IDS:
        model_short_name = model_id.split('/')[-1]
        style_info = model_plot_styles.get(model_id)
        if not style_info:
            print(f"Warning: No plot style defined for {model_id}. Skipping.")
            continue

        color, _, linestyle, marker = style_info

        if model_id in results.get(num_docs, {}):
            attention_matrix = results[num_docs][model_id]

            if attention_matrix.ndim != 2 or attention_matrix.shape[0] == 0:
                print(f"Warning: Invalid or empty attention data for {model_short_name} with K={num_docs}. Skipping plot.")
                continue

            if attention_matrix.shape[1] != num_docs:
                 print(f"Warning: Attention matrix width ({attention_matrix.shape[1]}) != K ({num_docs}) for {model_short_name}. Skipping plot.")
                 continue

            mean_attention = np.nanmean(attention_matrix, axis=0)
            doc_positions = np.arange(1, num_docs + 1)
            valid_plot_mask = ~np.isnan(mean_attention)

            if not np.any(valid_plot_mask):
                print(f"Warning: All mean attention values are NaN for {model_short_name} with K={num_docs}. Skipping plot.")
                continue

            plot_x = doc_positions[valid_plot_mask]
            plot_y = mean_attention[valid_plot_mask]

            line = ax.plot(
                plot_x, plot_y,
                color=color,
                linewidth=1.5,
                linestyle=linestyle,
                marker=marker,
                markersize=4,
                label=model_short_name
            )
            ax_has_data = True

            if model_short_name not in plot_handles:
                plot_handles[model_short_name] = line[0]

        else:
            print(f"No results found for {model_id} with K={num_docs}")

    # --- Subplot Styling ---
    ax.set_title(f"K = {num_docs}", fontsize=fontsize * 0.8)
    ax.set_xlabel("Document Position", fontsize=fontsize * 0.8)
    # Set Y label on the first plot, or uncomment the next line to label all
    if i == 0:
         ax.set_ylabel("Avg. Attention Weight", fontsize=fontsize* 0.8)
    # else: # Optional: Label Y axis on all plots if not shared
    #     ax.set_ylabel("Avg. Attention Weight", fontsize=fontsize)


    # Ticks
    ax.tick_params(axis='both', which='major', labelsize=fontsize * 0.8)
    tick_step = max(1, (num_docs -1) // 4 if num_docs > 1 else 1)
    ticks = np.arange(1, num_docs + 1, tick_step)
    if num_docs > 1 and ticks[-1] != num_docs :
         ticks = np.append(ticks, num_docs)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks.astype(int))

    # ax.set_ylim(bottom=0) # <<< MODIFICATION: Removed this line to allow auto-scaling
    ax.grid(False)

    # Automatically adjust y-limits based on data IN THIS SUBPLOT
    # Matplotlib does this by default when set_ylim is not called and sharey=False
    # Optionally add a small padding to the auto-calculated limits:
    if ax_has_data:
        current_ylim = ax.get_ylim()
        padding = (current_ylim[1] - current_ylim[0]) * 0.05 # 5% padding
        ax.set_ylim(current_ylim[0] - padding, current_ylim[1] + padding)


    if not ax_has_data:
        ax.text(0.5, 0.5, "No data available", horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes, fontsize=fontsize, color='grey')


# --- Unified Legend ---
sorted_handles = []
sorted_labels = []
model_order = [m.split('/')[-1] for m in MODEL_IDS]
for label in model_order:
    if label in plot_handles:
        sorted_labels.append(label)
        sorted_handles.append(plot_handles[label])

if sorted_handles:
    fig.legend(
        sorted_handles, sorted_labels,
        loc='center right',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=fontsize * 0.8,
        frameon=False,
        ncol=1
    )
else:
     print("Warning: No plot handles generated, skipping legend.")


# --- Final Adjustments and Saving ---
effective_num_examples = NUM_EXAMPLES_TO_RUN
#fig.suptitle(f"Model Attention vs. Document Position (Avg. over {effective_num_examples} NQ Examples)",
#             fontsize=fontsize * 1.4, y=0.98)

fig.tight_layout(rect=[0.03, 0.05, 0.85, 0.93])

try:
    plt.savefig(output_filename, dpi=600, bbox_inches='tight')
    print(f"\nPlot saved as {output_filename}")
except Exception as e:
     print(f"Error saving figure: {e}")

# plt.show()

print("\nExperiment finished.")