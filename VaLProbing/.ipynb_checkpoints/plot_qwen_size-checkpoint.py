# -*- coding: utf-8 -*-
# This source code is licensed under the MIT license

import json
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from tqdm import tqdm
import matplotlib.colors as mcolors

# === Configuration ===

# --- Plotting Style and Model Information ---
model_infos = [
    ['Qwen2.5-1.5B-Instruct', '#FF2E63', '#FF2E63', ':', 'o'],        
    ['Qwen2.5-3B-Instruct', '#08D9D6', '#08D9D6', 'dotted', 's'],
    ['Qwen2.5-7B-Instruct', '#F9A826', '#F9A826', '-.', '^']
]

fontsize = 20

# --- File Paths ---
PROJECT_BASE = '/root/FILM' 
SKIP_RESULTS_BASE = os.path.join(PROJECT_BASE, 'VaLProbing', 'VaLProbing-32K')
RESULTS_DIR = os.path.join(SKIP_RESULTS_BASE, 'results')
LABEL_BASE = os.path.join(PROJECT_BASE, 'ValProbing-32K')
FIGURES_DIR = os.path.join(PROJECT_BASE, 'VaLProbing', 'figures') 

os.makedirs(FIGURES_DIR, exist_ok=True)

DOC_LABEL_PATH = os.path.join(LABEL_BASE, 'document_bi_32k.jsonl')
DOC_SKIP_PATH = os.path.join(SKIP_RESULTS_BASE, 'document_bi_32k_skip_list.json')
DOC_RESULT_PATTERN = os.path.join(RESULTS_DIR, '{}/sample_document_bi_32k.jsonl')

CODE_LABEL_PATH = os.path.join(LABEL_BASE, 'code_backward_32k.jsonl')
CODE_SKIP_PATH = os.path.join(SKIP_RESULTS_BASE, 'code_backward_32k_skip_list.json')
CODE_RESULT_PATTERN = os.path.join(RESULTS_DIR, '{}/sample_code_backward_32k.jsonl')

DB_LABEL_PATH = os.path.join(LABEL_BASE, 'database_forward_32k.jsonl')
DB_SKIP_PATH = os.path.join(SKIP_RESULTS_BASE, 'database_forward_32k_skip_list.json')
DB_RESULT_PATTERN = os.path.join(RESULTS_DIR, '{}/sample_database_forward_32k.jsonl')

# Output figure paths (using constructed paths for flexibility)
PDF_OUTPUT_PATH = os.path.join(FIGURES_DIR, 'qwen_probing_fig.pdf')
JPG_OUTPUT_PATH = os.path.join(FIGURES_DIR, 'qwen_probing_fig.jpg')


# === Helper Functions ===

def hex_to_rgba(hex_color, alpha=0.7):
    rgba = mcolors.to_rgba(hex_color, alpha)
    return rgba

# === Core Computation and Plotting Function ===

def compute_and_plot(task_name, ax, total_len, span_len, label_path, skip_path, result_path_pattern):
    print(f"--- Processing Task: {task_name.capitalize()} ---")
    span_num = int(total_len / span_len)
    set_ids = ['set_' + str(i) for i in range(4)]

    # --- Load Ground Truth Labels ---
    print(f"Attempting to load labels from: {label_path}")
    label_infos = []
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines(), desc="Reading Labels"):
                info = json.loads(line)
                if task_name == 'document':
                    label_infos.append({
                        'gt': info['completion'], 'position_id': info['position_id'], 'set_id': 'set_' + str(info['set_id'])
                    })
                elif task_name == 'code':
                    label_infos.append({
                        'gt': info['completion'], 'position_id': info['position_id'], 'set_id': 'set_' + str(info['set_id'])
                    })
                elif task_name == 'structure':
                    label_infos.append({
                        'gt_label': info['label'], 'gt_description': info['description'], 'position_id': info['position_id'], 'set_id': 'set_' + str(info['set_id'])
                    })
                else:
                     print(f"Warning: Unknown task type '{task_name}' during label loading.")
        print(f"Successfully loaded {len(label_infos)} labels.")
    except FileNotFoundError:
        print(f"Error: Label file not found at {label_path}. Cannot proceed with this task.")
        return # Stop processing this task
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from label file {label_path}: {e}. Cannot proceed with this task.")
        return 
    except Exception as e:
        print(f"An unexpected error occurred loading labels from {label_path}: {e}. Cannot proceed with this task.")
        return 

    # --- Load Skip List ---
    print(f"Loading skip list from: {skip_path}")
    skip_list = []
    try:
        with open(skip_path, 'r', encoding='utf-8') as f:
            skip_list = json.load(f)
        print(f"Loaded skip list with {len(skip_list)} items.")
    except FileNotFoundError:
        print(f"Warning: Skip list file not found at {skip_path}. Assuming no skips.")
    except json.JSONDecodeError:
         print(f"Warning: Could not decode JSON from skip list file {skip_path}. Assuming no skips.")

    # --- Process Each Model ---
    for model_name, color_str, ecolor_str, linestyle, marker in model_infos: 
        print(f"\nProcessing model: {model_name}")
        set_ids_position2acc = {sid: {i: [] for i in range(total_len)} for sid in set_ids}

        result_path = result_path_pattern.format(model_name)
        print(f"Loading results from: {result_path}")
        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                pred_infos = [json.loads(line) for line in f.readlines()]

                if len(pred_infos) != len(label_infos):
                     print(f"Warning: Number of predictions ({len(pred_infos)}) does not match number of labels ({len(label_infos)}) for {model_name}. Results may be skewed.")

                for idx, (pred_info, label_info) in enumerate(tqdm(zip(pred_infos, label_infos), total=min(len(pred_infos), len(label_infos)), desc=f"Evaluating {model_name}")):
                    if idx in skip_list:
                        continue
                    # Prediction extraction logic (same as before)
                    pred = pred_info.get('sample', '') if 'GPT' in model_name else (pred_info.get('samples', [''])[0] if pred_info.get('samples') else '')
                    position_id = label_info['position_id']
                    set_id = label_info['set_id']
                    if task_name == 'document':
                        gt = label_info['gt']
                        gt_words = set(gt.strip().lower().split())
                        pred_words = set(pred.strip().lower().split())
                        if len(gt_words) > 0:
                            recall_score = len(gt_words & pred_words) / len(gt_words)
                            set_ids_position2acc[set_id][position_id].append(recall_score)
                    elif task_name == 'code':
                        gt = label_info['gt']
                        is_correct = 1 if gt.strip('.') in pred else 0
                        set_ids_position2acc[set_id][position_id].append(is_correct)
                    elif task_name == 'structure':
                        gt_label = label_info['gt_label']
                        gt_description = label_info['gt_description']
                        pred_lower = pred.lower()
                        is_correct = 1 if gt_label.strip('.').lower() in pred_lower or gt_description.strip('.').lower() in pred_lower else 0
                        set_ids_position2acc[set_id][position_id].append(is_correct)

            # --- Aggregate Accuracies into Spans ---
            set_ids2span_acc_list = {}
            for set_id in set_ids:
                 set_ids2span_acc_list[set_id] = []
                 for i in range(span_num):
                    span_start = span_len * i
                    span_end = span_start + span_len
                    accs_in_span = [acc for pos_id in range(span_start, span_end) for acc in set_ids_position2acc[set_id].get(pos_id, [])]
                    set_ids2span_acc_list[set_id].append(sum(accs_in_span) / len(accs_in_span) if accs_in_span else None)

            span_acc_list = []
            span_std_list = []
            for i in range(span_num):
                accs_across_sets = [set_ids2span_acc_list[set_id][i] for set_id in set_ids if set_ids2span_acc_list[set_id][i] is not None]
                if accs_across_sets:
                    span_acc_list.append(sum(accs_across_sets) / len(accs_across_sets))
                    span_std_list.append(np.std(np.array(accs_across_sets)) if len(accs_across_sets) > 1 else 0)
                else:
                    span_acc_list.append(None)
                    span_std_list.append(0)

            # --- Prepare Data for Plotting ---
            valid_indices = [i for i in range(span_num) if span_acc_list[i] is not None]
            x_values = valid_indices
            y_values = [span_acc_list[i] for i in valid_indices]
            y_errors = [span_std_list[i] for i in valid_indices]

            # --- Plotting ---
            if x_values:
                line_container = ax.errorbar(
                    x_values, y_values, yerr=y_errors,
                    color=color_str,          
                    linewidth=3,              
                    markersize=10,           
                    linestyle=linestyle,    
                    marker=marker,         
                    ecolor=ecolor_str,     
                    elinewidth=3,           
                    capsize=6,               
                    label=model_name
                )

                if not hasattr(ax, 'custom_handles'):
                    ax.custom_handles = []
                    ax.custom_labels = []
                ax.custom_handles.append(line_container[0])
                ax.custom_labels.append(model_name)
                # Remove auto-generated legend 
                ax.legend_.remove() if ax.legend_ else None
            else:
                 print(f"Warning: No valid data points to plot for model {model_name} on task {task_name}.")

            # Print statistics 
            print(f'{model_name} Statistics ({task_name}):')
            if y_values:
                print(f'  Avg Performance across Spans: {sum(y_values) / len(y_values):.4f}')
                print(f'  Max-Min Performance Gap:    {max(y_values) - min(y_values):.4f}')
            else:
                print('  No valid data points found.')

        except FileNotFoundError:
            print(f"Error: Result file not found for model {model_name} at {result_path}. Skipping this model.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from result file {result_path} for model {model_name}: {e}. Skipping this model.")
        except Exception as e:
            print(f"An error occurred while processing model {model_name}: {e}")

    # --- Finalize Subplot Appearance ---
    ax.autoscale() 

    # Set X-axis ticks and labels 
    x_tickets = [str(span_len * (i + 1)) for i in range(span_num)] 
    ax.set_xticks(range(span_num))
    ax.set_xticklabels(x_tickets, fontsize=fontsize * 0.5, rotation=45)

    # Set Y-axis ticks and labels 
    ax.tick_params(axis='y', labelsize=fontsize)

    # Set labels and title
    ax.set_xlabel('Relative Position', fontsize=fontsize * 1.5) 
    ax.set_title(task_name.capitalize(), fontsize=fontsize * 1.5) 

    print(f"--- Finished Task: {task_name.capitalize()} ---")


# === Main Execution ===

# --- Initialize Figure and Subplots ---
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10), dpi=600) 
fig.suptitle('Qwen 2.5', fontsize=fontsize * 1.7, y=1.05)

# --- Process and Plot Each Task ---

# Task 1: Document
total_len_doc = 800
span_len_doc = 50
compute_and_plot(
    task_name='document', ax=ax1, total_len=total_len_doc, span_len=span_len_doc,
    label_path=DOC_LABEL_PATH, skip_path=DOC_SKIP_PATH, result_path_pattern=DOC_RESULT_PATTERN,
)

# Task 2: Code
total_len_code = 800
span_len_code = 50
compute_and_plot(
    task_name='code', ax=ax2, total_len=total_len_code, span_len=span_len_code,
    label_path=CODE_LABEL_PATH, skip_path=CODE_SKIP_PATH, result_path_pattern=CODE_RESULT_PATTERN,
)

# Task 3: Structure (Database)
total_len_db = 750
span_len_db = 50
compute_and_plot(
    task_name='structure', ax=ax3, total_len=total_len_db, span_len=span_len_db,
    label_path=DB_LABEL_PATH, skip_path=DB_SKIP_PATH, result_path_pattern=DB_RESULT_PATTERN,
)


# --- Create a Unified Figure Legend ---
handles = []
labels = []
for ax in [ax1, ax2, ax3]:
    if hasattr(ax, 'custom_handles') and hasattr(ax, 'custom_labels'):
        for handle, label in zip(ax.custom_handles, ax.custom_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)

if handles and labels:
    fig.legend(
        handles, labels,
        loc='center left',           
        bbox_to_anchor=(0.8, 0.5),   
        fontsize=fontsize * 1.2,     
        frameon=False,               
        ncol=1,                     
        labelspacing=1.2,            
        handlelength=2.5             
    )
else:
    print("Warning: No plot handles found to create a figure legend.")


# --- Final Figure Adjustments and Saving ---

fig.text(0.01, 0.5, 'Performance', va='center', rotation='vertical', fontsize=fontsize * 1.5) 
fig.subplots_adjust(left=0.05, right=0.8, bottom=0.1, top=0.95)
print(f"\nSaving figure to {PDF_OUTPUT_PATH} and {JPG_OUTPUT_PATH}...")

try:
    with PdfPages(PDF_OUTPUT_PATH) as pp:
        pp.savefig(fig, bbox_inches='tight') 
    fig.savefig(JPG_OUTPUT_PATH, dpi=600, bbox_inches='tight') 
    print("Figure saved successfully.")
except Exception as e:
    print(f"Error saving figure: {e}")

# plt.show() 