"""
Worker Functions for Threading and Multiprocessing.

This module contains all worker functions used in concurrent processing benchmarks.
Functions must be at module level to be picklable for multiprocessing.
"""

import time
import random
import string
from typing import List, Dict, Tuple, Any, Optional


# =============================================================================
# GLOBAL STATE FOR MULTIPROCESSING WORKERS
# =============================================================================

# Global tokenizer for multiprocessing workers (initialized per-process)
_worker_tokenizer = None
_worker_config = None  # Store config for data generation


def init_worker(model_id: str, hf_token: Optional[str], config: Dict[str, Any] = None):
    """Initialize tokenizer once per worker process."""
    global _worker_tokenizer, _worker_config
    from transformers import AutoTokenizer
    
    _worker_tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=hf_token,
        use_fast=True
    )
    _worker_config = config or {}


# =============================================================================
# IN-WORKER DATA GENERATION (Avoids IPC overhead)
# =============================================================================

def _generate_random_word_in_worker(rng: random.Random) -> str:
    """Generate a random word."""
    length = rng.randint(4, 10)
    return ''.join(rng.choices(string.ascii_lowercase, k=length))


def _generate_structured_content_in_worker(
    rng: random.Random, target_words: int, content_type: int, is_user: bool, turn: int
) -> str:
    """Generate structured content inside worker process."""
    if content_type == 0:
        return _generate_prose_in_worker(rng, target_words, turn)
    elif content_type == 1:
        return _generate_code_in_worker(rng, target_words, turn)
    else:
        return _generate_list_in_worker(rng, target_words, turn)


def _generate_prose_in_worker(rng: random.Random, target_words: int, turn: int) -> str:
    """Generate prose-style paragraphs."""
    starters = [
        "The system", "Furthermore,", "In this case,", "Consider that",
        "However,", "Additionally,", "For example,", "Note that",
    ]
    
    words = []
    sentence_length = 0
    
    for i in range(target_words):
        if sentence_length == 0 and i > 0:
            if rng.random() < 0.3:
                words.append(rng.choice(starters))
            else:
                words.append(_generate_random_word_in_worker(rng).capitalize())
        else:
            words.append(_generate_random_word_in_worker(rng))
        
        sentence_length += 1
        
        if sentence_length >= rng.randint(8, 15):
            words[-1] = words[-1] + rng.choice(['.', '.', '.', '!', '?'])
            sentence_length = 0
        elif sentence_length > 4 and rng.random() < 0.15:
            words[-1] = words[-1] + ','
    
    if not words[-1].endswith(('.', '!', '?')):
        words[-1] = words[-1].rstrip(',') + '.'
    
    return f"Query {turn+1}: " + ' '.join(words) if turn >= 0 else ' '.join(words)


def _generate_code_in_worker(rng: random.Random, target_words: int, turn: int) -> str:
    """Generate Python-like code snippets."""
    operators = ['==', '!=', '>=', '<=', '+=', '-=']
    
    lines = ["```python"]
    word_count = 0
    current_indent = 0
    
    while word_count < target_words:
        indent = "    " * current_indent
        structure_type = rng.randint(0, 4)
        
        if structure_type == 0:
            func_name = '_'.join([_generate_random_word_in_worker(rng) for _ in range(rng.randint(1, 2))])
            params = ', '.join([f"{_generate_random_word_in_worker(rng)}_data" for _ in range(rng.randint(1, 3))])
            lines.append(f"{indent}def {func_name}({params}):")
            word_count += 4 + params.count(',')
            current_indent = min(current_indent + 1, 4)
        elif structure_type == 1:
            var_name = '_'.join([_generate_random_word_in_worker(rng) for _ in range(rng.randint(1, 2))])
            value = f'"{_generate_random_word_in_worker(rng)}_{rng.randint(0, 999)}"'
            lines.append(f"{indent}{var_name} = {value}")
            word_count += 3
        elif structure_type == 2:
            var = f"{_generate_random_word_in_worker(rng)}_value"
            op = rng.choice(operators)
            lines.append(f"{indent}if {var} {op} {rng.randint(0, 100)}:")
            word_count += 4
            current_indent = min(current_indent + 1, 4)
        elif structure_type == 3:
            iter_var = rng.choice(['i', 'j', 'idx', 'item'])
            collection = f"{_generate_random_word_in_worker(rng)}_list"
            lines.append(f"{indent}for {iter_var} in {collection}:")
            word_count += 4
            current_indent = min(current_indent + 1, 4)
        else:
            func = rng.choice(['print', 'return'])
            content = f'{_generate_random_word_in_worker(rng)}_{_generate_random_word_in_worker(rng)}'
            lines.append(f"{indent}{func}({content})" if func == 'print' else f"{indent}{func} {content}")
            word_count += 3
            if func == 'return' and current_indent > 0:
                current_indent -= 1
        
        if current_indent > 0 and rng.random() < 0.2:
            current_indent -= 1
            lines.append("")
    
    lines.append("```")
    return '\n'.join(lines)


def _generate_list_in_worker(rng: random.Random, target_words: int, turn: int) -> str:
    """Generate Markdown bullet point lists."""
    lines = [f"Query {turn+1} - Key points:", ""]
    word_count = 0
    item_num = 1
    
    while word_count < target_words:
        marker = f"{item_num}." if rng.random() < 0.3 else "-"
        item_num += 1
        
        item_words = [_generate_random_word_in_worker(rng) for _ in range(rng.randint(5, 12))]
        lines.append(f"{marker} {' '.join(item_words)}")
        word_count += len(item_words)
        
        if rng.random() < 0.3:
            nested_words = [_generate_random_word_in_worker(rng) for _ in range(rng.randint(3, 6))]
            lines.append(f"  - {' '.join(nested_words)}")
            word_count += len(nested_words)
    
    return '\n'.join(lines)


def _generate_chat_in_worker(num_turns: int, words_per_message: int, seed: int) -> List[Dict[str, str]]:
    """
    Generate chat conversation inside worker process with realistic structured content.
    
    Alternates between:
    - Prose: Natural language paragraphs
    - Code blocks: Python-like snippets with indentation
    - Lists: Markdown bullet points
    
    This stresses the tokenizer's handling of whitespace and special tokens,
    which is often slower than pure alphanumeric text.
    """
    rng = random.Random(seed)
    
    messages = [
        {"role": "system", "content": f"You are a helpful AI assistant. Session: {seed}"}
    ]
    
    for i in range(num_turns):
        # Rotate through content types: prose -> code -> list
        content_type = i % 3
        
        # User message
        user_content = _generate_structured_content_in_worker(rng, words_per_message, content_type, is_user=True, turn=i)
        messages.append({"role": "user", "content": user_content})
        
        # Assistant response (use next content type for variety)
        assistant_type = (content_type + 1) % 3
        assistant_content = _generate_structured_content_in_worker(rng, words_per_message * 2, assistant_type, is_user=False, turn=i)
        messages.append({"role": "assistant", "content": assistant_content})
    
    # Final user message - always code to stress tokenizer at the end
    final_content = _generate_structured_content_in_worker(rng, words_per_message, content_type=1, is_user=True, turn=num_turns)
    messages.append({"role": "user", "content": final_content})
    
    return messages


# =============================================================================
# MULTIPROCESSING WORKER FUNCTIONS
# =============================================================================

def mp_jinja_worker_with_data_gen(task_id: int) -> Tuple[str, int]:
    """
    Worker function that generates data internally to avoid IPC overhead.
    
    Instead of sending large message dicts over IPC, we send only a task_id
    and generate the data inside the worker process.
    """
    global _worker_tokenizer, _worker_config
    
    # Generate data inside the worker (avoids IPC overhead)
    turns = _worker_config.get("turns", 100)
    words_per_msg = _worker_config.get("words_per_message", 300)
    
    messages = _generate_chat_in_worker(turns, words_per_msg, task_id)
    result = _worker_tokenizer.apply_chat_template(messages, tokenize=False)
    return result, len(messages)


# =============================================================================
# THREADING WORKER FUNCTIONS
# =============================================================================

def jinja_worker_with_tokenizer(args: Tuple[Any, List[Dict]]) -> str:
    """Worker function for Jinja templating (for thread pool)."""
    tokenizer, messages = args
    return tokenizer.apply_chat_template(messages, tokenize=False)


def tokenize_worker(args: Tuple[Any, str]) -> List[int]:
    """Worker function for tokenization (for thread pool)."""
    tokenizer, text = args
    return tokenizer.encode(text, add_special_tokens=False)


def full_preprocess_worker(args: Tuple[Any, List[Dict]]) -> Tuple[str, List[int], float, float]:
    """Worker function for full preprocessing with timing."""
    tokenizer, messages = args
    
    start = time.perf_counter()
    raw_str = tokenizer.apply_chat_template(messages, tokenize=False)
    jinja_ms = (time.perf_counter() - start) * 1000
    
    start = time.perf_counter()
    input_ids = tokenizer.encode(raw_str, add_special_tokens=False)
    tokenize_ms = (time.perf_counter() - start) * 1000
    
    return raw_str, input_ids, jinja_ms, tokenize_ms
