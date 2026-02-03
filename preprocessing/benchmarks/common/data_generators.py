"""
Synthetic Data Generators for LLM Benchmarks.

This module provides functions to generate realistic synthetic chat conversations
and text data for benchmarking tokenizer and template processing systems.

Key features:
- Generates structured content (prose, code blocks, markdown lists)
- Supports fixed token count generation for fair comparisons
- Provides deterministic seeding for reproducibility
"""

import random
import string
from typing import List, Dict, Optional


# =============================================================================
# LARGE VOCABULARY FOR REALISTIC TOKENIZATION
# =============================================================================

def _generate_large_vocabulary(size: int = 10000, seed: int = 42) -> List[str]:
    """
    Generate a large vocabulary of pseudo-random words.
    """
    rng = random.Random(seed)
    vocab = []
    
    prefixes = ['pre', 'un', 'dis', 'mis', 're', 'over', 'under', 'out', 'sub', 'super', 
                'anti', 'auto', 'bi', 'co', 'de', 'ex', 'inter', 'multi', 'non', 'post']
    roots = ['form', 'duct', 'port', 'scrib', 'spect', 'struct', 'tract', 'vert', 'vid',
             'ject', 'mit', 'pend', 'pos', 'rupt', 'sect', 'tend', 'ven', 'vers', 'voc']
    suffixes = ['tion', 'ness', 'ment', 'able', 'ible', 'ful', 'less', 'ous', 'ive',
                'al', 'ly', 'er', 'or', 'ist', 'ism', 'ity', 'ance', 'ence', 'dom']
    
    for _ in range(size // 3):
        word = rng.choice(prefixes) + rng.choice(roots) + rng.choice(suffixes)
        vocab.append(word)
    
    for _ in range(size // 3):
        length = rng.randint(4, 12)
        word = ''.join(rng.choices(string.ascii_lowercase, k=length))
        vocab.append(word)
    
    for _ in range(size // 3):
        length = rng.randint(3, 8)
        chars = rng.choices(string.ascii_lowercase + string.digits, k=length)
        vocab.append(''.join(chars))
    
    return vocab


# Pre-generated vocabulary for consistent results
LARGE_VOCABULARY = _generate_large_vocabulary(10000)


# =============================================================================
# HELPER FUNCTIONS FOR CONTENT GENERATION
# =============================================================================

def _generate_random_word(rng: random.Random) -> str:
    """Generate a random word."""
    length = rng.randint(4, 10)
    return ''.join(rng.choices(string.ascii_lowercase, k=length))


def _generate_structured_content(rng: random.Random, target_words: int, content_type: int, is_user: bool, turn: int) -> str:
    """
    Generate structured content based on content_type:
    0 = Prose (paragraphs of natural text)
    1 = Code (Python-like snippets with indentation)
    2 = Lists (Markdown bullet points)
    
    This creates realistic LLM workload patterns that stress tokenizer handling
    of whitespace, special tokens, and structural elements.
    """
    if content_type == 0:
        return _generate_prose(rng, target_words, turn)
    elif content_type == 1:
        return _generate_code_block(rng, target_words, turn)
    else:
        return _generate_markdown_list(rng, target_words, turn)


def _generate_prose(rng: random.Random, target_words: int, turn: int) -> str:
    """Generate prose-style paragraphs with punctuation."""
    # Sentence starters for variety
    starters = [
        "The system", "Furthermore,", "In this case,", "Consider that",
        "However,", "Additionally,", "For example,", "Note that",
        "It appears", "Based on the analysis,", "When processing",
    ]
    
    words = []
    sentence_length = 0
    
    for i in range(target_words):
        if sentence_length == 0 and i > 0:
            # Start new sentence
            if rng.random() < 0.3:
                words.append(rng.choice(starters))
            else:
                words.append(_generate_random_word(rng).capitalize())
        else:
            words.append(_generate_random_word(rng))
        
        sentence_length += 1
        
        # End sentence periodically
        if sentence_length >= rng.randint(8, 15):
            words[-1] = words[-1] + rng.choice(['.', '.', '.', '!', '?'])
            sentence_length = 0
        elif sentence_length > 4 and rng.random() < 0.15:
            words[-1] = words[-1] + ','
    
    # Ensure proper ending
    if not words[-1].endswith(('.', '!', '?')):
        words[-1] = words[-1].rstrip(',') + '.'
    
    return f"Query {turn+1}: " + ' '.join(words) if turn >= 0 else ' '.join(words)


def _generate_code_block(rng: random.Random, target_words: int, turn: int) -> str:
    """
    Generate Python-like code snippets with heavy indentation.
    
    This stresses tokenizer handling of:
    - Whitespace (spaces, tabs, newlines)
    - Underscores (snake_case identifiers)
    - Special keywords (def, return, if, for, etc.)
    """
    # Python operators for conditionals
    operators = ['==', '!=', '>=', '<=', '+=', '-=', '*=', '/=', 'and', 'or', 'not', 'in', 'is']
    
    lines = []
    word_count = 0
    current_indent = 0
    
    # Start with optional intro text
    if rng.random() < 0.5:
        intro_words = [_generate_random_word(rng) for _ in range(rng.randint(5, 10))]
        lines.append(f"Here's the implementation for turn {turn+1}: " + ' '.join(intro_words))
        word_count += len(intro_words) + 5
        lines.append("")
        lines.append("```python")
    else:
        lines.append("```python")
    
    while word_count < target_words:
        indent = "    " * current_indent
        
        # Generate different code structures
        structure_type = rng.randint(0, 5)
        
        if structure_type == 0:
            # Function definition
            func_name = '_'.join([_generate_random_word(rng) for _ in range(rng.randint(1, 3))])
            params = ', '.join([f"{_generate_random_word(rng)}_{rng.choice(['data', 'value', 'input', 'config', 'idx'])}" for _ in range(rng.randint(1, 4))])
            lines.append(f"{indent}def {func_name}({params}):")
            word_count += 5 + params.count(',')
            current_indent = min(current_indent + 1, 4)
            
            # Add docstring sometimes
            if rng.random() < 0.4:
                doc_words = ' '.join([_generate_random_word(rng) for _ in range(rng.randint(5, 12))])
                lines.append(f"{'    ' * current_indent}\"\"\"{doc_words}\"\"\"")
                word_count += 8
        
        elif structure_type == 1:
            # Variable assignment with snake_case
            var_name = '_'.join([_generate_random_word(rng) for _ in range(rng.randint(1, 3))])
            if rng.random() < 0.5:
                value = f"[{', '.join([str(rng.randint(0, 100)) for _ in range(rng.randint(3, 8))])}]"
            else:
                value = f'"{_generate_random_word(rng)}_{rng.randint(0, 999)}"'
            lines.append(f"{indent}{var_name} = {value}")
            word_count += 4
        
        elif structure_type == 2:
            # If/else block
            var = f"{_generate_random_word(rng)}_value"
            op = rng.choice(operators[:4])
            lines.append(f"{indent}if {var} {op} {rng.randint(0, 100)}:")
            word_count += 4
            current_indent = min(current_indent + 1, 4)
        
        elif structure_type == 3:
            # For loop
            iter_var = rng.choice(['i', 'j', 'idx', 'item', 'element'])
            collection = f"{_generate_random_word(rng)}_list"
            lines.append(f"{indent}for {iter_var} in {collection}:")
            word_count += 4
            current_indent = min(current_indent + 1, 4)
        
        elif structure_type == 4:
            # Print or return statement
            func = rng.choice(['print', 'return', 'yield'])
            content = f'{_generate_random_word(rng)}_{_generate_random_word(rng)}'
            if func == 'print':
                lines.append(f"{indent}print(f\"{content}: {{{_generate_random_word(rng)}_result}}\")")
            else:
                lines.append(f"{indent}{func} {content}")
            word_count += 3
            # Dedent after return/yield
            if func in ['return', 'yield'] and current_indent > 0:
                current_indent -= 1
        
        else:
            # Comment line
            comment_words = ' '.join([_generate_random_word(rng) for _ in range(rng.randint(3, 8))])
            lines.append(f"{indent}# TODO: {comment_words}")
            word_count += 5
        
        # Randomly dedent
        if current_indent > 0 and rng.random() < 0.2:
            current_indent -= 1
            lines.append("")
    
    lines.append("```")
    
    return '\n'.join(lines)


def _generate_markdown_list(rng: random.Random, target_words: int, turn: int) -> str:
    """
    Generate Markdown bullet point lists.
    
    This stresses tokenizer handling of:
    - List markers (-, *, numbers)
    - Nested indentation
    - Mixed content patterns
    """
    lines = []
    word_count = 0
    
    # Intro text
    intro_words = [_generate_random_word(rng) for _ in range(rng.randint(5, 10))]
    lines.append(f"Query {turn+1} - Key points regarding {' '.join(intro_words[:3])}:")
    word_count += len(intro_words) + 5
    lines.append("")
    
    list_markers = ['-', '*', '•']
    numbered = rng.random() < 0.3
    item_num = 1
    
    while word_count < target_words:
        # Generate list item
        if numbered:
            marker = f"{item_num}."
            item_num += 1
        else:
            marker = rng.choice(list_markers)
        
        # Item content
        item_words = [_generate_random_word(rng) for _ in range(rng.randint(5, 15))]
        
        # Add emphasis sometimes
        if rng.random() < 0.3 and len(item_words) > 3:
            emphasis_idx = rng.randint(0, len(item_words) - 1)
            item_words[emphasis_idx] = f"**{item_words[emphasis_idx]}**"
        
        # Add inline code sometimes
        if rng.random() < 0.25 and len(item_words) > 3:
            code_idx = rng.randint(0, len(item_words) - 1)
            item_words[code_idx] = f"`{item_words[code_idx]}_config`"
        
        lines.append(f"{marker} {' '.join(item_words)}")
        word_count += len(item_words) + 1
        
        # Add nested items sometimes
        if rng.random() < 0.4:
            nested_count = rng.randint(1, 3)
            for _ in range(nested_count):
                nested_words = [_generate_random_word(rng) for _ in range(rng.randint(3, 8))]
                nested_marker = rng.choice(list_markers)
                lines.append(f"  {nested_marker} {' '.join(nested_words)}")
                word_count += len(nested_words) + 1
                if word_count >= target_words:
                    break
    
    # Add conclusion
    conclusion_words = [_generate_random_word(rng) for _ in range(rng.randint(5, 10))]
    lines.append("")
    lines.append(f"Summary: {' '.join(conclusion_words)}.")
    
    return '\n'.join(lines)


# =============================================================================
# PUBLIC DATA GENERATION FUNCTIONS
# =============================================================================

def generate_random_text(word_count: int, unique_seed: Optional[int] = None) -> str:
    """Generate random text that avoids tokenizer caching benefits."""
    if unique_seed is not None:
        rng = random.Random(unique_seed)
        words = [rng.choice(LARGE_VOCABULARY) for _ in range(word_count)]
    else:
        words = random.choices(LARGE_VOCABULARY, k=word_count)
    
    result = []
    for i, word in enumerate(words):
        result.append(word)
        if (i + 1) % 15 == 0:
            result[-1] = result[-1] + random.choice(['.', ',', '!', '?', ';', ':'])
    
    return ' '.join(result)


def generate_synthetic_message(word_count: int = 50) -> str:
    """Generate synthetic text for chat messages."""
    return generate_random_text(word_count)


def generate_chat_conversation(
    num_turns: int, 
    words_per_message: int = 50,
    unique_id: Optional[int] = None,
    use_structured_content: bool = True
) -> List[Dict[str, str]]:
    """
    Generate a synthetic multi-turn chat conversation.
    
    Args:
        num_turns: Number of conversation turns (user + assistant pairs)
        words_per_message: Target words per message
        unique_id: Optional seed for reproducibility
        use_structured_content: If True, alternates between prose, code, and lists
                               to stress tokenizer whitespace/special token handling
    
    Returns:
        List of message dicts with 'role' and 'content' keys
    """
    base_seed = unique_id if unique_id is not None else random.randint(0, 1000000)
    rng = random.Random(base_seed)
    
    messages = [
        {"role": "system", "content": f"You are a helpful AI assistant. Session ID: {base_seed}"}
    ]
    
    for i in range(num_turns):
        if use_structured_content:
            # Rotate through content types: prose -> code -> list
            content_type = i % 3
            
            # User message
            user_content = _generate_structured_content(
                rng, words_per_message, content_type, is_user=True, turn=i
            )
            messages.append({"role": "user", "content": user_content})
            
            # Assistant response (use next content type for variety)
            assistant_type = (content_type + 1) % 3
            assistant_content = _generate_structured_content(
                rng, words_per_message * 2, assistant_type, is_user=False, turn=i
            )
            messages.append({"role": "assistant", "content": assistant_content})
        else:
            # Legacy: simple random text
            user_seed = base_seed * 1000 + i * 2
            user_msg = f"Query {i+1}: {generate_random_text(words_per_message, unique_seed=user_seed)}"
            messages.append({"role": "user", "content": user_msg})
            
            assistant_seed = base_seed * 1000 + i * 2 + 1
            assistant_msg = f"Response {i+1}: {generate_random_text(words_per_message * 2, unique_seed=assistant_seed)}"
            messages.append({"role": "assistant", "content": assistant_msg})
    
    # Final user message
    if use_structured_content:
        # Always code to stress tokenizer at the end
        final_content = _generate_structured_content(
            rng, words_per_message, content_type=1, is_user=True, turn=num_turns
        )
        messages.append({"role": "user", "content": final_content})
    else:
        final_seed = base_seed * 1000 + num_turns * 2
        messages.append({"role": "user", "content": generate_random_text(words_per_message, unique_seed=final_seed)})
    
    return messages


def generate_fixed_token_conversation(
    tokenizer, 
    num_turns: int, 
    target_tokens: int = 100000,
    unique_id: Optional[int] = None,
    max_iterations: int = 5
) -> List[Dict[str, str]]:
    """
    Generate a conversation with fixed total token count but varying turns.
    
    Uses iterative measurement to ensure the final templated + tokenized
    result hits the target token count (within 10% tolerance).
    
    Args:
        tokenizer: HuggingFace tokenizer for measuring token count
        num_turns: Number of conversation turns
        target_tokens: Target total tokens after templating (default 100k to ensure
                       template overhead is negligible across all turn counts)
        unique_id: Optional seed for reproducibility
        max_iterations: Max refinement iterations
    
    Returns:
        List of message dicts that tokenize to ~target_tokens
    """
    base_seed = unique_id if unique_id is not None else random.randint(0, 1000000)
    
    # Calculate initial estimate for words per message
    # Total messages = system + (user + assistant) * turns + final user
    total_messages = num_turns * 2 + 2
    # Rough estimate: target_tokens / 1.5 words, distributed across messages
    words_per_message = max(10, int((target_tokens / 1.5) / total_messages))
    
    # Iteratively refine to hit target
    for iteration in range(max_iterations):
        messages = [
            {"role": "system", "content": f"You are a helpful AI assistant. Session: {base_seed}"}
        ]
        
        for i in range(num_turns):
            user_seed = base_seed * 1000 + i * 2 + iteration
            user_content = generate_random_text(words_per_message, unique_seed=user_seed)
            messages.append({"role": "user", "content": user_content})
            
            assistant_seed = base_seed * 1000 + i * 2 + 1 + iteration
            assistant_content = generate_random_text(words_per_message, unique_seed=assistant_seed)
            messages.append({"role": "assistant", "content": assistant_content})
        
        final_seed = base_seed * 1000 + num_turns * 2 + iteration
        final_content = generate_random_text(words_per_message, unique_seed=final_seed)
        messages.append({"role": "user", "content": final_content})
        
        # Measure actual token count
        templated = tokenizer.apply_chat_template(messages, tokenize=False)
        actual_tokens = len(tokenizer.encode(templated))
        
        # Check if within tolerance (10%)
        error_ratio = abs(actual_tokens - target_tokens) / target_tokens
        if error_ratio < 0.10:
            return messages
        
        # Adjust words_per_message based on error
        adjustment_ratio = target_tokens / actual_tokens
        words_per_message = max(10, int(words_per_message * adjustment_ratio))
    
    # Return best effort after max iterations
    return messages


def generate_target_token_conversation(
    tokenizer, 
    target_tokens: int = 50000,
    unique_id: Optional[int] = None
) -> List[Dict[str, str]]:
    """Generate a conversation that results in approximately target_tokens tokens."""
    base_seed = unique_id if unique_id is not None else random.randint(0, 1000000)
    
    messages = [
        {"role": "system", "content": f"You are a helpful AI assistant. Session: {base_seed}"}
    ]
    
    estimated_words = int(target_tokens / 1.5)
    content = generate_random_text(word_count=estimated_words, unique_seed=base_seed)
    messages.append({"role": "user", "content": content})
    
    templated = tokenizer.apply_chat_template(messages, tokenize=False)
    actual_tokens = len(tokenizer.encode(templated))
    
    if abs(actual_tokens - target_tokens) > target_tokens * 0.15:
        ratio = target_tokens / actual_tokens
        adjusted_words = int(estimated_words * ratio)
        content = generate_random_text(word_count=adjusted_words, unique_seed=base_seed + 1)
        messages[1]["content"] = content
    
    return messages


def generate_large_conversation(
    num_turns: int,
    target_tokens: int,
    unique_id: int
) -> List[Dict[str, str]]:
    """Generate a large conversation with specified token count and turns."""
    # Calculate words per message to hit target tokens
    # Formula: total_words ≈ target_tokens / 1.5 (avg tokens per word)
    # Each turn has ~3 messages worth of content (user + 2x assistant)
    total_words = int(target_tokens / 1.5)
    words_per_turn = total_words // num_turns
    words_per_message = max(50, words_per_turn // 3)
    
    return generate_chat_conversation(
        num_turns=num_turns,
        words_per_message=words_per_message,
        unique_id=unique_id
    )
