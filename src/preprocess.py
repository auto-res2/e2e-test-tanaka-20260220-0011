"""
Dataset loading and preprocessing for GSM8K
"""

import re
from typing import Dict, List, Optional
from datasets import load_dataset
from pathlib import Path


def extract_numeric_answer(answer_text: str) -> Optional[float]:
    """
    Extract the numeric answer from GSM8K answer text.
    GSM8K answers typically end with "#### {number}"
    
    Args:
        answer_text: Raw answer string from GSM8K
        
    Returns:
        Numeric answer as float, or None if parsing fails
    """
    # GSM8K format: "#### 42" at the end
    match = re.search(r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', answer_text)
    if match:
        # Remove commas from numbers like "1,000"
        num_str = match.group(1).replace(',', '')
        try:
            return float(num_str)
        except ValueError:
            return None
    return None


def extract_answer_from_generation(text: str) -> Optional[float]:
    """
    Extract numeric answer from model generation.
    Looks for various patterns like "The answer is 42", "= 42", etc.
    
    Args:
        text: Generated text from the model
        
    Returns:
        Extracted numeric answer, or None if not found
    """
    # Try multiple patterns
    patterns = [
        r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)',  # GSM8K format
        r'(?:the answer is|answer:)\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)',  # "the answer is X"
        r'=\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*$',  # "= X" at end
        r'\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*$',  # Just number at end
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            num_str = match.group(1).replace(',', '')
            try:
                return float(num_str)
            except ValueError:
                continue
    
    # Fallback: find last number in text
    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
    if numbers:
        try:
            return float(numbers[-1].replace(',', ''))
        except ValueError:
            pass
    
    return None


def load_gsm8k_dataset(
    cache_dir: str = ".cache/",
    max_samples: int = 200,
    val_split: int = 50,
    test_split: int = 150,
    seed: int = 42
) -> Dict[str, List[Dict]]:
    """
    Load GSM8K dataset and split into validation (for tuning) and test sets.
    
    Args:
        cache_dir: Directory to cache the dataset
        max_samples: Total number of samples to use from test set
        val_split: Number of samples for validation (hyperparameter tuning)
        test_split: Number of samples for final test evaluation
        seed: Random seed for reproducible splitting
        
    Returns:
        Dictionary with 'val' and 'test' splits, each containing list of examples
    """
    # Load GSM8K test split
    dataset = load_dataset(
        "gsm8k",
        "main",
        split="test",
        cache_dir=cache_dir
    )
    
    # Take only the first max_samples
    dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    # Shuffle with seed for reproducibility
    dataset = dataset.shuffle(seed=seed)
    
    # Process examples
    processed_examples = []
    for example in dataset:
        question = example['question']
        answer_text = example['answer']
        numeric_answer = extract_numeric_answer(answer_text)
        
        if numeric_answer is not None:
            processed_examples.append({
                'question': question,
                'answer_text': answer_text,
                'answer': numeric_answer,
            })
    
    # Split into val and test
    val_examples = processed_examples[:val_split]
    test_examples = processed_examples[val_split:val_split + test_split]
    
    return {
        'val': val_examples,
        'test': test_examples,
    }


def get_direct_prompt(question: str) -> str:
    """
    Create a direct answer prompt (System-1) without reasoning.
    
    Args:
        question: The math problem
        
    Returns:
        Prompt string for direct answer
    """
    return f"""Solve this math problem and give only the final numeric answer.

Question: {question}

Answer:"""


def get_cot_prompt(question: str) -> str:
    """
    Create a Chain-of-Thought prompt (System-2) with reasoning.
    
    Args:
        question: The math problem
        
    Returns:
        Prompt string for CoT reasoning
    """
    return f"""Solve this math problem step by step.

Question: {question}

Let's solve this step by step:"""


def get_rationale_to_answer_prompt(question: str, rationale: str) -> str:
    """
    Create a prompt for teacher-forcing the answer given a rationale.
    Used to compute p(answer | question, rationale) for self-entailment.
    
    Args:
        question: The math problem
        rationale: The generated reasoning
        
    Returns:
        Prompt string for conditioning on rationale
    """
    return f"""Given the reasoning below, what is the final numeric answer?

Question: {question}

Reasoning: {rationale}

Therefore, the answer is:"""
