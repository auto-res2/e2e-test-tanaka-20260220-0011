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
    # [VALIDATOR FIX - Attempt 2]
    # [PROBLEM]: 4.67% accuracy suggests answer extraction may miss valid answers
    # [CAUSE]: Current patterns may not match all formats FLAN-T5 outputs
    # [FIX]: Add more extraction patterns, prioritize later occurrences, and be more
    #        flexible with whitespace and formatting
    #
    # [OLD CODE]: (see below)
    #
    # [NEW CODE]:
    
    text_lower = text.lower()
    
    # Try multiple patterns in priority order (most specific to least specific)
    patterns = [
        r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)',  # GSM8K format: #### X
        r'(?:the answer is|answer is|answer:)\s*\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)',  # "the answer is X" or "answer: X"
        r'(?:therefore|thus|hence|so),?\s+(?:the answer is)?\s*\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)',  # "Therefore, X" or "Thus the answer is X"
        r'=\s*\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*\.?\s*$',  # "= X" at end
        r'is\s+\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*\.?\s*$',  # "is X" at end
    ]
    
    for pattern in patterns:
        # Find all matches and take the last one (final answer is typically last)
        matches = list(re.finditer(pattern, text_lower))
        if matches:
            num_str = matches[-1].group(1).replace(',', '')
            try:
                return float(num_str)
            except ValueError:
                continue
    
    # Fallback: find all numbers and take the last one
    # This is risky but better than returning None
    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
    if numbers:
        # Try last few numbers in reverse order
        for num_str in reversed(numbers[-3:]):
            try:
                return float(num_str.replace(',', ''))
            except ValueError:
                continue
    
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
    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: Direct answers may also be extracted incorrectly
    # [CAUSE]: Generic "Answer:" prompt doesn't enforce format that answer extraction expects
    # [FIX]: Add explicit instruction to output just the number, matching GSM8K conventions
    #
    # [OLD CODE]:
    # return f"""Solve this math problem and give only the final numeric answer.
    #
    # Question: {question}
    #
    # Answer:"""
    #
    # [NEW CODE]:
    
    # [VALIDATOR FIX - Attempt 2]
    # [PROBLEM]: Low accuracy (4.67%) suggests poor reasoning and answer format
    # [CAUSE]: FLAN-T5 works better with concise Q/A format from its instruction tuning
    # [FIX]: Use minimal Q/A format that FLAN-T5 recognizes from training
    #
    # [OLD CODE]:
    # return f"""Solve this math problem and give only the final numeric answer.
    #
    # Question: {question}
    #
    # The answer is:"""
    #
    # [NEW CODE]:
    return f"""Q: {question}
A:"""


def get_cot_prompt(question: str) -> str:
    """
    Create a Chain-of-Thought prompt (System-2) with reasoning.
    
    Args:
        question: The math problem
        
    Returns:
        Prompt string for CoT reasoning
    """
    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: 0% accuracy on sanity check (0/5 correct answers)
    # [CAUSE]: The prompt "Let's solve this step by step:" doesn't guide FLAN-T5 to
    #          output a clear final answer. The model generates reasoning but may not
    #          conclude with a marked final answer, causing answer extraction to pick up
    #          intermediate calculation steps instead of the final result.
    # [FIX]: Add explicit instruction to output final answer after "####" to match
    #        GSM8K format. FLAN-T5 is trained on GSM8K format so it should recognize
    #        this pattern and generate accordingly.
    #
    # [OLD CODE]:
    # return f"""Solve this math problem step by step.
    #
    # Question: {question}
    #
    # Let's solve this step by step:"""
    #
    # [NEW CODE]:
    
    # [VALIDATOR FIX - Attempt 2]
    # [PROBLEM]: 4.67% accuracy on test set (7/150), best tuning accuracy 6% (3/50)
    # [CAUSE]: FLAN-T5-large struggles with multi-step arithmetic in GSM8K. The previous
    #          prompt format may not be optimal for FLAN-T5's instruction-following training.
    #          FLAN-T5 was fine-tuned with specific prompt formats that work better.
    # [FIX]: Use a more explicit FLAN-T5-style instruction format that emphasizes showing
    #        the calculation steps and explicitly ending with "The answer is: [number]" which
    #        matches FLAN-T5's training better than the #### format.
    #
    # [OLD CODE]:
    # return f"""Solve this math problem step by step. Show your work and put your final numeric answer after ####.
    #
    # Question: {question}
    #
    # Let's solve this step by step:"""
    #
    # [NEW CODE]:
    return f"""Q: {question}
A: Let's think step by step."""


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
