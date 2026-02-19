"""
Model loading and inference utilities for T5-based models
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict, Tuple
import numpy as np


class T5InferenceModel:
    """
    Wrapper for T5 model inference with support for:
    - Generation (sampling)
    - Teacher-forced log-likelihood computation
    """
    
    def __init__(
        self,
        model_name: str = "google/flan-t5-large",
        device: str = "cuda",
        dtype: str = "bfloat16",
        cache_dir: str = ".cache/"
    ):
        """
        Initialize the T5 model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on
            dtype: Data type (bfloat16, float16, float32)
            cache_dir: Cache directory for model weights
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        
        # Set dtype
        if dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        elif dtype == "float16":
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        # Load model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=self.torch_dtype,
            device_map=device if device == "auto" else None
        )
        
        if device != "auto":
            self.model = self.model.to(device)
        
        self.model.eval()
    
    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        do_sample: bool = True,
        num_return_sequences: int = 1,
    ) -> List[List[str]]:
        """
        Generate text from prompts.
        
        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to sample (True) or use greedy decoding (False)
            num_return_sequences: Number of sequences to generate per prompt
            
        Returns:
            List of lists of generated texts (outer list per prompt, inner list per sequence)
        """
        # Tokenize
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Reshape to [num_prompts, num_return_sequences]
        result = []
        for i in range(len(prompts)):
            start_idx = i * num_return_sequences
            end_idx = start_idx + num_return_sequences
            result.append(generated_texts[start_idx:end_idx])
        
        return result
    
    def compute_log_likelihood(
        self,
        prompts: List[str],
        completions: List[str]
    ) -> List[float]:
        """
        Compute log p(completion | prompt) via teacher forcing.
        Used for self-entailment likelihood ratio computation.
        
        Args:
            prompts: List of input prompts
            completions: List of target completions (one per prompt)
            
        Returns:
            List of log-likelihoods (one per prompt-completion pair)
        """
        assert len(prompts) == len(completions), "Must have same number of prompts and completions"
        
        # Tokenize inputs
        input_ids = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).input_ids.to(self.device)
        
        # Tokenize targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                completions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).input_ids.to(self.device)
        
        # Replace padding token id with -100 so they're ignored in loss
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # Compute log-likelihood via teacher forcing
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                labels=labels
            )
            
            # Get per-token log-likelihoods
            logits = outputs.logits  # [batch, seq_len, vocab]
            
            # Compute log probabilities
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            
            # Gather log-probs for actual tokens
            # Shift labels to align with logits
            batch_size = labels.shape[0]
            seq_len = labels.shape[1]
            
            total_log_liks = []
            for i in range(batch_size):
                label_seq = labels[i]
                log_prob_seq = log_probs[i]
                
                # Sum log-probs for non-padding tokens
                total_log_lik = 0.0
                num_tokens = 0
                for j in range(seq_len):
                    if label_seq[j] != -100:
                        token_id = label_seq[j]
                        token_log_prob = log_prob_seq[j, token_id].item()
                        total_log_lik += token_log_prob
                        num_tokens += 1
                
                total_log_liks.append(total_log_lik)
        
        return total_log_liks
    
    def compute_entropy(self, log_probs: List[float]) -> float:
        """
        Compute entropy of a discrete probability distribution.
        
        Args:
            log_probs: List of log probabilities (must sum to 0 in log-space, i.e., sum to 1 in prob-space)
            
        Returns:
            Entropy in nats
        """
        probs = np.exp(log_probs)
        probs = probs / probs.sum()  # Normalize
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return entropy
