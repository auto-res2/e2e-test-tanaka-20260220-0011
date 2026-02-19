"""
Inference methods: DT-SEACoT (proposed) and CoT-SC (baseline)
"""

import sys
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter
from pathlib import Path
import wandb
from omegaconf import DictConfig, OmegaConf
import hydra

from src.model import T5InferenceModel
from src.preprocess import (
    load_gsm8k_dataset,
    get_direct_prompt,
    get_cot_prompt,
    get_rationale_to_answer_prompt,
    extract_answer_from_generation,
)


def run_dt_seacot_inference(cfg: DictConfig, model: T5InferenceModel, dataset: List[Dict]) -> Dict:
    """
    Run DT-SEACoT inference: Decision-Theoretic, Self-Entailed Adaptive CoT.
    
    Implements:
    1. System-1: Sample N0 direct answers, estimate uncertainty
    2. Value-aware gating: Skip deliberation if confident enough
    3. System-2: Adaptively sample CoT rationales, compute self-entailment weights
    4. Early stopping: Stop when posterior is confident
    
    Args:
        cfg: Hydra config
        model: T5 inference model
        dataset: List of examples to evaluate
        
    Returns:
        Dictionary with metrics and results
    """
    inf_cfg = cfg.inference
    
    # Results storage
    predictions = []
    ground_truths = []
    num_direct_samples_used = []
    num_cot_samples_used = []
    skipped_deliberation = []
    stopped_early = []
    
    for example_idx, example in enumerate(dataset):
        question = example['question']
        true_answer = example['answer']
        
        # System-1: Direct answer sampling
        direct_prompt = get_direct_prompt(question)
        direct_generations = model.generate(
            [direct_prompt] * inf_cfg.n0_direct_samples,
            max_new_tokens=inf_cfg.direct_max_tokens,
            temperature=inf_cfg.direct_temperature,
            do_sample=True,
            num_return_sequences=1,
        )
        
        # Extract answers from direct generations
        direct_answers = []
        direct_answer_texts = []
        for gen_list in direct_generations:
            for gen_text in gen_list:
                direct_answer_texts.append(gen_text)
                answer = extract_answer_from_generation(gen_text)
                if answer is not None:
                    direct_answers.append(answer)
        
        # Estimate System-1 uncertainty
        if len(direct_answers) == 0:
            # No valid answers, must deliberate
            max_prob = 0.0
            entropy = 10.0
        else:
            answer_counts = Counter(direct_answers)
            total = len(direct_answers)
            probs = {ans: count / total for ans, count in answer_counts.items()}
            max_prob = max(probs.values())
            
            # Compute entropy
            prob_values = np.array(list(probs.values()))
            entropy = -np.sum(prob_values * np.log(prob_values + 1e-10))
        
        # Value-aware gating decision
        should_deliberate = (
            max_prob < inf_cfg.gate_confidence_threshold or
            entropy > inf_cfg.gate_entropy_threshold
        )
        
        if not should_deliberate and len(direct_answers) > 0:
            # Return most common direct answer
            most_common = Counter(direct_answers).most_common(1)[0][0]
            predictions.append(most_common)
            ground_truths.append(true_answer)
            num_direct_samples_used.append(len(direct_answers))
            num_cot_samples_used.append(0)
            skipped_deliberation.append(1)
            stopped_early.append(0)
            continue
        
        # System-2: CoT deliberation with self-entailment weighting
        skipped_deliberation.append(0)
        
        # Initialize answer weights (in log-space for numerical stability)
        answer_weights = {}  # answer -> log-weight
        answer_samples = {}  # answer -> list of (rationale, generation_text)
        
        num_cot_generated = 0
        early_stopped = False
        
        for cot_idx in range(inf_cfg.k_max_cot_samples):
            # Generate CoT sample
            cot_prompt = get_cot_prompt(question)
            cot_generations = model.generate(
                [cot_prompt],
                max_new_tokens=inf_cfg.cot_max_tokens,
                temperature=inf_cfg.cot_temperature,
                do_sample=True,
                num_return_sequences=1,
            )
            
            cot_text = cot_generations[0][0]
            cot_answer = extract_answer_from_generation(cot_text)
            
            if cot_answer is None:
                # Invalid generation, skip
                num_cot_generated += 1
                continue
            
            # Compute self-entailment likelihood ratio
            if inf_cfg.use_self_entailment:
                # p(answer | question) - direct baseline
                answer_str = str(cot_answer)
                ll_direct = model.compute_log_likelihood(
                    [direct_prompt],
                    [answer_str]
                )[0]
                
                # p(answer | question, rationale) - with reasoning
                r2a_prompt = get_rationale_to_answer_prompt(question, cot_text)
                ll_with_rationale = model.compute_log_likelihood(
                    [r2a_prompt],
                    [answer_str]
                )[0]
                
                # Self-entailment likelihood ratio
                delta_ll = ll_with_rationale - ll_direct
                evidence_weight = delta_ll
            else:
                # Uniform weighting (standard self-consistency)
                evidence_weight = 0.0
            
            # Update answer weights
            if cot_answer not in answer_weights:
                answer_weights[cot_answer] = evidence_weight
                answer_samples[cot_answer] = [(cot_text, cot_text)]
            else:
                # Log-sum-exp for adding probabilities in log-space
                old_weight = answer_weights[cot_answer]
                answer_weights[cot_answer] = np.logaddexp(old_weight, evidence_weight)
                answer_samples[cot_answer].append((cot_text, cot_text))
            
            num_cot_generated += 1
            
            # Early stopping check
            if inf_cfg.early_stop_enabled and num_cot_generated >= 3:
                # Convert log-weights to probabilities
                log_weights = np.array(list(answer_weights.values()))
                weights = np.exp(log_weights - np.max(log_weights))  # Numerical stability
                total_weight = np.sum(weights)
                
                if total_weight > 0:
                    probs = weights / total_weight
                    max_posterior = np.max(probs)
                    
                    if max_posterior > inf_cfg.early_stop_posterior_threshold:
                        early_stopped = True
                        break
        
        # Select final answer
        if len(answer_weights) == 0:
            # Fallback to direct answer if available
            if len(direct_answers) > 0:
                final_answer = Counter(direct_answers).most_common(1)[0][0]
            else:
                final_answer = 0.0  # Default fallback
        else:
            # Return answer with highest weight
            final_answer = max(answer_weights.items(), key=lambda x: x[1])[0]
        
        predictions.append(final_answer)
        ground_truths.append(true_answer)
        num_direct_samples_used.append(len(direct_answers))
        num_cot_samples_used.append(num_cot_generated)
        stopped_early.append(1 if early_stopped else 0)
    
    # Compute metrics
    correct = sum(abs(pred - gt) < 1e-6 for pred, gt in zip(predictions, ground_truths))
    accuracy = correct / len(predictions) if len(predictions) > 0 else 0.0
    
    avg_direct_samples = np.mean(num_direct_samples_used)
    avg_cot_samples = np.mean(num_cot_samples_used)
    total_samples_per_question = avg_direct_samples + avg_cot_samples
    skip_rate = np.mean(skipped_deliberation)
    early_stop_rate = np.mean(stopped_early)
    
    # Estimate token counts
    avg_tokens_per_question = (
        avg_direct_samples * inf_cfg.direct_max_tokens +
        avg_cot_samples * inf_cfg.cot_max_tokens
    )
    
    return {
        'accuracy': accuracy,
        'num_correct': correct,
        'num_total': len(predictions),
        'avg_direct_samples': avg_direct_samples,
        'avg_cot_samples': avg_cot_samples,
        'total_samples_per_question': total_samples_per_question,
        'skip_rate': skip_rate,
        'early_stop_rate': early_stop_rate,
        'avg_tokens_per_question': avg_tokens_per_question,
        'predictions': predictions,
        'ground_truths': ground_truths,
    }


def run_cot_sc_inference(cfg: DictConfig, model: T5InferenceModel, dataset: List[Dict]) -> Dict:
    """
    Run baseline CoT-SC inference: Fixed-K Chain-of-Thought with Self-Consistency.
    
    Always samples K CoT rationales and returns plurality vote.
    No gating, no self-entailment weighting, no early stopping.
    
    Args:
        cfg: Hydra config
        model: T5 inference model
        dataset: List of examples to evaluate
        
    Returns:
        Dictionary with metrics and results
    """
    inf_cfg = cfg.inference
    
    predictions = []
    ground_truths = []
    
    for example in dataset:
        question = example['question']
        true_answer = example['answer']
        
        # Generate K CoT samples
        cot_prompt = get_cot_prompt(question)
        cot_generations = model.generate(
            [cot_prompt] * inf_cfg.k_cot_samples,
            max_new_tokens=inf_cfg.cot_max_tokens,
            temperature=inf_cfg.cot_temperature,
            do_sample=True,
            num_return_sequences=1,
        )
        
        # Extract answers
        cot_answers = []
        for gen_list in cot_generations:
            for gen_text in gen_list:
                answer = extract_answer_from_generation(gen_text)
                if answer is not None:
                    cot_answers.append(answer)
        
        # Plurality vote
        if len(cot_answers) == 0:
            final_answer = 0.0  # Fallback
        else:
            final_answer = Counter(cot_answers).most_common(1)[0][0]
        
        predictions.append(final_answer)
        ground_truths.append(true_answer)
    
    # Compute metrics
    correct = sum(abs(pred - gt) < 1e-6 for pred, gt in zip(predictions, ground_truths))
    accuracy = correct / len(predictions) if len(predictions) > 0 else 0.0
    
    # Token usage
    avg_tokens_per_question = inf_cfg.k_cot_samples * inf_cfg.cot_max_tokens
    
    return {
        'accuracy': accuracy,
        'num_correct': correct,
        'num_total': len(predictions),
        'avg_cot_samples': inf_cfg.k_cot_samples,
        'total_samples_per_question': inf_cfg.k_cot_samples,
        'avg_tokens_per_question': avg_tokens_per_question,
        'predictions': predictions,
        'ground_truths': ground_truths,
    }


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    """
    Main inference script (invoked by main.py as subprocess).
    """
    print(f"Starting inference run: {cfg.run.run_id}")
    print(f"Method: {cfg.run.method_name}")
    print(f"Mode: {cfg.mode}")
    
    # Load dataset
    split_name = 'val' if cfg.mode == 'sanity_check' or (hasattr(cfg, 'use_val_split') and cfg.use_val_split) else 'test'
    datasets = load_gsm8k_dataset(
        cache_dir=cfg.dataset.cache_dir,
        max_samples=cfg.dataset.max_samples,
        val_split=cfg.dataset.val_split,
        test_split=cfg.dataset.test_split,
    )
    dataset = datasets[split_name]
    
    # In sanity_check mode, use only first 10 samples
    if cfg.mode == 'sanity_check':
        dataset = dataset[:10]
        print(f"Sanity check mode: using {len(dataset)} samples")
    
    # Initialize model
    print(f"Loading model: {cfg.model.name}")
    model = T5InferenceModel(
        model_name=cfg.model.name,
        device=cfg.model.device,
        dtype=cfg.model.dtype,
        cache_dir=cfg.model.cache_dir,
    )
    
    # Initialize WandB (unless disabled)
    if cfg.wandb.mode != 'disabled':
        # In sanity_check mode, use separate project
        project = cfg.wandb.project
        if cfg.mode == 'sanity_check':
            project = f"{project}-sanity"
        
        wandb.init(
            entity=cfg.wandb.entity,
            project=project,
            id=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
            mode=cfg.wandb.mode,
        )
        print(f"WandB initialized: {wandb.run.url}")
    
    # Run inference based on method
    if cfg.inference.method == 'dt_seacot':
        results = run_dt_seacot_inference(cfg, model, dataset)
    elif cfg.inference.method == 'cot_sc':
        results = run_cot_sc_inference(cfg, model, dataset)
    else:
        raise ValueError(f"Unknown inference method: {cfg.inference.method}")
    
    # Log metrics
    print(f"\nResults for {cfg.run.run_id}:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Correct: {results['num_correct']}/{results['num_total']}")
    if 'avg_tokens_per_question' in results:
        print(f"  Avg tokens/question: {results['avg_tokens_per_question']:.1f}")
    if 'skip_rate' in results:
        print(f"  Skip rate: {results['skip_rate']:.2%}")
    if 'early_stop_rate' in results:
        print(f"  Early stop rate: {results['early_stop_rate']:.2%}")
    
    # Log to WandB
    if cfg.wandb.mode != 'disabled':
        wandb.log(results)
        wandb.summary.update(results)
        wandb.finish()
    
    # Save results
    results_dir = Path(cfg.results_dir) / cfg.run.run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = results_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    # Sanity validation
    if cfg.mode == 'sanity_check':
        perform_sanity_validation(results, cfg)
    
    return results


def perform_sanity_validation(results: Dict, cfg: DictConfig):
    """
    Perform sanity validation checks and print verdict.
    """
    passed = True
    reasons = []
    
    # Check: At least 5 samples processed
    if results['num_total'] < 5:
        passed = False
        reasons.append("insufficient_samples")
    
    # Check: All metrics are finite
    if not np.isfinite(results['accuracy']):
        passed = False
        reasons.append("non_finite_metrics")
    
    # Check: Not all predictions are identical (unless all ground truths are identical)
    predictions = results.get('predictions', [])
    ground_truths = results.get('ground_truths', [])
    
    if len(predictions) > 1 and len(set(predictions)) == 1:
        # All predictions are identical - check if this is reasonable
        if len(set(ground_truths)) > 1:
            # Ground truths vary but predictions don't - suspicious
            passed = False
            reasons.append("all_predictions_identical")
    
    # Print summary
    summary = {
        'samples': results['num_total'],
        'accuracy': results['accuracy'],
        'num_correct': results['num_correct'],
    }
    
    if 'avg_cot_samples' in results:
        summary['avg_cot_samples'] = results['avg_cot_samples']
    
    print(f"SANITY_VALIDATION_SUMMARY: {json.dumps(summary)}")
    
    # Print verdict
    if passed:
        print("SANITY_VALIDATION: PASS")
    else:
        reason_str = ",".join(reasons)
        print(f"SANITY_VALIDATION: FAIL reason={reason_str}")


if __name__ == "__main__":
    main()
