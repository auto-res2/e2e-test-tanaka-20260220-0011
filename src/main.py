"""
Main orchestrator for inference experiments.
Handles mode overrides and subprocess invocation.
"""

import sys
import subprocess
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import optuna


def run_hyperparameter_tuning(cfg: DictConfig) -> DictConfig:
    """
    Run Optuna hyperparameter tuning on validation set.
    
    Args:
        cfg: Hydra config
        
    Returns:
        Updated config with best hyperparameters
    """
    print(f"\nStarting hyperparameter tuning with Optuna...")
    print(f"Study: {cfg.optuna.study_name}")
    print(f"Trials: {cfg.optuna.n_trials}")
    
    # Import inference functions
    from src.model import T5InferenceModel
    from src.preprocess import load_gsm8k_dataset
    from src.inference import run_dt_seacot_inference, run_cot_sc_inference
    
    # Load validation dataset
    datasets = load_gsm8k_dataset(
        cache_dir=cfg.dataset.cache_dir,
        max_samples=cfg.dataset.max_samples,
        val_split=cfg.dataset.val_split,
        test_split=cfg.dataset.test_split,
    )
    val_dataset = datasets['val']
    
    # Load model once (shared across trials)
    print(f"Loading model: {cfg.model.name}")
    model = T5InferenceModel(
        model_name=cfg.model.name,
        device=cfg.model.device,
        dtype=cfg.model.dtype,
        cache_dir=cfg.model.cache_dir,
    )
    
    # Define objective function
    def objective(trial):
        # Create trial config by updating inference params
        trial_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        
        # Sample hyperparameters based on search space
        for param_name, param_spec in cfg.optuna.search_space.items():
            if param_spec.type == 'float':
                value = trial.suggest_float(param_name, param_spec.low, param_spec.high)
            elif param_spec.type == 'int':
                value = trial.suggest_int(param_name, param_spec.low, param_spec.high)
            elif param_spec.type == 'categorical':
                value = trial.suggest_categorical(param_name, param_spec.choices)
            else:
                raise ValueError(f"Unknown parameter type: {param_spec.type}")
            
            # Update config (navigate nested structure)
            OmegaConf.update(trial_cfg, f"inference.{param_name}", value)
        
        # Run inference on validation set
        if trial_cfg.inference.method == 'dt_seacot':
            results = run_dt_seacot_inference(trial_cfg, model, val_dataset)
        elif trial_cfg.inference.method == 'cot_sc':
            results = run_cot_sc_inference(trial_cfg, model, val_dataset)
        else:
            raise ValueError(f"Unknown method: {trial_cfg.inference.method}")
        
        # Return metric to optimize
        metric_value = results[cfg.optuna.metric]
        
        print(f"  Trial {trial.number}: {cfg.optuna.metric}={metric_value:.4f}")
        
        return metric_value
    
    # Create and run study
    study = optuna.create_study(
        study_name=cfg.optuna.study_name,
        direction=cfg.optuna.direction,
        load_if_exists=True,
    )
    
    study.optimize(objective, n_trials=cfg.optuna.n_trials)
    
    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    print(f"\nHyperparameter tuning complete!")
    print(f"Best {cfg.optuna.metric}: {best_value:.4f}")
    print(f"Best parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Update config with best parameters
    for param_name, value in best_params.items():
        OmegaConf.update(cfg, f"inference.{param_name}", value)
    
    return cfg


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    """
    Main orchestrator for a single run.
    Applies mode overrides and invokes inference.py as subprocess.
    """
    print(f"=" * 80)
    print(f"Run ID: {cfg.run.run_id}")
    print(f"Method: {cfg.run.method_name}")
    print(f"Mode: {cfg.mode}")
    print(f"=" * 80)
    
    # Apply mode-specific overrides
    if cfg.mode == 'sanity_check':
        print("\nApplying sanity_check mode overrides...")
        
        # Reduce dataset size
        cfg.dataset.max_samples = 10
        cfg.dataset.val_split = 5
        cfg.dataset.test_split = 5
        
        # Disable or reduce optuna trials
        if hasattr(cfg, 'optuna') and cfg.optuna.enabled:
            cfg.optuna.n_trials = 0
            cfg.optuna.enabled = False
        
        # Ensure WandB is online for sanity check
        cfg.wandb.mode = 'online'
        
        print("  - Reduced dataset to 10 samples")
        print("  - Disabled hyperparameter tuning")
        print("  - WandB mode: online")
    
    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: ConfigAttributeError: Key 'use_val_split' is not in struct
    # [CAUSE]: Lines 146 and 149 tried to set cfg.use_val_split, but OmegaConf struct mode
    #          prevents adding new keys that don't exist in the config schema.
    # [FIX]: Track the split selection locally without modifying the config. Use a local
    #        variable that's initialized before the if block so it's available in all code paths.
    #
    # [OLD CODE]:
    # if hasattr(cfg, 'optuna') and cfg.optuna.enabled and cfg.optuna.n_trials > 0:
    #     cfg.use_val_split = True  # Use validation set for tuning
    #     cfg = run_hyperparameter_tuning(cfg)
    #     print("\nProceeding to final evaluation with tuned hyperparameters...")
    #     cfg.use_val_split = False  # Switch to test set for final run
    #
    # [NEW CODE]:
    # Run hyperparameter tuning if enabled (track split choice locally)
    use_val_split_for_final_run = False  # Initialize to False (use test split by default)
    if hasattr(cfg, 'optuna') and cfg.optuna.enabled and cfg.optuna.n_trials > 0:
        # Use validation set for tuning
        cfg = run_hyperparameter_tuning(cfg)
        print("\nProceeding to final evaluation with tuned hyperparameters...")
        # After tuning, we want to use test set for final evaluation
        use_val_split_for_final_run = False
    
    # Run inference directly (no subprocess needed since it's a single script)
    print("\nRunning inference...")
    
    # Import and run inference main
    from src.inference import main as inference_main
    
    # Override sys.argv to pass config
    original_argv = sys.argv.copy()
    sys.argv = ['src.inference']
    
    try:
        # Note: Hydra will reinitialize from the current config
        # We need to invoke the inference function directly
        from src.model import T5InferenceModel
        from src.preprocess import load_gsm8k_dataset
        from src.inference import run_dt_seacot_inference, run_cot_sc_inference, perform_sanity_validation
        import wandb
        import json
        
        # Load dataset
        # In sanity_check mode or when tuning just finished, use val; otherwise use test
        split_name = 'val' if cfg.mode == 'sanity_check' or use_val_split_for_final_run else 'test'
        datasets = load_gsm8k_dataset(
            cache_dir=cfg.dataset.cache_dir,
            max_samples=cfg.dataset.max_samples,
            val_split=cfg.dataset.val_split,
            test_split=cfg.dataset.test_split,
        )
        dataset = datasets[split_name]
        
        # In sanity_check mode, use only first 10 samples
        if cfg.mode == 'sanity_check':
            dataset = dataset[:min(10, len(dataset))]
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
            print(f"WandB run URL: {wandb.run.url}")
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
        
    finally:
        sys.argv = original_argv
    
    print(f"\n{'=' * 80}")
    print(f"Run completed: {cfg.run.run_id}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
