# Hyperparameter Tuning with HypderDrive

Take a look at the definition of the hyperdrive step in `pipeline_slave.py`.

```
    est = Estimator(
        source_directory=script_folder,
        compute_target=gpu_compute_target,
        entry_script="train.py",
        node_count=1,
        environment_definition=env,
    )

    ps = BayesianParameterSampling(
        {
            "--batch_size": choice(1, 2, 4, 10),
            "--filter_sizes": choice("3, 3, 3", "4, 4, 4", "5, 5, 5"),
            "--stack_sizes": choice(
                "48, 96, 192", "36, 72, 144", "12, 24, 48"
            ),
            "--learning_rate": uniform(1e-6, 1e-3),
            "--lr_decay": uniform(1e-9, 1e-2),
            "--freeze_layers": choice(
                "0, 1, 2", "1, 2, 3", "0, 1", "1, 2", "2, 3", "0", "3"
            ),
            "--transfer_learning": choice("True", "False"),
        }
    )

    hdc = HyperDriveConfig(
        estimator=est,
        hyperparameter_sampling=ps,
        primary_metric_name="val_loss",
        primary_metric_goal=PrimaryMetricGoal.MINIMIZE,
        max_total_runs=1,
        max_concurrent_runs=1,
        max_duration_minutes=60 * 6,
    )

    train_prednet = HyperDriveStep(
        "train_w_hyperdrive",
        hdc,
        estimator_entry_script_arguments=[
            "--preprocessed_data",
            preprocessed_data,
            "--remote_execution",
            "--dataset",
            dataset,
            # "--hd_child_cwd",
            # hd_child_cwd
        ],
        inputs=[preprocessed_data],
        outputs=[hd_child_cwd],
        metrics_output=data_metrics,
        allow_reuse=True,
    )
```

What this does is tell hyperdrive to explore whether transfer_learning benefits training. It also explores which layers to freeze during transfer learning. 

If transfer_learning is performed, the `train.py` script looks for an existing model in the model registry, downloads it, and starts retraining it for the current dataset. 
