import mlflow
import numpy as np
from tqdm import tqdm
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MNIST_mlp")

mnist = fetch_openml(
    "mnist_784",
    version=1,
    as_frame=False
)

X = mnist.data.astype(np.float32) / 255.0
y = mnist.target.astype(np.int64)
X = X[:20000]
y = y[:20000]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



experiments = [
    {"learning_rate": 0.001, "batch_size": 32},
    {"learning_rate": 0.001, "batch_size": 64},
    {"learning_rate": 0.001, "batch_size": 128},

    {"learning_rate": 0.010, "batch_size": 32},
    {"learning_rate": 0.010, "batch_size": 64},
    {"learning_rate": 0.010, "batch_size": 128},
]

results_summary = []

experiment_bar = tqdm(experiments, desc="Experiments", unit="exp")
for i, config in enumerate(experiment_bar, start=1):

    experiment_bar.set_description(f"Experiment {i}/6 {config}")

    with mlflow.start_run(run_name=f"mlp-mnist-{i}"):

        # -------------------------------------------------
        # Hyperparameters
        # -------------------------------------------------
        learning_rate = config["learning_rate"]
        batch_size = config["batch_size"]

        hidden_layers = (128, 64)
        max_epochs = 15
        mlflow.log_param("model", "MLP")
        mlflow.log_param("dataset", "MNIST")
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("hidden_layers", str(hidden_layers))
        mlflow.log_param("max_epochs", max_epochs)

        # -------------------------------------------------
        # MLP, trained one epoch at a time (warm_start) so we
        # can show a live per-epoch progress bar and log
        # train_loss / val_accuracy as we go.
        # -------------------------------------------------
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.1,
            random_state=42,
            stratify=y_train
        )

        model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="adam",
            learning_rate_init=learning_rate,
            batch_size=batch_size,
            max_iter=1,
            random_state=42,
            verbose=False
        )
        classes = np.unique(y_tr)

        train_losses = []
        val_accuracies = []
        best_val_accuracy = -1.0
        epochs_no_improve = 0
        patience = 10

        epoch_bar = tqdm(
            range(1, max_epochs + 1),
            desc="  epochs",
            unit="epoch",
            leave=False
        )
        for epoch in epoch_bar:
            # partial_fit preserves the Adam optimizer's momentum/variance
            # state across calls (unlike calling fit() repeatedly with
            # warm_start, which resets the optimizer every call).
            model.partial_fit(X_tr, y_tr, classes=classes)

            train_loss = model.loss_curve_[-1]
            val_acc = accuracy_score(y_val, model.predict(X_val))

            train_losses.append(train_loss)
            val_accuracies.append(val_acc)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

            epoch_bar.set_postfix(loss=f"{train_loss:.4f}", val_acc=f"{val_acc:.4f}")

            if val_acc > best_val_accuracy + 1e-4:
                best_val_accuracy = val_acc
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        # -------------------------------------------------
        # Final test accuracy
        # -------------------------------------------------
        y_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)

        # Final validation accuracy
        final_val_accuracy = val_accuracies[-1]

        # Best validation accuracy across epochs
        best_val_accuracy = max(val_accuracies)

        # -------------------------------------------------
        # Log final metrics
        # -------------------------------------------------
        mlflow.log_metric(
            "test_accuracy",
            test_accuracy
        )

        mlflow.log_metric(
            "final_val_accuracy",
            final_val_accuracy
        )

        mlflow.log_metric(
            "best_val_accuracy",
            best_val_accuracy
        )

        mlflow.log_metric(
            "epochs_trained",
            len(train_losses)
        )

        tqdm.write(
            f"Experiment {i}/6 {config} -> "
            f"epochs={len(train_losses)} "
            f"train_loss={train_losses[-1]:.4f} "
            f"val_acc={final_val_accuracy:.4f} "
            f"best_val_acc={best_val_accuracy:.4f} "
            f"test_acc={test_accuracy:.4f}"
        )

        results_summary.append({
            **config,
            "epochs_trained": len(train_losses),
            "final_train_loss": train_losses[-1],
            "final_val_accuracy": final_val_accuracy,
            "best_val_accuracy": best_val_accuracy,
            "test_accuracy": test_accuracy,
        })

print("\nAll six experiments completed.\n")
print(f"{'lr':>8} {'batch':>6} {'epochs':>7} {'train_loss':>11} {'val_acc':>8} {'best_val':>9} {'test_acc':>9}")
for r in results_summary:
    print(
        f"{r['learning_rate']:>8} {r['batch_size']:>6} {r['epochs_trained']:>7} "
        f"{r['final_train_loss']:>11.4f} {r['final_val_accuracy']:>8.4f} "
        f"{r['best_val_accuracy']:>9.4f} {r['test_accuracy']:>9.4f}"
    )

best = max(results_summary, key=lambda r: r["test_accuracy"])
print(
    f"\nBest config: lr={best['learning_rate']} batch_size={best['batch_size']} "
    f"-> test_accuracy={best['test_accuracy']:.4f}"
)
print("\nFull metrics/plots: run `mlflow ui --port 5000` (or open http://localhost:5000) "
      "and look at experiment 'MNIST_mlp'.")
