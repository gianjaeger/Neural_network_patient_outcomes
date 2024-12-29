import torch
import torch.nn as nn
import torch.optim as optim
import random

def random_search_hyperparameters(X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, input_size, num_epochs=20, n_trials=20):

    # Define the search space
    search_space = {
        "hidden_layer_1_sizes": [16, 32, 64, 128],
        "hidden_layer_2_sizes": [16, 32, 64, 128],
        "learning_rates": [0.1, 0.01, 0.001],
        "batch_sizes": [16, 32, 64],
        "activation_functions": [nn.ReLU, nn.Sigmoid],
    }

    # Track the best model and metrics
    best_model = None
    best_val_balanced_accuracy = 0.0
    best_hyperparams = {}

    # Random search
    for trial in range(n_trials):
        print(f"Trial {trial + 1}/{n_trials}")

        # Randomly sample hyperparameters
        hidden_layer_1_size = random.choice(search_space["hidden_layer_1_sizes"])
        hidden_layer_2_size = random.choice(search_space["hidden_layer_2_sizes"])
        learning_rate = random.choice(search_space["learning_rates"])
        batch_size = random.choice(search_space["batch_sizes"])
        activation_function = random.choice(search_space["activation_functions"])

        # Define the model
        model = nn.Sequential(
            nn.Linear(input_size, hidden_layer_1_size),
            activation_function(),
            nn.Linear(hidden_layer_1_size, hidden_layer_2_size),
            activation_function(),
            nn.Linear(hidden_layer_2_size, 1),
        )

        # Define loss function and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(num_epochs):
            permutation = torch.randperm(X_train_tensor.size(0))
            X_train_tensor = X_train_tensor[permutation]
            y_train_tensor = y_train_tensor[permutation]

            for i in range(0, X_train_tensor.size(0), batch_size):
                X_batch = X_train_tensor[i:i + batch_size]
                y_batch = y_train_tensor[i:i + batch_size]

                logits = model(X_batch)
                loss = criterion(logits, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_tensor)
            val_probabilities = torch.sigmoid(val_logits)
            val_predictions = (val_probabilities > 0.5).float()

            val_recall = (
                ((val_predictions == 1) & (y_val_tensor == 1)).sum().item() /
                ((y_val_tensor == 1).sum().item() + 1e-8)
            )
            val_specificity = (
                ((val_predictions == 0) & (y_val_tensor == 0)).sum().item() /
                ((y_val_tensor == 0).sum().item() + 1e-8)
            )
            val_balanced_accuracy = (val_recall + val_specificity) / 2

        # Track the best model
        if val_balanced_accuracy > best_val_balanced_accuracy:
            best_val_balanced_accuracy = val_balanced_accuracy
            best_model = model
            best_hyperparams = {
                "hidden_layer_1_size": hidden_layer_1_size,
                "hidden_layer_2_size": hidden_layer_2_size,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "activation_function": activation_function.__name__,
            }

        print(f"Validation Balanced Accuracy: {val_balanced_accuracy:.4f}")

    return {
        "best_model": best_model,
        "best_hyperparams": best_hyperparams,
        "best_val_balanced_accuracy": best_val_balanced_accuracy,
    }
