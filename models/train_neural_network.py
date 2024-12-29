import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

class NeuralNetworkTrainer:
    def __init__(self, input_size, pos_weight=4.5, num_epochs=20, batch_size=32, lr=0.01):
        self.input_size = input_size
        self.pos_weight = torch.tensor(pos_weight)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr

        # Define the model
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),  # Input layer -> Hidden Layer 1 (64 units)
            nn.Sigmoid(),              # Activation function for Hidden Layer 1
            nn.Linear(64, 32),         # Hidden Layer 1 -> Hidden Layer 2 (32 units)
            nn.Sigmoid(),              # Activation function for Hidden Layer 2
            nn.Linear(32, 1)           # Hidden Layer 2 -> Output Layer (1 unit for binary classification)
        )

        # Loss function and optimizer
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Initialize TensorBoard writer
        self.writer = SummaryWriter("runs/myocardial_infarction_complications")

    def train(self, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor):
        dummy_input = torch.randn(1, self.input_size)
        try:
            self.writer.add_graph(self.model, dummy_input)
        except Exception as e:
            print(f"Error logging graph: {e}")

        for epoch in range(self.num_epochs):
            # Shuffle the training data
            permutation = torch.randperm(X_train_tensor.size(0))
            X_train_tensor = X_train_tensor[permutation]
            y_train_tensor = y_train_tensor[permutation]

            # Initialize training metrics
            train_loss = 0.0
            correct_train_predictions = 0
            true_positives_train = 0
            true_negatives_train = 0
            false_negatives_train = 0
            false_positives_train = 0

            for i in range(0, X_train_tensor.size(0), self.batch_size):
                X_batch = X_train_tensor[i:i + self.batch_size]
                y_batch = y_train_tensor[i:i + self.batch_size]

                # Forward pass
                logits = self.model(X_batch)
                if logits.shape != y_batch.shape:
                    logits = logits.view_as(y_batch)
                loss = self.criterion(logits, y_batch)
                train_loss += loss.item()

                # Training predictions
                train_probabilities = torch.sigmoid(logits)
                train_predictions = (train_probabilities > 0.5).float()

                # Count correct predictions
                correct_train_predictions += (train_predictions == y_batch).sum().item()

                # Calculate confusion matrix components
                true_positives_train += ((train_predictions == 1) & (y_batch == 1)).sum().item()
                true_negatives_train += ((train_predictions == 0) & (y_batch == 0)).sum().item()
                false_negatives_train += ((train_predictions == 0) & (y_batch == 1)).sum().item()
                false_positives_train += ((train_predictions == 1) & (y_batch == 0)).sum().item()

                # Backward pass (update weights)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Compute training metrics
            train_loss /= (X_train_tensor.size(0) // self.batch_size)
            train_accuracy = correct_train_predictions / X_train_tensor.size(0)
            train_recall = (
                true_positives_train / (true_positives_train + false_negatives_train)
                if true_positives_train + false_negatives_train > 0
                else 0
            )
            train_precision = (
                true_positives_train / (true_positives_train + false_positives_train)
                if true_positives_train + false_positives_train > 0
                else 0
            )
            train_f1 = (
                2 * (train_precision * train_recall) / (train_precision + train_recall)
                if train_precision + train_recall > 0
                else 0
            )
            train_specificity = (
                true_negatives_train / (true_negatives_train + false_positives_train)
                if true_negatives_train + false_positives_train > 0
                else 0
            )
            train_balanced_accuracy = (train_recall + train_specificity) / 2

            # Validation metrics
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_val_tensor)
                if val_logits.shape != y_val_tensor.shape:
                    val_logits = val_logits.view_as(y_val_tensor)
                val_loss = self.criterion(val_logits, y_val_tensor).item()
                val_probabilities = torch.sigmoid(val_logits)
                val_predictions = (val_probabilities > 0.5).float()
                correct_val_predictions = (val_predictions == y_val_tensor).sum().item()
                true_positives_val = ((val_predictions == 1) & (y_val_tensor == 1)).sum().item()
                true_negatives_val = ((val_predictions == 0) & (y_val_tensor == 0)).sum().item()
                false_negatives_val = ((val_predictions == 0) & (y_val_tensor == 1)).sum().item()
                false_positives_val = ((val_predictions == 1) & (y_val_tensor == 0)).sum().item()

                val_accuracy = correct_val_predictions / X_val_tensor.size(0)
                val_recall = (
                    true_positives_val / (true_positives_val + false_negatives_val)
                    if true_positives_val + false_negatives_val > 0
                    else 0
                )
                val_precision = (
                    true_positives_val / (true_positives_val + false_positives_val)
                    if true_positives_val + false_positives_val > 0
                    else 0
                )
                val_f1 = (
                    2 * (val_precision * val_recall) / (val_precision + val_recall)
                    if val_precision + val_recall > 0
                    else 0
                )
                val_specificity = (
                    true_negatives_val / (true_negatives_val + false_positives_val)
                    if true_negatives_val + false_positives_val > 0
                    else 0
                )
                val_balanced_accuracy = (val_recall + val_specificity) / 2

            self.model.train()

            # Log metrics to TensorBoard
            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Accuracy/Train", train_accuracy, epoch)
            self.writer.add_scalar("Recall/Train", train_recall, epoch)
            self.writer.add_scalar("Precision/Train", train_precision, epoch)
            self.writer.add_scalar("F1/Train", train_f1, epoch)
            self.writer.add_scalar("Balanced Accuracy/Train", train_balanced_accuracy, epoch)

            self.writer.add_scalar("Loss/Validation", val_loss, epoch)
            self.writer.add_scalar("Accuracy/Validation", val_accuracy, epoch)
            self.writer.add_scalar("Recall/Validation", val_recall, epoch)
            self.writer.add_scalar("Precision/Validation", val_precision, epoch)
            self.writer.add_scalar("F1/Validation", val_f1, epoch)
            self.writer.add_scalar("Balanced Accuracy/Validation", val_balanced_accuracy, epoch)

            # Print metrics for the epoch
            print(f"Epoch {epoch + 1}/{self.num_epochs}, "
                  f"Training Loss: {train_loss:.4f}, "
                  f"Training Accuracy: {train_accuracy:.4f}, "
                  f"Training Recall: {train_recall:.4f}, "
                  f"Training Precision: {train_precision:.4f}, "
                  f"Training F1: {train_f1:.4f}, "
                  f"Training Balanced Accuracy: {train_balanced_accuracy:.4f}, "
                  f"Validation Loss: {val_loss:.4f}, "
                  f"Validation Accuracy: {val_accuracy:.4f}, "
                  f"Validation Recall: {val_recall:.4f}, "
                  f"Validation Precision: {val_precision:.4f}, "
                  f"Validation F1: {val_f1:.4f}, "
                  f"Validation Balanced Accuracy: {val_balanced_accuracy:.4f}")

        self.writer.close()
        return self.model

    def save_model(self, path):
        torch.save(self.model, path)

    def load_model(self, path):
        self.model = torch.load(path)
        self.model.eval()