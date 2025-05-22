import os
import json
import numpy as np
import matplotlib.pyplot as plt

from utils.logger import  LoggerManager

log = LoggerManager.get_logger(__name__)

class LossTracker:
    def __init__(self, output_dir, early_stopping_patience=3):
        self.output_dir = output_dir

        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.early_stopped = False
        self.early_stopping_patience = early_stopping_patience

        self.training_losses = []
        self.val_losses = []
        self.epochs = []
        self.steps = []

        self.plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)

    def add_training_loss(self, loss, step, epoch=None):
        """Record a training loss value"""
        self.training_losses.append(loss)
        self.steps.append(step)
        if epoch is not None:
            self.epochs.append(epoch)

    def add_validation_loss(self, loss):
        """Record a validation loss value"""
        self.val_losses.append(loss)

        if loss < self.best_val_loss:
            self.best_val_loss = loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= self.early_stopping_patience:
            log.info(f"Early stopping triggered after {self.epochs_without_improvement} epochs without improvement.")
            self.early_stopped = True

        return self.early_stopped

    def plot_training_loss(self):
        """Generate a plot of training loss over steps"""
        if not self.training_losses:
            return

        plt.figure(figsize=(12, 8))
        plt.plot(self.steps, self.training_losses, label='Training Loss')

        if len(self.training_losses) > 10:
            window_size = min(10, len(self.training_losses) // 5)
            smoothed_losses = np.convolve(self.training_losses,
                                          np.ones(window_size) / window_size,
                                          mode='valid')
            plt.plot(self.steps[window_size - 1:], smoothed_losses,
                     'r--', linewidth=2, label=f'Smoothed (window={window_size})')

        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.plots_dir, 'training_loss.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_validation_loss(self):
        """Generate a plot of validation loss over evaluations"""
        if not self.val_losses:
            return

        plt.figure(figsize=(12, 8))
        evals = list(range(1, len(self.val_losses) + 1))
        plt.plot(evals, self.val_losses, 'o-', label='Validation Loss')
        plt.xlabel('Evaluation')
        plt.ylabel('Loss')
        plt.title('Validation Loss')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.plots_dir, 'validation_loss.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_combined_losses(self):
        """Generate a plot comparing training and validation losses"""
        if not self.val_losses:
            return

        # create evenly spaced x-values for validation
        val_x = np.linspace(min(self.steps), max(self.steps), len(self.val_losses))

        plt.figure(figsize=(10, 6))
        plt.plot(self.steps, self.training_losses, label='Training Loss', alpha=0.7)
        plt.plot(val_x, self.val_losses, 'o-', label='Validation Loss')

        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training vs Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.plots_dir, 'combined_losses.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def save_data(self):
        """Save tracking data to JSON file"""
        data = {
            "training_losses": self.training_losses,
            "validation_losses": self.val_losses,
            "steps": self.steps,
            "best_val_loss": self.best_val_loss,
        }

        with open(os.path.join(self.output_dir, "training_history.json"), "w") as f:
            json.dump(data, f, indent=2)

    def generate_all_plots(self):
        """Generate all available plots"""
        self.plot_training_loss()
        self.plot_validation_loss()
        self.plot_combined_losses()
        self.save_data()