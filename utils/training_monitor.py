import os
import numpy as np
import matplotlib.pyplot as plt

from utils.logger import  LoggerManager

log = LoggerManager.get_logger(__name__)

class TrainingMonitor:
    def __init__(self, output_dir, early_stopping_patience=3, early_stopping_metric='val_loss'):
        self.output_dir = output_dir

        self.best_metric_value = float('inf')
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_metric = early_stopping_metric
        self.epochs_without_improvement = 0
        self.early_stopped = False

        self.metrics = {}

        self.plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)

    def log_metric(self, name, value, step):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append((step, value))

        if name == self.early_stopping_metric:
            if value < self.best_metric_value:
                self.best_metric_value = value
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            if self.epochs_without_improvement >= self.early_stopping_patience:
                log.info(f"Early stopping triggered after {self.epochs_without_improvement} epochs without improvement on metric {self.early_stopping_metric}.")
                self.early_stopped = True

        return self.early_stopped

    def plot_metric(self, name):
        if name not in self.metrics or not self.metrics[name]:
            return

        steps, values = zip(*self.metrics[name])

        plt.figure(figsize=(12, 8))
        plt.plot(steps, values, label=name)

        if len(values) > 10:
            window_size = min(10, len(values) // 5)
            smoothed_values = np.convolve(values,
                                          np.ones(window_size) / window_size,
                                          mode='valid')
            plt.plot(steps[window_size - 1:], smoothed_values,
                     'r--', linewidth=2, label=f'Smoothed (window={window_size})')

        plt.xlabel('Steps')
        plt.ylabel(name.replace('_', ' ').title())
        plt.title(f'{name.replace("_", " ").title()} Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.plots_dir, f'{name}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_all_metrics(self):
        for metric_name in self.metrics:
            self.plot_metric(metric_name)

    def generate_training_stats(self):
        self.plot_all_metrics()