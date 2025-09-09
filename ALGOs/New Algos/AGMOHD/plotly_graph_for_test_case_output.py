#!/usr/bin/env python3
"""
YALGO-S Interactive Visualization Dashboard

Advanced plotly-based visualization system for YALGO-S algorithms.
Includes performance monitoring, algorithm comparison, and real-time analytics.

Features:
- Interactive performance dashboards
- Algorithm benchmarking visualizations
- Real-time training monitoring
- Cross-platform compatibility charts
- GPU utilization tracking
- Memory usage analytics
"""

import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available. Using numpy arrays for demonstration.")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
from typing import Dict, List, Any, Optional
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YALGOVisualizer:
    """
    Comprehensive visualization system for YALGO-S algorithms.

    Provides interactive dashboards for:
    - Algorithm performance comparison
    - Real-time training monitoring
    - Hardware utilization tracking
    - Benchmark result analysis
    """

    def __init__(self, theme='plotly_white'):
        """
        Initialize the visualizer with theme settings.

        Args:
            theme (str): Plotly theme ('plotly_white', 'plotly_dark', 'ggplot2', etc.)
        """
        self.theme = theme
        self.color_palette = px.colors.qualitative.Set3
        self.performance_data = []
        self.system_metrics = []

    def create_algorithm_comparison_dashboard(self, results_data: Dict[str, Any]) -> go.Figure:
        """
        Create comprehensive algorithm comparison dashboard.

        Args:
            results_data: Dictionary containing algorithm performance results

        Returns:
            plotly Figure: Interactive dashboard
        """
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Algorithm Accuracy Comparison', 'Training Time Analysis',
                          'Memory Usage Comparison', 'GPU Utilization Trends'),
        specs=[[{"type": "bar"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "scatter"}]]
        )

        # Extract data
        algorithms = list(results_data.keys())
        accuracies = [results_data[alg].get('accuracy', 0) for alg in algorithms]
        training_times = [results_data[alg].get('training_time', 0) for alg in algorithms]
        memory_usage = [results_data[alg].get('memory_usage', 0) for alg in algorithms]
        gpu_utilization = [results_data[alg].get('gpu_utilization', []) for alg in algorithms]

        # Accuracy comparison
        fig.add_trace(
            go.Bar(x=algorithms, y=accuracies, name='Accuracy',
                   marker_color=self.color_palette[0],
                   text=[f'{acc:.2f}%' for acc in accuracies],
                   textposition='auto'),
            row=1, col=1
        )

        # Training time analysis
        fig.add_trace(
            go.Scatter(x=algorithms, y=training_times, mode='lines+markers',
                      name='Training Time', marker_color=self.color_palette[1],
                      line=dict(width=3)),
            row=1, col=2
        )

        # Memory usage comparison
        fig.add_trace(
            go.Bar(x=algorithms, y=memory_usage, name='Memory Usage (MB)',
                   marker_color=self.color_palette[2],
                   text=[f'{mem:.1f}MB' for mem in memory_usage],
                   textposition='auto'),
            row=2, col=1
        )

        # GPU utilization trends (simplified)
        for i, (alg, gpu_data) in enumerate(zip(algorithms, gpu_utilization)):
            if gpu_data:
                fig.add_trace(
                    go.Scatter(x=list(range(len(gpu_data))), y=gpu_data,
                              mode='lines', name=f'{alg} GPU',
                              line=dict(color=self.color_palette[i % len(self.color_palette)])),
                    row=2, col=2
                )

        # Update layout
        fig.update_layout(
            title_text="YALGO-S Algorithm Performance Dashboard",
            title_x=0.5,
            height=800,
            showlegend=True,
            template=self.theme
        )

        # Update axes
        fig.update_xaxes(title_text="Algorithms", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy (%)", row=1, col=1)
        fig.update_xaxes(title_text="Algorithms", row=1, col=2)
        fig.update_yaxes(title_text="Time (seconds)", row=1, col=2)
        fig.update_xaxes(title_text="Algorithms", row=2, col=1)
        fig.update_yaxes(title_text="Memory (MB)", row=2, col=1)
        fig.update_xaxes(title_text="Time Steps", row=2, col=2)
        fig.update_yaxes(title_text="GPU Utilization (%)", row=2, col=2)

        return fig

    def create_training_progress_dashboard(self, training_history: Dict[str, List[float]],
                                         metrics: List[str] = None) -> go.Figure:
        """
        Create real-time training progress dashboard.

        Args:
            training_history: Dictionary with training metrics over time
            metrics: List of metrics to display

        Returns:
            plotly Figure: Training progress dashboard
        """
        if metrics is None:
            metrics = ['loss', 'accuracy', 'learning_rate']

        # Create subplot figure
        n_metrics = len(metrics)
        rows = (n_metrics + 1) // 2
        cols = min(2, n_metrics)

        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f'{metric.title()} vs Epoch' for metric in metrics[:rows*cols]],
            specs=[[{"type": "scatter"} for _ in range(cols)] for _ in range(rows)]
        )

        colors = self.color_palette[:len(metrics)]

        for i, (metric, color) in enumerate(zip(metrics, colors)):
            if metric in training_history:
                data = training_history[metric]
                row = (i // cols) + 1
                col = (i % cols) + 1

                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(data))),
                        y=data,
                        mode='lines+markers',
                        name=metric.title(),
                        line=dict(color=color, width=2),
                        marker=dict(size=4)
                    ),
                    row=row, col=col
                )

                # Update axis labels
                fig.update_xaxes(title_text="Epoch", row=row, col=col)
                fig.update_yaxes(title_text=metric.title(), row=row, col=col)

        # Update layout
        fig.update_layout(
            title_text="YALGO-S Training Progress Dashboard",
            title_x=0.5,
            height=400 * rows,
            showlegend=True,
            template=self.theme
        )

        return fig

    def create_system_monitoring_dashboard(self, duration: int = 60) -> go.Figure:
        """
        Create real-time system monitoring dashboard.

        Args:
            duration: Monitoring duration in seconds

        Returns:
            plotly Figure: System monitoring dashboard
        """
        # Collect system metrics
        timestamps = []
        cpu_usage = []
        memory_usage = []
        gpu_usage = []
        gpu_memory = []

        start_time = time.time()

        for _ in range(duration):
            timestamps.append(time.time() - start_time)

            # CPU and Memory
            cpu_usage.append(psutil.cpu_percent())
            memory = psutil.virtual_memory()
            memory_usage.append(memory.percent)

            # GPU metrics (if available)
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_usage.append(gpus[0].load * 100)
                    gpu_memory.append(gpus[0].memoryUtil * 100)
                else:
                    gpu_usage.append(0)
                    gpu_memory.append(0)
            except:
                gpu_usage.append(0)
                gpu_memory.append(0)

            time.sleep(1)

        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('CPU Usage', 'Memory Usage', 'GPU Usage', 'GPU Memory'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )

        # CPU Usage
        fig.add_trace(
            go.Scatter(x=timestamps, y=cpu_usage, mode='lines',
                      name='CPU %', line=dict(color='red', width=2)),
            row=1, col=1
        )

        # Memory Usage
        fig.add_trace(
            go.Scatter(x=timestamps, y=memory_usage, mode='lines',
                      name='Memory %', line=dict(color='blue', width=2)),
            row=1, col=2
        )

        # GPU Usage
        fig.add_trace(
            go.Scatter(x=timestamps, y=gpu_usage, mode='lines',
                      name='GPU %', line=dict(color='green', width=2)),
            row=2, col=1
        )

        # GPU Memory
        fig.add_trace(
            go.Scatter(x=timestamps, y=gpu_memory, mode='lines',
                      name='GPU Memory %', line=dict(color='orange', width=2)),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            title_text="YALGO-S System Monitoring Dashboard",
            title_x=0.5,
            height=600,
            showlegend=True,
            template=self.theme
        )

        # Update axes
        for row in range(1, 3):
            for col in range(1, 3):
                fig.update_xaxes(title_text="Time (s)", row=row, col=col)
                fig.update_yaxes(title_text="Usage (%)", row=row, col=col)

        return fig

    def create_dataset_visualization(self, X, y,
                                   task_type: str = 'regression') -> go.Figure:
        """
        Create interactive dataset visualization.

        Args:
            X: Feature tensor
            y: Target tensor
            task_type: 'regression' or 'classification'

        Returns:
            plotly Figure: Dataset visualization
        """
        if TORCH_AVAILABLE:
            if isinstance(X, torch.Tensor):
                X = X.numpy()
            if isinstance(y, torch.Tensor):
                y = y.numpy()

        if task_type == 'regression':
            # Regression: scatter plot of first feature vs target
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=X[:, 0],
                    y=y.squeeze(),
                    mode='markers',
                    name='Data Points',
                    marker=dict(
                        color=y.squeeze(),
                        colorscale='Viridis',
                        size=8,
                        showscale=True,
                        colorbar=dict(title="Target Value")
                    )
                )
            )

            fig.update_layout(
                title="Synthetic Regression Dataset Visualization",
                xaxis_title="Feature 1",
                yaxis_title="Target",
                template=self.theme
            )

        elif task_type == 'classification':
            # Classification: scatter plot with color coding
            fig = go.Figure()

            # Handle multi-dimensional features
            if X.shape[1] >= 2:
                fig.add_trace(
                    go.Scatter(
                        x=X[:, 0],
                        y=X[:, 1],
                        mode='markers',
                        name='Data Points',
                        marker=dict(
                            color=y.squeeze(),
                            colorscale='RdYlBu',
                            size=8,
                            showscale=True,
                            colorbar=dict(title="Class Label")
                        )
                    )
                )

                fig.update_layout(
                    title="Synthetic Classification Dataset Visualization",
                    xaxis_title="Feature 1",
                    yaxis_title="Feature 2",
                    template=self.theme
                )
            else:
                # Single feature classification
                fig.add_trace(
                    go.Scatter(
                        x=X[:, 0],
                        y=y.squeeze(),
                        mode='markers',
                        name='Data Points',
                        marker=dict(
                            color=y.squeeze(),
                            colorscale='RdYlBu',
                            size=8,
                            showscale=True,
                            colorbar=dict(title="Class Label")
                        )
                    )
                )

                fig.update_layout(
                    title="Synthetic Classification Dataset Visualization",
                    xaxis_title="Feature 1",
                    yaxis_title="Class Label",
                    template=self.theme
                )

        return fig

    def create_performance_heatmap(self, benchmark_results: Dict[str, Dict[str, float]]) -> go.Figure:
        """
        Create performance comparison heatmap.

        Args:
            benchmark_results: Nested dictionary of benchmark results

        Returns:
            plotly Figure: Performance heatmap
        """
        # Extract data
        algorithms = list(benchmark_results.keys())
        metrics = []

        # Find all metrics
        for alg_results in benchmark_results.values():
            for metric in alg_results.keys():
                if metric not in metrics:
                    metrics.append(metric)

        # Create data matrix (only numeric values for heatmap)
        data_matrix = []
        text_matrix = []
        for alg in algorithms:
            row = []
            text_row = []
            for metric in metrics:
                value = benchmark_results[alg].get(metric, 0)
                # Handle lists (like gpu_utilization) by taking average or first value
                if isinstance(value, list):
                    if value:
                        numeric_value = sum(value) / len(value)  # Average for lists
                        text_value = f'{numeric_value:.1f}'
                    else:
                        numeric_value = 0
                        text_value = 'N/A'
                else:
                    numeric_value = float(value) if isinstance(value, (int, float)) else 0
                    text_value = f'{numeric_value:.2f}' if isinstance(value, (int, float)) else str(value)

                row.append(numeric_value)
                text_row.append(text_value)
            data_matrix.append(row)
            text_matrix.append(text_row)

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=data_matrix,
            x=metrics,
            y=algorithms,
            colorscale='RdYlGn',
            text=text_matrix,
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))

        fig.update_layout(
            title="YALGO-S Algorithm Performance Heatmap",
            xaxis_title="Metrics",
            yaxis_title="Algorithms",
            template=self.theme
        )

        return fig

    def create_cross_platform_comparison(self, platform_results: Dict[str, Dict[str, float]]) -> go.Figure:
        """
        Create cross-platform performance comparison.

        Args:
            platform_results: Dictionary with platform-specific results

        Returns:
            plotly Figure: Cross-platform comparison chart
        """
        platforms = list(platform_results.keys())
        metrics = list(platform_results[platforms[0]].keys()) if platforms else []

        fig = go.Figure()

        for i, metric in enumerate(metrics):
            values = [platform_results[platform].get(metric, 0) for platform in platforms]

            fig.add_trace(
                go.Bar(
                    name=metric.title(),
                    x=platforms,
                    y=values,
                    offsetgroup=i,
                    marker_color=self.color_palette[i % len(self.color_palette)]
                )
            )

        fig.update_layout(
            title="YALGO-S Cross-Platform Performance Comparison",
            xaxis_title="Platform",
            yaxis_title="Performance Score",
            barmode='group',
            template=self.theme
        )

        return fig

    def save_dashboard(self, fig: go.Figure, filename: str, format: str = 'html'):
        """
        Save dashboard to file.

        Args:
            fig: Plotly figure
            filename: Output filename (without extension)
            format: Output format ('html', 'png', 'svg', 'pdf')
        """
        if format == 'html':
            fig.write_html(f"{filename}.html")
        else:
            fig.write_image(f"{filename}.{format}")

        logger.info(f"Dashboard saved as {filename}.{format}")

    def show_dashboard(self, fig: go.Figure):
        """
        Display dashboard in browser or notebook.

        Args:
            fig: Plotly figure
        """
        fig.show()

# Utility functions for data generation and visualization
def generate_sample_data():
    """Generate sample data for demonstration."""
    if TORCH_AVAILABLE:
        # Regression data
        X_reg = torch.randn(200, 1)
        y_reg = 2 * X_reg.squeeze() + 1 + 0.1 * torch.randn(200)

        # Classification data
        X_clf = torch.randn(200, 2)
        y_clf = (X_clf[:, 0] + X_clf[:, 1] > 0).long()
    else:
        # Use numpy as fallback
        # Regression data
        X_reg = np.random.randn(200, 1)
        y_reg = 2 * X_reg.squeeze() + 1 + 0.1 * np.random.randn(200)

        # Classification data
        X_clf = np.random.randn(200, 2)
        y_clf = (X_clf[:, 0] + X_clf[:, 1] > 0).astype(int)

    return X_reg, y_reg, X_clf, y_clf

def create_sample_benchmark_results():
    """Create sample benchmark results for demonstration."""
    return {
        'AGMOHD': {
            'accuracy': 98.8,
            'training_time': 85.0,
            'memory_usage': 2100,
            'gpu_utilization': [45, 67, 78, 82, 75]
        },
        'POIC-NET': {
            'accuracy': 92.1,
            'training_time': 120.0,
            'memory_usage': 3200,
            'gpu_utilization': [52, 71, 85, 88, 79]
        },
        'ARCE': {
            'accuracy': 94.2,
            'training_time': 95.0,
            'memory_usage': 1800,
            'gpu_utilization': [38, 55, 68, 72, 65]
        }
    }

def create_sample_training_history():
    """Create sample training history for demonstration."""
    epochs = list(range(20))
    loss = [2.3 * np.exp(-0.1 * i) + 0.1 * np.random.randn() for i in epochs]
    accuracy = [50 + 45 * (1 - np.exp(-0.15 * i)) + 2 * np.random.randn() for i in epochs]
    learning_rate = [0.01 * (0.9 ** i) for i in epochs]

    return {
        'loss': loss,
        'accuracy': accuracy,
        'learning_rate': learning_rate
    }

def create_sample_platform_results():
    """Create sample cross-platform results."""
    return {
        'Linux': {'speedup': 2.8, 'memory_efficiency': 85, 'compatibility': 100},
        'macOS': {'speedup': 1.9, 'memory_efficiency': 92, 'compatibility': 100},
        'Windows': {'speedup': 2.6, 'memory_efficiency': 87, 'compatibility': 100}
    }

# Main demonstration function
def main():
    """Main function to demonstrate YALGO-S visualization capabilities."""
    print("🚀 YALGO-S Interactive Visualization Dashboard")
    print("=" * 50)

    # Initialize visualizer
    visualizer = YALGOVisualizer(theme='plotly_white')

    # Generate sample data
    print("📊 Generating sample data...")
    X_reg, y_reg, X_clf, y_clf = generate_sample_data()

    # Create dataset visualizations
    print("📈 Creating dataset visualizations...")
    reg_fig = visualizer.create_dataset_visualization(X_reg, y_reg, 'regression')
    clf_fig = visualizer.create_dataset_visualization(X_clf, y_clf, 'classification')

    # Create benchmark results
    print("📊 Creating benchmark visualizations...")
    benchmark_data = create_sample_benchmark_results()
    benchmark_fig = visualizer.create_algorithm_comparison_dashboard(benchmark_data)

    # Create training history
    print("📈 Creating training progress visualization...")
    training_data = create_sample_training_history()
    training_fig = visualizer.create_training_progress_dashboard(training_data)

    # Create performance heatmap
    print("🔥 Creating performance heatmap...")
    heatmap_fig = visualizer.create_performance_heatmap(benchmark_data)

    # Create cross-platform comparison
    print("🌐 Creating cross-platform comparison...")
    platform_data = create_sample_platform_results()
    platform_fig = visualizer.create_cross_platform_comparison(platform_data)

    # Save all visualizations
    print("💾 Saving visualizations...")
    visualizer.save_dashboard(reg_fig, "yalgo_s_regression_dataset", 'html')
    visualizer.save_dashboard(clf_fig, "yalgo_s_classification_dataset", 'html')
    visualizer.save_dashboard(benchmark_fig, "yalgo_s_algorithm_comparison", 'html')
    visualizer.save_dashboard(training_fig, "yalgo_s_training_progress", 'html')
    visualizer.save_dashboard(heatmap_fig, "yalgo_s_performance_heatmap", 'html')
    visualizer.save_dashboard(platform_fig, "yalgo_s_cross_platform", 'html')

    print("✅ All visualizations created and saved!")
    print("\n📁 Generated files:")
    print("  - yalgo_s_regression_dataset.html")
    print("  - yalgo_s_classification_dataset.html")
    print("  - yalgo_s_algorithm_comparison.html")
    print("  - yalgo_s_training_progress.html")
    print("  - yalgo_s_performance_heatmap.html")
    print("  - yalgo_s_cross_platform.html")

    # Optionally display in browser (uncomment to show)
    # print("\n🌐 Displaying visualizations in browser...")
    # visualizer.show_dashboard(benchmark_fig)

    print("\n🎉 YALGO-S Visualization Dashboard Complete!")
    print("💡 Open the HTML files in your browser for interactive exploration")

if __name__ == "__main__":
    main()
