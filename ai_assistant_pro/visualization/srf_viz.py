"""
Visualization tools for Stone Retrieval Function

Creates visual representations of SRF scores and components.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
import seaborn as sns

from ai_assistant_pro.srf import RetrievalResult


class SRFVisualizer:
    """Visualize SRF retrieval results"""

    def __init__(self, style: str = "darkgrid"):
        """
        Initialize visualizer

        Args:
            style: Seaborn style (darkgrid, whitegrid, dark, white, ticks)
        """
        sns.set_style(style)
        plt.rcParams['figure.figsize'] = (12, 6)

    def plot_score_breakdown(
        self,
        results: List[RetrievalResult],
        top_k: int = 10,
        save_path: Optional[str] = None,
    ):
        """
        Plot score breakdown for top-k results

        Args:
            results: Retrieval results from SRF
            top_k: Number of top results to show
            save_path: Optional path to save figure
        """
        results = results[:top_k]

        # Prepare data
        labels = [f"#{i+1}" for i in range(len(results))]
        components = ["semantic", "emotional", "associative", "recency", "decay"]

        data = {comp: [] for comp in components}
        for result in results:
            for comp in components:
                data[comp].append(result.components.get(comp, 0))

        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(labels))
        width = 0.6

        bottom = np.zeros(len(labels))

        colors = {
            "semantic": "#3498db",
            "emotional": "#e74c3c",
            "associative": "#2ecc71",
            "recency": "#f39c12",
            "decay": "#95a5a6",
        }

        for comp in components:
            values = data[comp]
            if comp == "decay":
                # Decay is negative, show as downward
                values = [-v for v in values]

            ax.bar(x, values, width, label=comp.capitalize(),
                   bottom=bottom, color=colors[comp])

            if comp != "decay":
                bottom += values

        ax.set_ylabel('Score Contribution')
        ax.set_xlabel('Retrieval Rank')
        ax.set_title('SRF Score Component Breakdown (Top-{})'.format(top_k))
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(loc='upper right')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {save_path}")
        else:
            plt.show()

    def plot_score_comparison(
        self,
        results: List[RetrievalResult],
        baseline_scores: Optional[List[float]] = None,
        top_k: int = 20,
        save_path: Optional[str] = None,
    ):
        """
        Compare SRF scores with baseline (e.g., semantic similarity only)

        Args:
            results: SRF retrieval results
            baseline_scores: Optional baseline scores for comparison
            top_k: Number of results to show
            save_path: Optional path to save figure
        """
        results = results[:top_k]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: SRF scores
        srf_scores = [r.score for r in results]
        x = np.arange(len(srf_scores))

        ax1.bar(x, srf_scores, color='#3498db', alpha=0.7)
        ax1.set_xlabel('Rank')
        ax1.set_ylabel('SRF Score')
        ax1.set_title('SRF Retrieval Scores')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Comparison
        if baseline_scores:
            baseline_scores = baseline_scores[:top_k]

            ax2.plot(x, srf_scores, 'o-', label='SRF', color='#3498db', linewidth=2)
            ax2.plot(x, baseline_scores, 's-', label='Baseline', color='#e74c3c', linewidth=2)
            ax2.set_xlabel('Rank')
            ax2.set_ylabel('Score')
            ax2.set_title('SRF vs Baseline')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            # Show component contributions
            semantic_scores = [r.components['semantic'] for r in results]
            ax2.scatter(semantic_scores, srf_scores, alpha=0.6, s=100)
            ax2.set_xlabel('Semantic Similarity')
            ax2.set_ylabel('SRF Score')
            ax2.set_title('SRF Score vs Semantic Similarity')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {save_path}")
        else:
            plt.show()

    def plot_component_heatmap(
        self,
        results: List[RetrievalResult],
        top_k: int = 20,
        save_path: Optional[str] = None,
    ):
        """
        Create heatmap of component scores

        Args:
            results: Retrieval results
            top_k: Number of results
            save_path: Optional save path
        """
        results = results[:top_k]

        # Prepare data
        components = ["semantic", "emotional", "associative", "recency", "decay"]
        data = []

        for result in results:
            row = [result.components.get(comp, 0) for comp in components]
            data.append(row)

        data = np.array(data).T

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 4))

        im = ax.imshow(data, cmap='RdYlGn', aspect='auto')

        # Labels
        ax.set_xticks(np.arange(len(results)))
        ax.set_yticks(np.arange(len(components)))
        ax.set_xticklabels([f"#{i+1}" for i in range(len(results))])
        ax.set_yticklabels([c.capitalize() for c in components])

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Score Contribution', rotation=270, labelpad=15)

        # Title
        ax.set_title('SRF Component Heatmap (Top-{})'.format(top_k))
        ax.set_xlabel('Retrieval Rank')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {save_path}")
        else:
            plt.show()

    def plot_temporal_analysis(
        self,
        results: List[RetrievalResult],
        save_path: Optional[str] = None,
    ):
        """
        Analyze temporal components (recency and decay)

        Args:
            results: Retrieval results
            save_path: Optional save path
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Extract data
        recency_scores = [r.components['recency'] for r in results]
        decay_scores = [r.components['decay'] for r in results]
        final_scores = [r.score for r in results]

        # Plot 1: Recency vs Final Score
        ax1.scatter(recency_scores, final_scores, alpha=0.6, s=100, color='#3498db')
        ax1.set_xlabel('Recency Score')
        ax1.set_ylabel('Final SRF Score')
        ax1.set_title('Impact of Recency on Final Score')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Decay vs Final Score
        ax2.scatter(decay_scores, final_scores, alpha=0.6, s=100, color='#e74c3c')
        ax2.set_xlabel('Decay Score')
        ax2.set_ylabel('Final SRF Score')
        ax2.set_title('Impact of Decay on Final Score')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {save_path}")
        else:
            plt.show()


def visualize_srf_results(
    results: List[RetrievalResult],
    output_dir: str = "visualizations",
):
    """
    Create comprehensive visualization suite

    Args:
        results: SRF retrieval results
        output_dir: Directory to save visualizations
    """
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    viz = SRFVisualizer()

    print("Creating visualizations...")

    # Score breakdown
    viz.plot_score_breakdown(
        results,
        top_k=10,
        save_path=output_path / "score_breakdown.png"
    )

    # Score comparison
    viz.plot_score_comparison(
        results,
        top_k=20,
        save_path=output_path / "score_comparison.png"
    )

    # Component heatmap
    viz.plot_component_heatmap(
        results,
        top_k=20,
        save_path=output_path / "component_heatmap.png"
    )

    # Temporal analysis
    viz.plot_temporal_analysis(
        results,
        save_path=output_path / "temporal_analysis.png"
    )

    print(f"✓ Visualizations saved to {output_path}/")
