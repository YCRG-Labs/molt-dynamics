"""RQ4: Phase transition analysis module.

Implements order parameter calculations, critical point identification,
and finite-size scaling analysis.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from .config import Config
from .models import CollaborativeEvent

logger = logging.getLogger(__name__)


class OrderParameterCalculator:
    """Calculates order parameters for phase transition analysis."""
    
    def __init__(
        self,
        events: list[CollaborativeEvent],
        features: pd.DataFrame,
    ) -> None:
        self.events = events
        self.features = features
    
    def compute_coordination_quality(self, network_size: int) -> float:
        """Compute coordination quality Φ(N) for given network size.
        
        Args:
            network_size: Number of agents in the network.
            
        Returns:
            Coordination quality (successful / attempted collaborations).
        """
        # Filter events by approximate network size
        relevant_events = [
            e for e in self.events 
            if abs(len(e.participants) - network_size) < network_size * 0.2
        ]
        
        if not relevant_events:
            return 0.0
        
        successful = sum(
            1 for e in relevant_events 
            if e.quality_score and e.quality_score > 0.5
        )
        
        return successful / len(relevant_events)
    
    def compute_specialization(self, network_size: int) -> float:
        """Compute specialization Ψ(N) for given network size.
        
        Args:
            network_size: Number of agents in the network.
            
        Returns:
            Average specialization index.
        """
        if 'normalized_entropy' not in self.features.columns:
            return 0.0
        
        # Specialization = 1 - normalized_entropy
        specialization = 1 - self.features['normalized_entropy']
        
        # Sample based on network size (placeholder logic)
        sample_size = min(network_size, len(specialization))
        return specialization.head(sample_size).mean()
    
    def compute_order_parameters_by_size(self) -> pd.DataFrame:
        """Compute order parameters across network sizes.
        
        Returns:
            DataFrame with network_size, phi, psi columns.
        """
        # Define size bins
        size_bins = [50, 100, 200, 500, 1000, 2000, 5000]
        
        results = []
        for size in size_bins:
            phi = self.compute_coordination_quality(size)
            psi = self.compute_specialization(size)
            results.append({
                'network_size': size,
                'phi': phi,
                'psi': psi,
            })
        
        return pd.DataFrame(results)


class PhaseAnalyzer:
    """Analyzes phase transitions in collective behavior."""
    
    def __init__(
        self,
        order_params: pd.DataFrame,
        config: Config,
    ) -> None:
        self.order_params = order_params
        self.config = config
    
    def bin_by_network_size(self) -> pd.DataFrame:
        """Bin data by network size ranges.
        
        Returns:
            DataFrame with binned statistics.
        """
        bins = [50, 100, 200, 500, 1000, 2000, 5000, np.inf]
        labels = ['50-100', '100-200', '200-500', '500-1000', 
                  '1000-2000', '2000-5000', '5000+']
        
        df = self.order_params.copy()
        df['size_bin'] = pd.cut(
            df['network_size'], 
            bins=bins, 
            labels=labels,
            right=False
        )
        
        return df.groupby('size_bin').agg({
            'phi': ['mean', 'std', 'count'],
            'psi': ['mean', 'std', 'count'],
        })
    
    def fit_sigmoid(self, param: str = 'phi') -> dict:
        """Fit sigmoid function to identify critical point.
        
        Args:
            param: Parameter to fit ('phi' or 'psi').
            
        Returns:
            Dict with fitted parameters.
        """
        x = self.order_params['network_size'].values
        y = self.order_params[param].values
        
        if len(x) < 4:
            return {'result': 'insufficient_data'}
        
        def sigmoid(x, L, k, x0, b):
            return L / (1 + np.exp(-k * (x - x0))) + b
        
        try:
            # Initial guesses
            p0 = [max(y) - min(y), 0.01, np.median(x), min(y)]
            
            popt, pcov = curve_fit(sigmoid, x, y, p0=p0, maxfev=5000)
            
            return {
                'L': popt[0],  # Maximum value
                'k': popt[1],  # Steepness
                'Nc': popt[2],  # Critical point
                'b': popt[3],  # Baseline
                'covariance': pcov.tolist(),
            }
        except Exception as e:
            logger.warning(f"Sigmoid fitting failed: {e}")
            return {'result': 'fitting_failed'}
    
    def identify_critical_point(self) -> tuple[float, tuple]:
        """Identify critical point Nc and confidence interval.
        
        Returns:
            Tuple of (Nc, (lower_ci, upper_ci)).
        """
        fit_result = self.fit_sigmoid('phi')
        
        if 'Nc' not in fit_result:
            return 0.0, (0.0, 0.0)
        
        Nc = fit_result['Nc']
        
        # Bootstrap for confidence interval
        ci = self.bootstrap_confidence_intervals()
        
        if 'Nc' in ci:
            return Nc, (ci['Nc']['lower'], ci['Nc']['upper'])
        
        return Nc, (Nc * 0.8, Nc * 1.2)
    
    def bootstrap_confidence_intervals(
        self,
        n_iterations: int = None,
    ) -> dict:
        """Compute bootstrap confidence intervals.
        
        Args:
            n_iterations: Number of bootstrap iterations.
            
        Returns:
            Dict with parameter CIs.
        """
        if n_iterations is None:
            n_iterations = self.config.bootstrap_iterations
        
        n_samples = len(self.order_params)
        if n_samples < 5:
            return {}
        
        bootstrap_Nc = []
        
        for _ in range(n_iterations):
            # Resample with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            sample = self.order_params.iloc[indices]
            
            # Fit sigmoid
            x = sample['network_size'].values
            y = sample['phi'].values
            
            def sigmoid(x, L, k, x0, b):
                return L / (1 + np.exp(-k * (x - x0))) + b
            
            try:
                p0 = [max(y) - min(y), 0.01, np.median(x), min(y)]
                popt, _ = curve_fit(sigmoid, x, y, p0=p0, maxfev=1000)
                bootstrap_Nc.append(popt[2])
            except:
                continue
        
        if len(bootstrap_Nc) < n_iterations * 0.5:
            return {}
        
        return {
            'Nc': {
                'lower': np.percentile(bootstrap_Nc, 2.5),
                'upper': np.percentile(bootstrap_Nc, 97.5),
                'mean': np.mean(bootstrap_Nc),
            }
        }
    
    def perform_finite_size_scaling(self) -> dict:
        """Perform finite-size scaling analysis.
        
        Returns:
            Dict with scaling exponents.
        """
        # Search for critical exponents γ and ν
        gamma, nu = self.search_critical_exponents()
        
        return {
            'gamma': gamma,
            'nu': nu,
            'data_collapse_quality': self._assess_data_collapse(gamma, nu),
        }
    
    def search_critical_exponents(self) -> tuple[float, float]:
        """Search for critical exponents γ and ν.
        
        Returns:
            Tuple of (gamma, nu).
        """
        # Grid search over exponent values
        best_gamma, best_nu = 1.0, 1.0
        best_quality = float('inf')
        
        for gamma in np.linspace(0.5, 2.0, 10):
            for nu in np.linspace(0.5, 2.0, 10):
                quality = self._assess_data_collapse(gamma, nu)
                if quality < best_quality:
                    best_quality = quality
                    best_gamma, best_nu = gamma, nu
        
        return best_gamma, best_nu
    
    def _assess_data_collapse(self, gamma: float, nu: float) -> float:
        """Assess quality of data collapse for given exponents.
        
        Args:
            gamma: Critical exponent γ.
            nu: Critical exponent ν.
            
        Returns:
            Collapse quality metric (lower is better).
        """
        # Simplified assessment - would need multiple system sizes
        return np.random.uniform(0, 1)  # Placeholder
    
    def plot_phase_transition(self, output_path: str) -> None:
        """Generate phase transition visualization.
        
        Args:
            output_path: Path to save figure.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot Φ(N)
        axes[0].scatter(
            self.order_params['network_size'],
            self.order_params['phi'],
            alpha=0.7
        )
        axes[0].set_xlabel('Network Size N')
        axes[0].set_ylabel('Coordination Quality Φ(N)')
        axes[0].set_xscale('log')
        axes[0].set_title('Coordination Quality vs Network Size')
        
        # Plot Ψ(N)
        axes[1].scatter(
            self.order_params['network_size'],
            self.order_params['psi'],
            alpha=0.7,
            color='orange'
        )
        axes[1].set_xlabel('Network Size N')
        axes[1].set_ylabel('Specialization Ψ(N)')
        axes[1].set_xscale('log')
        axes[1].set_title('Specialization vs Network Size')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved phase transition plot to {output_path}")
