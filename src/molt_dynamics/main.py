"""Main analysis pipeline for Molt Dynamics.

Orchestrates the complete analysis workflow from data collection through
output generation.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from .config import Config
from .storage import JSONStorage
from .dataset_loader import MoltBookDatasetLoader
from .network import NetworkBuilder
from .features import FeatureExtractor, TopicModeler
from .rq1_roles import RoleAnalyzer
from .rq3_collaboration import CollaborationIdentifier
from .validation import StatisticalFramework, RobustnessChecker
from .output import OutputGenerator

logger = logging.getLogger(__name__)


def setup_logging(config: Config) -> None:
    """Configure logging for the analysis pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Path(config.output_dir) / 'analysis.log'),
            logging.StreamHandler(sys.stdout),
        ]
    )


def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    logger.info(f"Set random seed to {seed}")


def run_data_loading(config: Config, storage: JSONStorage, dataset_path: str) -> None:
    """Load data from the MoltBook Observatory Archive dataset."""
    logger.info("Loading data from dataset...")
    
    loader = MoltBookDatasetLoader(config, storage, dataset_path)
    results = loader.load_all()
    
    logger.info(f"Data loading complete: {results}")


def run_network_analysis(storage: JSONStorage) -> tuple:
    """Build and analyze interaction networks.
    
    Returns:
        Tuple of (NetworkBuilder, network graph) to avoid rebuilding.
    """
    logger.info("Building interaction networks...")
    
    builder = NetworkBuilder(storage)
    G = builder.build_interaction_network(directed=True)
    
    stats = builder.get_network_statistics(G)
    logger.info(f"Network statistics: {stats}")
    
    return builder, G


def run_feature_extraction(
    storage: JSONStorage, 
    network, 
    config: Config
) -> tuple:
    """Extract behavioral features for all agents."""
    logger.info("Extracting agent features...")
    
    extractor = FeatureExtractor(storage, network, config)
    features = extractor.extract_all_features()
    features_std = extractor.standardize_features(features)
    
    logger.info(f"Extracted features for {len(features)} agents")
    
    return features, features_std


def run_rq1_analysis(features_std, config: Config) -> dict:
    """RQ1: Role emergence analysis."""
    logger.info("Running RQ1: Role emergence analysis...")
    
    analyzer = RoleAnalyzer(features_std, config)
    
    # Find optimal k
    optimal_k, silhouette_scores = analyzer.find_optimal_k()
    
    # Perform clustering
    labels = analyzer.perform_clustering(optimal_k)
    
    # Classify roles
    roles = analyzer.classify_roles()
    
    # Save all data instead of just generating visualization
    saved_files = analyzer.save_all_data(config.output_dir)
    
    return {
        'optimal_k': optimal_k,
        'silhouette_scores': silhouette_scores,
        'role_distribution': roles.value_counts().to_dict(),
        'saved_files': saved_files,
    }


def run_rq2_analysis(storage: JSONStorage, network, config: Config) -> dict:
    """RQ2: Information diffusion analysis."""
    logger.info("Running RQ2: Information diffusion analysis...")
    
    from .rq2_diffusion import save_rq2_data
    
    # Save all RQ2 data
    saved_files = save_rq2_data(storage, network, config, config.output_dir)
    
    # Load summary for return value
    import json
    with open(f"{config.output_dir}/rq2_summary.json", 'r') as f:
        summary = json.load(f)
    
    return {
        **summary,
        'saved_files': saved_files,
    }


def run_rq3_analysis(storage: JSONStorage, network, config: Config) -> dict:
    """RQ3: Collective problem-solving analysis."""
    logger.info("Running RQ3: Collective problem-solving analysis...")
    
    from .rq3_collaboration import save_rq3_data
    
    # Save all RQ3 data
    saved_files = save_rq3_data(storage, network, config, config.output_dir)
    
    # Load summary for return value
    import json
    with open(f"{config.output_dir}/rq3_summary.json", 'r') as f:
        summary = json.load(f)
    
    return {
        **summary,
        'saved_files': saved_files,
    }


def run_validation(storage: JSONStorage, features, config: Config) -> dict:
    """Run statistical validation and robustness checks."""
    logger.info("Running validation checks...")
    
    checker = RobustnessChecker(storage, config)
    
    # Clustering robustness
    clustering_robustness = checker.verify_clustering_robustness(features)
    
    return {
        'clustering_robustness': clustering_robustness,
    }


def generate_outputs(config: Config, storage: JSONStorage, results: dict) -> None:
    """Generate all output files."""
    logger.info("Generating outputs...")
    
    generator = OutputGenerator(config)
    
    # Export de-identified dataset
    generator.export_deidentified_dataset(storage)
    
    # Generate README
    readme = generator.generate_readme()
    (Path(config.output_dir) / 'README.md').write_text(readme)
    
    # Generate data dictionary
    dictionary = generator.generate_data_dictionary()
    (Path(config.output_dir) / 'DATA_DICTIONARY.md').write_text(dictionary)
    
    logger.info("Output generation complete")


def main():
    """Main entry point for the analysis pipeline."""
    parser = argparse.ArgumentParser(description='Molt Dynamics Analysis Pipeline')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--dataset-path', type=str, default='moltbook-observatory-archive',
                        help='Path to cloned dataset repository')
    parser.add_argument('--skip-loading', action='store_true',
                        help='Skip data loading phase (use existing data)')
    parser.add_argument('--rq', type=str, nargs='+', choices=['1', '2', '3', 'all'],
                        default=['all'], help='Research questions to analyze')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config.from_yaml(args.config)
    
    # Setup
    setup_logging(config)
    set_random_seeds(config.random_seed)
    
    logger.info("=" * 60)
    logger.info("Molt Dynamics Analysis Pipeline")
    logger.info("=" * 60)
    
    # Initialize JSON storage
    storage = JSONStorage(config)
    storage.connect()
    
    try:
        # Data loading
        if not args.skip_loading:
            run_data_loading(config, storage, args.dataset_path)
        
        # Build networks
        builder, network = run_network_analysis(storage)
        
        # Extract features
        features, features_std = run_feature_extraction(storage, network, config)
        
        results = {}
        rqs = args.rq if 'all' not in args.rq else ['1', '2', '3']
        
        # Run analyses
        if '1' in rqs:
            results['rq1'] = run_rq1_analysis(features_std, config)
        
        if '2' in rqs:
            results['rq2'] = run_rq2_analysis(storage, network, config)
        
        if '3' in rqs:
            results['rq3'] = run_rq3_analysis(storage, network, config)
        
        # Validation
        results['validation'] = run_validation(storage, features, config)
        
        # Generate outputs
        generate_outputs(config, storage, results)
        
        logger.info("=" * 60)
        logger.info("Analysis complete!")
        logger.info("=" * 60)
        
        return results
        
    finally:
        storage.close()


if __name__ == '__main__':
    main()
