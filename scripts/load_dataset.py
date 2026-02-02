#!/usr/bin/env python3
"""
Load MoltBook data from Hugging Face dataset repository.

This script loads pre-collected MoltBook data from the Hugging Face dataset
instead of scraping from the API.

Dataset: https://huggingface.co/datasets/SimulaMet/moltbook-observatory-archive

Usage:
    # Clone the dataset first
    git clone https://huggingface.co/datasets/SimulaMet/moltbook-observatory-archive

    # Then load the data
    python scripts/load_dataset.py --dataset-path moltbook-observatory-archive
    python scripts/load_dataset.py --dataset-path moltbook-observatory-archive --max-posts 10000
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from molt_dynamics.config import Config
from molt_dynamics.storage import JSONStorage
from molt_dynamics.dataset_loader import MoltBookDatasetLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_loading.log'),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Load MoltBook data from Hugging Face dataset'
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        required=True,
        help='Path to cloned dataset repository'
    )
    parser.add_argument(
        '--max-agents',
        type=int,
        default=None,
        help='Maximum number of agents to load (default: all)'
    )
    parser.add_argument(
        '--max-posts',
        type=int,
        default=None,
        help='Maximum number of posts to load (default: all)'
    )
    parser.add_argument(
        '--max-comments',
        type=int,
        default=None,
        help='Maximum number of comments to load (default: all)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Override output directory from config'
    )
    
    args = parser.parse_args()
    
    # Validate dataset path
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}")
        logger.error("Please clone it first:")
        logger.error("git clone https://huggingface.co/datasets/SimulaMet/moltbook-observatory-archive")
        sys.exit(1)
    
    # Load config
    config = Config.from_yaml(args.config)
    
    # Override output dir if specified
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # Initialize storage
    storage = JSONStorage(config)
    storage.connect()
    
    try:
        # Initialize dataset loader
        loader = MoltBookDatasetLoader(config, storage, str(dataset_path))
        
        # Load all data
        logger.info("=" * 60)
        logger.info("Starting dataset loading")
        logger.info(f"Dataset path: {dataset_path}")
        logger.info(f"Output directory: {config.output_dir}")
        if args.max_agents:
            logger.info(f"Max agents: {args.max_agents}")
        if args.max_posts:
            logger.info(f"Max posts: {args.max_posts}")
        if args.max_comments:
            logger.info(f"Max comments: {args.max_comments}")
        logger.info("=" * 60)
        
        results = loader.load_all(
            max_agents=args.max_agents,
            max_posts=args.max_posts,
            max_comments=args.max_comments
        )
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("LOADING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Submolts loaded: {results['submolts']}")
        logger.info(f"Agents loaded: {results['agents']}")
        logger.info(f"Posts loaded: {results['posts']}")
        logger.info(f"Comments loaded: {results['comments']}")
        logger.info(f"Duration: {results['duration_seconds']:.2f} seconds")
        logger.info(f"Data saved to: {config.output_dir}/data/")
        logger.info("=" * 60)
    
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("\nLoading interrupted by user")
        storage.save()
    
    except Exception as e:
        logger.error(f"Error during loading: {e}", exc_info=True)
        sys.exit(1)
    
    finally:
        storage.close()
        logger.info("Storage closed")


if __name__ == '__main__':
    main()
