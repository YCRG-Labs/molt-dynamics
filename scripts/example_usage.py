#!/usr/bin/env python3
"""
Example usage of the MoltBook dataset loader.

Demonstrates how to load and explore the dataset.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from molt_dynamics.config import Config
from molt_dynamics.storage import JSONStorage
from molt_dynamics.dataset_loader import MoltBookDatasetLoader


def main():
    print("=" * 60)
    print("MoltBook Dataset Loader - Example Usage")
    print("=" * 60)
    
    # Initialize configuration
    config = Config.from_yaml("config/default.yaml")
    
    # Initialize storage
    storage = JSONStorage(config)
    storage.connect()
    
    # Initialize dataset loader
    dataset_path = "moltbook-observatory-archive"
    
    try:
        loader = MoltBookDatasetLoader(config, storage, dataset_path)
        
        # Example 1: Load a small sample
        print("\n--- Example 1: Load Sample Data ---")
        results = loader.load_all(
            max_agents=100,
            max_posts=500,
            max_comments=200
        )
        
        print(f"\nLoaded:")
        print(f"  - {results['agents']} agents")
        print(f"  - {results['posts']} posts")
        print(f"  - {results['comments']} comments")
        print(f"  - {results['submolts']} submolts")
        print(f"  - Duration: {results['duration_seconds']:.2f}s")
        
        # Example 2: Query the loaded data
        print("\n--- Example 2: Query Loaded Data ---")
        
        agents = storage.get_agents()
        print(f"\nTotal agents in storage: {len(agents)}")
        
        if agents:
            sample_agent = agents[0]
            print(f"\nSample agent:")
            print(f"  - ID: {sample_agent.agent_id}")
            print(f"  - Username: {sample_agent.username}")
            print(f"  - Karma: {sample_agent.karma}")
            print(f"  - First seen: {sample_agent.first_seen}")
        
        posts = storage.get_posts()
        print(f"\nTotal posts in storage: {len(posts)}")
        
        if posts:
            sample_post = posts[0]
            print(f"\nSample post:")
            print(f"  - ID: {sample_post.post_id}")
            print(f"  - Title: {sample_post.title[:50]}...")
            print(f"  - Submolt: {sample_post.submolt}")
            print(f"  - Score: {sample_post.upvotes}")
            print(f"  - Created: {sample_post.created_at}")
        
        comments = storage.get_comments()
        print(f"\nTotal comments in storage: {len(comments)}")
        
        submolts = storage.get_submolts()
        print(f"\nTotal submolts in storage: {len(submolts)}")
        
        if submolts:
            print(f"\nTop 5 submolts by post count:")
            for i, submolt in enumerate(submolts[:5], 1):
                print(f"  {i}. {submolt.name}: {submolt.post_count} posts")
        
        # Example 3: Get statistics
        print("\n--- Example 3: Storage Statistics ---")
        stats = storage.get_statistics()
        print(f"\nStorage statistics:")
        for key, value in stats.items():
            print(f"  - {key}: {value}")
        
        print(f"\nData saved to: {config.output_dir}/data/")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease clone the dataset first:")
        print("git clone https://huggingface.co/datasets/SimulaMet/moltbook-observatory-archive")
        sys.exit(1)
    
    finally:
        storage.close()
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
