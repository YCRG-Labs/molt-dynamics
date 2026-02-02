"""Output generation module for Molt Dynamics analysis.

Produces publication-ready LaTeX tables, figures, and dataset exports.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt

from .config import Config
from .storage import JSONStorage, anonymize_agent_id

logger = logging.getLogger(__name__)


class OutputGenerator:
    """Generates publication-ready outputs."""
    
    def __init__(self, config: Config) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_latex_table(
        self,
        df: pd.DataFrame,
        caption: str,
        label: str,
        float_format: str = '%.3f',
    ) -> str:
        """Generate LaTeX table from DataFrame.
        
        Args:
            df: DataFrame to convert.
            caption: Table caption.
            label: LaTeX label for referencing.
            float_format: Format string for floats.
            
        Returns:
            LaTeX table string.
        """
        # Convert to LaTeX
        latex = df.to_latex(
            index=True,
            float_format=float_format,
            escape=True,
        )
        
        # Wrap in table environment
        table = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
{latex}
\\end{{table}}
"""
        return table
    
    def save_table(self, content: str, filename: str) -> None:
        """Save LaTeX table to file.
        
        Args:
            content: LaTeX content.
            filename: Output filename.
        """
        filepath = self.output_dir / filename
        filepath.write_text(content)
        logger.info(f"Saved table to {filepath}")
    
    def save_figure(
        self,
        fig: plt.Figure,
        filename: str,
        formats: list = None,
    ) -> None:
        """Save figure in multiple formats.
        
        Args:
            fig: Matplotlib figure.
            filename: Base filename (without extension).
            formats: List of formats ('png', 'pdf', 'svg').
        """
        if formats is None:
            formats = ['png', 'pdf']
        
        for fmt in formats:
            filepath = self.output_dir / f"{filename}.{fmt}"
            fig.savefig(
                filepath,
                dpi=self.config.figure_dpi,
                bbox_inches='tight',
                format=fmt,
            )
            logger.info(f"Saved figure to {filepath}")
    
    def export_deidentified_dataset(
        self,
        storage: JSONStorage,
        output_path: str = None,
    ) -> None:
        """Export de-identified dataset.
        
        Args:
            storage: JSONStorage instance.
            output_path: Output directory path.
        """
        if output_path is None:
            output_path = self.output_dir / 'dataset'
        else:
            output_path = Path(output_path)
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export agents (already anonymized in storage)
        agents = storage.get_agents()
        agents_df = pd.DataFrame([
            {
                'agent_id': a.agent_id,  # Already hashed
                'join_date': a.join_date,
                'post_count': a.post_count,
                'comment_count': a.comment_count,
                'karma': a.karma,
            }
            for a in agents
        ])
        agents_df.to_csv(output_path / 'agents.csv', index=False)
        
        # Export posts
        posts = storage.get_posts()
        posts_df = pd.DataFrame([
            {
                'post_id': p.post_id,
                'author_id': p.author_id,  # Already hashed
                'submolt': p.submolt,
                'created_at': p.created_at,
                'upvotes': p.upvotes,
                'downvotes': p.downvotes,
            }
            for p in posts
        ])
        posts_df.to_csv(output_path / 'posts.csv', index=False)
        
        # Export interactions
        interactions = storage.get_interactions()
        interactions_df = pd.DataFrame([
            {
                'source_agent_id': i.source_agent_id,
                'target_agent_id': i.target_agent_id,
                'interaction_type': i.interaction_type,
                'timestamp': i.timestamp,
            }
            for i in interactions
        ])
        interactions_df.to_csv(output_path / 'interactions.csv', index=False)
        
        logger.info(f"Exported de-identified dataset to {output_path}")
    
    def generate_readme(self) -> str:
        """Generate README for reproduction.
        
        Returns:
            README content string.
        """
        readme = """# Molt Dynamics Analysis - Reproduction Guide

## Overview
This repository contains the analysis code and data for studying emergent social
phenomena in AI agent networks on MoltBook.

## Requirements
- Python 3.10+
- PostgreSQL 14+
- See requirements.txt for Python dependencies

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Configure database in config/default.yaml
3. Set MOLTBOOK_API_KEY in .env file

## Running the Analysis
```bash
python -m molt_dynamics.main --config config/default.yaml
```

## Output Files
- output/tables/ - LaTeX tables for paper
- output/figures/ - PNG and PDF figures
- output/dataset/ - De-identified dataset

## Reproducibility
All analyses use random seed 42 for reproducibility.
"""
        return readme
    
    def generate_data_dictionary(self) -> str:
        """Generate data dictionary documentation.
        
        Returns:
            Data dictionary content string.
        """
        dictionary = """# Data Dictionary

## agents.csv
- agent_id: Anonymized agent identifier (SHA-256 hash, first 16 chars)
- join_date: Date agent joined MoltBook
- post_count: Total number of posts
- comment_count: Total number of comments
- karma: Agent's karma score

## posts.csv
- post_id: Unique post identifier
- author_id: Anonymized author identifier
- submolt: Community where post was made
- created_at: Post creation timestamp
- upvotes: Number of upvotes
- downvotes: Number of downvotes

## interactions.csv
- source_agent_id: Agent who initiated interaction
- target_agent_id: Agent who received interaction
- interaction_type: Type of interaction (reply_to_post, reply_to_comment)
- timestamp: Interaction timestamp

## Feature Definitions
- topic_entropy: Shannon entropy of submolt participation
- normalized_entropy: Entropy normalized to [0, 1]
- betweenness: Betweenness centrality in interaction network
- pagerank: PageRank centrality (damping=0.85)
- burst_coefficient: Coefficient of variation of inter-event times
"""
        return dictionary


def validate_deidentified_export(export_path: str) -> bool:
    """Validate that export contains no original identifiers.
    
    Args:
        export_path: Path to exported dataset.
        
    Returns:
        True if validation passes.
    """
    export_path = Path(export_path)
    
    # Check each CSV file
    for csv_file in export_path.glob('*.csv'):
        df = pd.read_csv(csv_file)
        
        # Check for any columns that might contain original IDs
        for col in df.columns:
            if 'id' in col.lower():
                # All IDs should be 16-char hex strings (hashed)
                for value in df[col].dropna():
                    if isinstance(value, str):
                        # Should be 16 hex characters
                        if len(value) != 16 or not all(c in '0123456789abcdef' for c in value):
                            # Could be a post_id or other non-agent ID
                            if 'agent' in col.lower():
                                logger.warning(f"Potential unhashed ID in {csv_file}: {value}")
                                return False
    
    return True
