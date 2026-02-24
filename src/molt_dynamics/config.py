"""Configuration management for Molt Dynamics analysis pipeline.

Handles loading configuration from environment variables, config files,
and command-line arguments with proper precedence.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os
import yaml


@dataclass
class Config:
    """Central configuration for all analysis parameters.
    
    Attributes match paper specifications with sensible defaults.
    Command-line arguments take precedence over config file values.
    """
    
    # Database settings (legacy, not used with JSON storage)
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "moltbook"
    db_user: str = "postgres"
    db_password: str = ""
    
    # Analysis parameters
    random_seed: int = 42
    lda_topics: int = 20
    kmeans_k_range: tuple[int, int] = (3, 8)
    kmeans_n_init: int = 100
    pagerank_damping: float = 0.85
    bootstrap_iterations: int = 1000
    significance_level: float = 0.05
    
    # Parallelization (-1 = use all CPUs)
    n_jobs: int = -1
    
    # Cascade identification
    min_cascade_adopters: int = 5
    ngram_range: tuple[int, int] = (2, 5)
    
    # Collaboration identification
    min_collab_agents: int = 3
    min_collab_comments: int = 5
    min_collab_duration_minutes: int = 30
    
    # Phase transition binning
    network_size_bins: list[int] = field(
        default_factory=lambda: [50, 100, 200, 500, 1000, 2000, 5000]
    )
    
    # Output settings
    figure_dpi: int = 300
    output_dir: str = "output"
    figure_formats: list[str] = field(default_factory=lambda: ["png", "pdf"])
    
    # Logging
    log_file: str = "molt_analysis.log"
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration with values from .env file.
        
        Returns:
            Config instance with environment variables applied.
        """
        from dotenv import load_dotenv
        load_dotenv()
        
        return cls(
            db_host=os.getenv("DB_HOST", "localhost"),
            db_port=int(os.getenv("DB_PORT", "5432")),
            db_name=os.getenv("DB_NAME", "moltbook"),
            db_user=os.getenv("DB_USER", "postgres"),
            db_password=os.getenv("DB_PASSWORD", ""),
            random_seed=int(os.getenv("RANDOM_SEED", "42")),
            output_dir=os.getenv("OUTPUT_DIR", "output"),
        )
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from YAML file.
        
        Args:
            path: Path to YAML configuration file.
            
        Returns:
            Config instance with YAML values applied.
        """
        path = Path(path)
        if not path.exists():
            return cls.from_env()
        
        with open(path) as f:
            yaml_config = yaml.safe_load(f) or {}
        
        # Start with env config, then override with YAML
        config = cls.from_env()
        
        for key, value in yaml_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    def override_from_args(self, **kwargs) -> "Config":
        """Override configuration with command-line arguments.
        
        Args:
            **kwargs: Key-value pairs to override.
            
        Returns:
            New Config instance with overrides applied.
        """
        from dataclasses import asdict
        
        current = asdict(self)
        for key, value in kwargs.items():
            if value is not None and key in current:
                current[key] = value
        
        return Config(**current)
    
    def get_db_connection_string(self) -> str:
        """Generate PostgreSQL connection string.
        
        Returns:
            Connection string for psycopg2.
        """
        if self.db_password:
            return (
                f"host={self.db_host} port={self.db_port} "
                f"dbname={self.db_name} user={self.db_user} "
                f"password={self.db_password}"
            )
        return (
            f"host={self.db_host} port={self.db_port} "
            f"dbname={self.db_name} user={self.db_user}"
        )
    
    def ensure_output_dir(self) -> Path:
        """Create output directory if it doesn't exist.
        
        Returns:
            Path to output directory.
        """
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path
    
    def validate(self) -> list[str]:
        """Validate configuration values.
        
        Returns:
            List of validation error messages (empty if valid).
        """
        errors = []
        
        if self.random_seed < 0:
            errors.append("random_seed must be non-negative")
        
        if self.kmeans_k_range[0] > self.kmeans_k_range[1]:
            errors.append("kmeans_k_range[0] must be <= kmeans_k_range[1]")
        
        if not 0 < self.pagerank_damping < 1:
            errors.append("pagerank_damping must be between 0 and 1")
        
        if not 0 < self.significance_level < 1:
            errors.append("significance_level must be between 0 and 1")
        
        if self.bootstrap_iterations < 100:
            errors.append("bootstrap_iterations should be at least 100")
        
        return errors
