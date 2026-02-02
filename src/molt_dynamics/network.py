"""Network construction module for Molt Dynamics analysis.

Builds interaction and affiliation networks from MoltBook data using NetworkX.
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

import networkx as nx
import pandas as pd

from .storage import JSONStorage
from .models import Interaction

logger = logging.getLogger(__name__)


class NetworkBuilder:
    """Constructs interaction and affiliation networks from MoltBook data."""
    
    def __init__(self, storage: JSONStorage) -> None:
        """Initialize network builder.
        
        Args:
            storage: JSONStorage instance for querying interaction data.
        """
        self.storage = storage
    
    def build_interaction_network(
        self,
        until_time: Optional[datetime] = None,
        directed: bool = True,
    ) -> nx.DiGraph | nx.Graph:
        """Build interaction network from reply relationships.
        
        If no comment-based interactions exist, falls back to building
        a co-posting network based on agents posting in the same submolts.
        
        Args:
            until_time: Optional cutoff time for temporal snapshot.
            directed: If True, return directed graph; otherwise undirected.
            
        Returns:
            NetworkX graph with agents as nodes and replies as weighted edges.
        """
        # Get interactions from storage
        if until_time:
            interactions = self.storage.get_interactions(
                time_range=(datetime.min, until_time)
            )
        else:
            interactions = self.storage.get_interactions()
        
        # Count interactions between agent pairs
        edge_counts: dict[tuple[str, str], int] = defaultdict(int)
        
        for interaction in interactions:
            source = interaction.source_agent_id
            target = interaction.target_agent_id
            edge_counts[(source, target)] += 1
        
        # If no interactions, build co-posting network from posts
        if not edge_counts:
            logger.info("No comment-based interactions found, building co-posting network")
            return self._build_coposting_network(until_time, directed)
        
        # Build graph
        if directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        
        # Add edges with weights
        for (source, target), weight in edge_counts.items():
            if directed:
                G.add_edge(source, target, weight=weight)
            else:
                # For undirected, sum bidirectional weights
                if G.has_edge(source, target):
                    G[source][target]['weight'] += weight
                else:
                    G.add_edge(source, target, weight=weight)
        
        logger.info(
            f"Built {'directed' if directed else 'undirected'} network: "
            f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
        )
        
        return G
    
    def _build_coposting_network(
        self,
        until_time: Optional[datetime] = None,
        directed: bool = True,
    ) -> nx.DiGraph | nx.Graph:
        """Build network from co-posting relationships in submolts.
        
        Agents are connected if they post in the same submolt, with edge
        weights based on the number of shared submolts and post counts.
        
        Args:
            until_time: Optional cutoff time for temporal snapshot.
            directed: If True, return directed graph; otherwise undirected.
            
        Returns:
            NetworkX graph with agents as nodes and co-posting as edges.
        """
        posts = self.storage.get_posts()
        
        # Filter by time if specified
        if until_time:
            posts = [p for p in posts if p.created_at and p.created_at <= until_time]
        
        # Group posts by submolt
        submolt_agents: dict[str, list[tuple[str, datetime]]] = defaultdict(list)
        for post in posts:
            if post.submolt and post.author_id:
                submolt_agents[post.submolt].append(
                    (post.author_id, post.created_at or datetime.now())
                )
        
        # Build edges: agents who post in the same submolt are connected
        # For directed: earlier poster -> later poster (temporal influence)
        edge_counts: dict[tuple[str, str], int] = defaultdict(int)
        
        for submolt, agent_posts in submolt_agents.items():
            # Sort by time
            agent_posts.sort(key=lambda x: x[1])
            
            # Get unique agents in this submolt
            agents_in_submolt = list(dict.fromkeys([a for a, _ in agent_posts]))
            
            if len(agents_in_submolt) < 2:
                continue
            
            # Create edges between agents in same submolt
            for i, agent1 in enumerate(agents_in_submolt):
                for agent2 in agents_in_submolt[i+1:]:
                    if agent1 != agent2:
                        if directed:
                            # Earlier poster influences later poster
                            edge_counts[(agent1, agent2)] += 1
                        else:
                            edge_counts[(agent1, agent2)] += 1
        
        # Build graph
        if directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        
        # Add all agents as nodes first
        all_agents = set()
        for post in posts:
            if post.author_id:
                all_agents.add(post.author_id)
        
        for agent in all_agents:
            G.add_node(agent)
        
        # Add edges with weights
        for (source, target), weight in edge_counts.items():
            if directed:
                G.add_edge(source, target, weight=weight)
            else:
                if G.has_edge(source, target):
                    G[source][target]['weight'] += weight
                else:
                    G.add_edge(source, target, weight=weight)
        
        logger.info(
            f"Built co-posting {'directed' if directed else 'undirected'} network: "
            f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
        )
        
        return G
    
    def build_interaction_network_from_dataframe(
        self,
        interactions_df: pd.DataFrame,
        until_time: Optional[datetime] = None,
        directed: bool = True,
    ) -> nx.DiGraph | nx.Graph:
        """Build interaction network from DataFrame.
        
        Args:
            interactions_df: DataFrame with source_agent_id, target_agent_id, timestamp.
            until_time: Optional cutoff time for temporal snapshot.
            directed: If True, return directed graph; otherwise undirected.
            
        Returns:
            NetworkX graph with agents as nodes and replies as weighted edges.
        """
        df = interactions_df.copy()
        
        # Apply temporal filter
        if until_time:
            df = df[df['timestamp'] <= until_time]
        
        # Count interactions between agent pairs
        edge_counts = df.groupby(['source_agent_id', 'target_agent_id']).size()
        
        # Build graph
        if directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        
        for (source, target), weight in edge_counts.items():
            if directed:
                G.add_edge(source, target, weight=weight)
            else:
                if G.has_edge(source, target):
                    G[source][target]['weight'] += weight
                else:
                    G.add_edge(source, target, weight=weight)
        
        return G
    
    def convert_to_undirected(self, G: nx.DiGraph) -> nx.Graph:
        """Convert directed network to undirected by summing bidirectional weights.
        
        Args:
            G: Directed graph.
            
        Returns:
            Undirected graph with summed edge weights.
        """
        H = nx.Graph()
        
        for u, v, data in G.edges(data=True):
            weight = data.get('weight', 1)
            
            if H.has_edge(u, v):
                H[u][v]['weight'] += weight
            else:
                H.add_edge(u, v, weight=weight)
        
        return H
    
    def get_temporal_snapshots(
        self,
        interval: timedelta,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> list[tuple[datetime, nx.DiGraph]]:
        """Generate temporal network snapshots at regular intervals.
        
        Args:
            interval: Time interval between snapshots.
            start_time: Start of observation period.
            end_time: End of observation period.
            
        Returns:
            List of (timestamp, network) tuples.
        """
        # Get all interactions
        interactions = self.storage.get_interactions()
        
        if not interactions:
            return []
        
        # Determine time bounds
        timestamps = [i.timestamp for i in interactions if i.timestamp]
        if not timestamps:
            return []
        
        if start_time is None:
            start_time = min(timestamps)
        if end_time is None:
            end_time = max(timestamps)
        
        # Generate snapshots
        snapshots = []
        current_time = start_time
        
        while current_time <= end_time:
            G = self.build_interaction_network(until_time=current_time)
            snapshots.append((current_time, G))
            current_time += interval
        
        logger.info(f"Generated {len(snapshots)} temporal snapshots")
        return snapshots
    
    def build_submolt_affiliation_network(self) -> nx.Graph:
        """Build bipartite agent-submolt affiliation network.
        
        Returns:
            Bipartite graph connecting agents to submolts where they posted.
        """
        memberships = self.storage.get_agent_submolt_memberships()
        
        B = nx.Graph()
        
        # Add nodes with bipartite attribute
        agents = set()
        submolts = set()
        
        for _, row in memberships.iterrows():
            agent_id = row['agent_id']
            submolt_name = row['submolt_name']
            post_count = row.get('post_count', 1)
            
            agents.add(agent_id)
            submolts.add(submolt_name)
            
            B.add_edge(agent_id, submolt_name, weight=post_count)
        
        # Set bipartite attribute
        for agent in agents:
            B.nodes[agent]['bipartite'] = 0  # Agents
        for submolt in submolts:
            B.nodes[submolt]['bipartite'] = 1  # Submolts
        
        logger.info(
            f"Built bipartite network: {len(agents)} agents, "
            f"{len(submolts)} submolts, {B.number_of_edges()} edges"
        )
        
        return B
    
    def project_agent_similarity(
        self,
        bipartite: nx.Graph,
        weight_func: str = 'jaccard',
    ) -> nx.Graph:
        """Project agent-agent similarity from bipartite network.
        
        Args:
            bipartite: Bipartite agent-submolt network.
            weight_func: Similarity function ('jaccard', 'overlap', 'weighted').
            
        Returns:
            Agent-agent similarity network.
        """
        # Get agent nodes (bipartite=0)
        agents = [n for n, d in bipartite.nodes(data=True) if d.get('bipartite') == 0]
        
        # Build similarity network
        G = nx.Graph()
        
        for i, agent1 in enumerate(agents):
            neighbors1 = set(bipartite.neighbors(agent1))
            
            for agent2 in agents[i+1:]:
                neighbors2 = set(bipartite.neighbors(agent2))
                
                # Compute similarity
                intersection = neighbors1 & neighbors2
                
                if not intersection:
                    continue
                
                if weight_func == 'jaccard':
                    union = neighbors1 | neighbors2
                    similarity = len(intersection) / len(union) if union else 0
                elif weight_func == 'overlap':
                    min_size = min(len(neighbors1), len(neighbors2))
                    similarity = len(intersection) / min_size if min_size else 0
                elif weight_func == 'weighted':
                    # Sum of minimum weights for shared submolts
                    similarity = sum(
                        min(
                            bipartite[agent1][s].get('weight', 1),
                            bipartite[agent2][s].get('weight', 1)
                        )
                        for s in intersection
                    )
                else:
                    similarity = len(intersection)
                
                if similarity > 0:
                    G.add_edge(agent1, agent2, weight=similarity)
        
        logger.info(
            f"Projected similarity network: {G.number_of_nodes()} agents, "
            f"{G.number_of_edges()} edges"
        )
        
        return G
    
    def get_network_statistics(self, G: nx.Graph | nx.DiGraph) -> dict:
        """Compute basic network statistics.
        
        Args:
            G: NetworkX graph.
            
        Returns:
            Dict with network statistics.
        """
        stats = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
        }
        
        if G.number_of_nodes() > 0:
            if isinstance(G, nx.DiGraph):
                stats['avg_in_degree'] = sum(d for _, d in G.in_degree()) / G.number_of_nodes()
                stats['avg_out_degree'] = sum(d for _, d in G.out_degree()) / G.number_of_nodes()
            else:
                stats['avg_degree'] = sum(d for _, d in G.degree()) / G.number_of_nodes()
            
            # Connected components
            if isinstance(G, nx.DiGraph):
                stats['num_weakly_connected'] = nx.number_weakly_connected_components(G)
                stats['num_strongly_connected'] = nx.number_strongly_connected_components(G)
            else:
                stats['num_connected_components'] = nx.number_connected_components(G)
        
        return stats
