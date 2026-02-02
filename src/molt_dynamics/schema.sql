-- Molt Dynamics Analysis - PostgreSQL Schema
-- Normalized schema for MoltBook data storage

-- Drop existing tables if they exist (for clean setup)
DROP TABLE IF EXISTS agent_submolt_membership CASCADE;
DROP TABLE IF EXISTS interactions CASCADE;
DROP TABLE IF EXISTS comments CASCADE;
DROP TABLE IF EXISTS posts CASCADE;
DROP TABLE IF EXISTS submolts CASCADE;
DROP TABLE IF EXISTS agents CASCADE;

-- Agents table: stores anonymized agent information
CREATE TABLE agents (
    agent_id VARCHAR(16) PRIMARY KEY,  -- SHA-256 hash (first 16 chars)
    username VARCHAR(255),              -- Original username (internal only)
    join_date TIMESTAMP,
    bio TEXT,
    post_count INTEGER DEFAULT 0,
    comment_count INTEGER DEFAULT 0,
    karma INTEGER DEFAULT 0,
    first_seen TIMESTAMP,
    last_seen TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Submolts table: topic-specific communities
CREATE TABLE submolts (
    name VARCHAR(255) PRIMARY KEY,
    description TEXT,
    member_count INTEGER DEFAULT 0,
    post_count INTEGER DEFAULT 0,
    created_at TIMESTAMP
);

-- Posts table: top-level discussions
CREATE TABLE posts (
    post_id VARCHAR(64) PRIMARY KEY,
    author_id VARCHAR(16) REFERENCES agents(agent_id) ON DELETE SET NULL,
    title TEXT NOT NULL,
    body TEXT,
    submolt VARCHAR(255) REFERENCES submolts(name) ON DELETE SET NULL,
    upvotes INTEGER DEFAULT 0,
    downvotes INTEGER DEFAULT 0,
    created_at TIMESTAMP,
    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Comments table: threaded replies
CREATE TABLE comments (
    comment_id VARCHAR(64) PRIMARY KEY,
    post_id VARCHAR(64) REFERENCES posts(post_id) ON DELETE CASCADE,
    author_id VARCHAR(16) REFERENCES agents(agent_id) ON DELETE SET NULL,
    parent_comment_id VARCHAR(64) REFERENCES comments(comment_id) ON DELETE SET NULL,
    body TEXT,
    upvotes INTEGER DEFAULT 0,
    downvotes INTEGER DEFAULT 0,
    created_at TIMESTAMP,
    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Interactions table: derived reply relationships
CREATE TABLE interactions (
    id SERIAL PRIMARY KEY,
    source_agent_id VARCHAR(16) REFERENCES agents(agent_id) ON DELETE CASCADE,
    target_agent_id VARCHAR(16) REFERENCES agents(agent_id) ON DELETE CASCADE,
    interaction_type VARCHAR(32),  -- 'reply_to_post', 'reply_to_comment'
    post_id VARCHAR(64) REFERENCES posts(post_id) ON DELETE CASCADE,
    comment_id VARCHAR(64) REFERENCES comments(comment_id) ON DELETE SET NULL,
    timestamp TIMESTAMP
);

-- Agent-submolt membership: bipartite relationship
CREATE TABLE agent_submolt_membership (
    agent_id VARCHAR(16) REFERENCES agents(agent_id) ON DELETE CASCADE,
    submolt_name VARCHAR(255) REFERENCES submolts(name) ON DELETE CASCADE,
    post_count INTEGER DEFAULT 0,
    first_post TIMESTAMP,
    last_post TIMESTAMP,
    PRIMARY KEY (agent_id, submolt_name)
);

-- Indexes for efficient querying
CREATE INDEX idx_posts_author ON posts(author_id);
CREATE INDEX idx_posts_submolt ON posts(submolt);
CREATE INDEX idx_posts_created ON posts(created_at);

CREATE INDEX idx_comments_post ON comments(post_id);
CREATE INDEX idx_comments_author ON comments(author_id);
CREATE INDEX idx_comments_parent ON comments(parent_comment_id);
CREATE INDEX idx_comments_created ON comments(created_at);

CREATE INDEX idx_interactions_source ON interactions(source_agent_id);
CREATE INDEX idx_interactions_target ON interactions(target_agent_id);
CREATE INDEX idx_interactions_timestamp ON interactions(timestamp);
CREATE INDEX idx_interactions_type ON interactions(interaction_type);

CREATE INDEX idx_membership_agent ON agent_submolt_membership(agent_id);
CREATE INDEX idx_membership_submolt ON agent_submolt_membership(submolt_name);

CREATE INDEX idx_agents_join_date ON agents(join_date);
CREATE INDEX idx_agents_first_seen ON agents(first_seen);
CREATE INDEX idx_agents_last_seen ON agents(last_seen);

-- Comments for documentation
COMMENT ON TABLE agents IS 'Anonymized agent profiles from MoltBook';
COMMENT ON TABLE posts IS 'Top-level discussions created by agents';
COMMENT ON TABLE comments IS 'Threaded replies to posts or other comments';
COMMENT ON TABLE interactions IS 'Derived interaction records for network analysis';
COMMENT ON TABLE submolts IS 'Topic-specific communities (like subreddits)';
COMMENT ON TABLE agent_submolt_membership IS 'Bipartite agent-submolt relationships';

COMMENT ON COLUMN agents.agent_id IS 'SHA-256 hash of original ID (first 16 chars)';
COMMENT ON COLUMN interactions.interaction_type IS 'Either reply_to_post or reply_to_comment';
