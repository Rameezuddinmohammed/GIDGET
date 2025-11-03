-- Supabase Schema for Multi-Agent Code Intelligence System
-- Execute these commands in the Supabase SQL editor

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- Create repositories table
CREATE TABLE IF NOT EXISTS repositories (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    name VARCHAR NOT NULL,
    url VARCHAR NOT NULL UNIQUE,
    description TEXT,
    language VARCHAR,
    default_branch VARCHAR DEFAULT 'main',
    analysis_status VARCHAR DEFAULT 'pending' CHECK (analysis_status IN ('pending', 'analyzing', 'completed', 'failed')),
    last_analyzed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create analysis_cache table
CREATE TABLE IF NOT EXISTS analysis_cache (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    repository_id UUID REFERENCES repositories(id) ON DELETE CASCADE,
    query_hash VARCHAR NOT NULL,
    query_text TEXT NOT NULL,
    result_data JSONB NOT NULL,
    confidence_score FLOAT CHECK (confidence_score >= 0 AND confidence_score <= 1),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    UNIQUE(repository_id, query_hash)
);

-- Create user_preferences table
CREATE TABLE IF NOT EXISTS user_preferences (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL,
    preferences JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id)
);

-- Create code_embeddings table for semantic search
CREATE TABLE IF NOT EXISTS code_embeddings (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    repository_id UUID REFERENCES repositories(id) ON DELETE CASCADE,
    file_path VARCHAR NOT NULL,
    element_type VARCHAR NOT NULL CHECK (element_type IN ('function', 'class', 'method', 'variable', 'module')),
    element_name VARCHAR NOT NULL,
    code_snippet TEXT NOT NULL,
    embedding vector(768), -- 768-dimensional embeddings (adjust based on model)
    metadata JSONB DEFAULT '{}',
    commit_sha VARCHAR NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create query_history table
CREATE TABLE IF NOT EXISTS query_history (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID,
    repository_id UUID REFERENCES repositories(id) ON DELETE CASCADE,
    query_text TEXT NOT NULL,
    query_type VARCHAR,
    execution_time_ms INTEGER,
    agent_count INTEGER,
    confidence_score FLOAT CHECK (confidence_score >= 0 AND confidence_score <= 1),
    result_summary TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create agent_execution_logs table
CREATE TABLE IF NOT EXISTS agent_execution_logs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    query_id UUID REFERENCES query_history(id) ON DELETE CASCADE,
    agent_name VARCHAR NOT NULL,
    agent_type VARCHAR NOT NULL,
    status VARCHAR NOT NULL CHECK (status IN ('started', 'completed', 'failed', 'timeout')),
    start_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    end_time TIMESTAMP WITH TIME ZONE,
    execution_time_ms INTEGER,
    input_data JSONB,
    output_data JSONB,
    error_message TEXT,
    confidence_score FLOAT CHECK (confidence_score >= 0 AND confidence_score <= 1)
);

-- Create performance indexes
CREATE INDEX IF NOT EXISTS idx_repositories_url ON repositories(url);
CREATE INDEX IF NOT EXISTS idx_repositories_analysis_status ON repositories(analysis_status);
CREATE INDEX IF NOT EXISTS idx_repositories_created_at ON repositories(created_at);

CREATE INDEX IF NOT EXISTS idx_analysis_cache_repository_id ON analysis_cache(repository_id);
CREATE INDEX IF NOT EXISTS idx_analysis_cache_query_hash ON analysis_cache(query_hash);
CREATE INDEX IF NOT EXISTS idx_analysis_cache_expires_at ON analysis_cache(expires_at);

CREATE INDEX IF NOT EXISTS idx_user_preferences_user_id ON user_preferences(user_id);

CREATE INDEX IF NOT EXISTS idx_code_embeddings_repository_id ON code_embeddings(repository_id);
CREATE INDEX IF NOT EXISTS idx_code_embeddings_element_type ON code_embeddings(element_type);
CREATE INDEX IF NOT EXISTS idx_code_embeddings_element_name ON code_embeddings(element_name);
CREATE INDEX IF NOT EXISTS idx_code_embeddings_file_path ON code_embeddings(file_path);
CREATE INDEX IF NOT EXISTS idx_code_embeddings_commit_sha ON code_embeddings(commit_sha);

CREATE INDEX IF NOT EXISTS idx_query_history_user_id ON query_history(user_id);
CREATE INDEX IF NOT EXISTS idx_query_history_repository_id ON query_history(repository_id);
CREATE INDEX IF NOT EXISTS idx_query_history_created_at ON query_history(created_at);

CREATE INDEX IF NOT EXISTS idx_agent_execution_logs_query_id ON agent_execution_logs(query_id);
CREATE INDEX IF NOT EXISTS idx_agent_execution_logs_agent_name ON agent_execution_logs(agent_name);
CREATE INDEX IF NOT EXISTS idx_agent_execution_logs_status ON agent_execution_logs(status);

-- Create vector similarity search index for code embeddings
-- This creates an IVFFlat index for approximate nearest neighbor search
CREATE INDEX IF NOT EXISTS idx_code_embeddings_vector 
ON code_embeddings USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at columns
CREATE TRIGGER update_repositories_updated_at 
    BEFORE UPDATE ON repositories 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_preferences_updated_at 
    BEFORE UPDATE ON user_preferences 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create Row Level Security (RLS) policies
ALTER TABLE repositories ENABLE ROW LEVEL SECURITY;
ALTER TABLE analysis_cache ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_preferences ENABLE ROW LEVEL SECURITY;
ALTER TABLE code_embeddings ENABLE ROW LEVEL SECURITY;
ALTER TABLE query_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_execution_logs ENABLE ROW LEVEL SECURITY;

-- Basic RLS policies (adjust based on your authentication requirements)
-- Allow authenticated users to read repositories
CREATE POLICY "Allow authenticated users to read repositories" ON repositories
    FOR SELECT USING (auth.role() = 'authenticated');

-- Allow authenticated users to manage their own preferences
CREATE POLICY "Users can manage their own preferences" ON user_preferences
    FOR ALL USING (auth.uid() = user_id);

-- Allow authenticated users to read their own query history
CREATE POLICY "Users can read their own query history" ON query_history
    FOR SELECT USING (auth.uid() = user_id);

-- Create helpful views
CREATE OR REPLACE VIEW repository_stats AS
SELECT 
    r.id,
    r.name,
    r.url,
    r.analysis_status,
    r.last_analyzed_at,
    COUNT(ce.id) as embedding_count,
    COUNT(DISTINCT ce.element_type) as element_types,
    COUNT(qh.id) as query_count
FROM repositories r
LEFT JOIN code_embeddings ce ON r.id = ce.repository_id
LEFT JOIN query_history qh ON r.id = qh.repository_id
GROUP BY r.id, r.name, r.url, r.analysis_status, r.last_analyzed_at;

-- Create function for similarity search
CREATE OR REPLACE FUNCTION search_similar_code(
    query_embedding vector(768),
    repository_id_param UUID DEFAULT NULL,
    element_type_param VARCHAR DEFAULT NULL,
    similarity_threshold FLOAT DEFAULT 0.7,
    max_results INTEGER DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    repository_id UUID,
    file_path VARCHAR,
    element_type VARCHAR,
    element_name VARCHAR,
    code_snippet TEXT,
    similarity FLOAT,
    metadata JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ce.id,
        ce.repository_id,
        ce.file_path,
        ce.element_type,
        ce.element_name,
        ce.code_snippet,
        1 - (ce.embedding <=> query_embedding) as similarity,
        ce.metadata
    FROM code_embeddings ce
    WHERE 
        (repository_id_param IS NULL OR ce.repository_id = repository_id_param)
        AND (element_type_param IS NULL OR ce.element_type = element_type_param)
        AND (1 - (ce.embedding <=> query_embedding)) >= similarity_threshold
    ORDER BY ce.embedding <=> query_embedding
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;