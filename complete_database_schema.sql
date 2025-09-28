-- ================================================
-- COMPLETE DATABASE SCHEMA FOR CODESAGE AI
-- ================================================
-- Run this in Supabase SQL Editor to create/update all tables

-- Drop existing tables (in correct order to handle foreign keys)
DROP TABLE IF EXISTS audio_chunks;
DROP TABLE IF EXISTS code_history;
DROP TABLE IF EXISTS execution_results;
DROP TABLE IF EXISTS candidates;
DROP TABLE IF EXISTS sessions;

-- ================================================
-- CREATE ALL TABLES
-- ================================================

-- 1. SESSIONS TABLE
CREATE TABLE IF NOT EXISTS sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_activity TIMESTAMPTZ DEFAULT NOW(),
    disconnected_at TIMESTAMPTZ DEFAULT NULL
);

-- 2. CANDIDATES TABLE (for storing candidate information)
CREATE TABLE IF NOT EXISTS candidates (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    skills TEXT,
    level VARCHAR(50),
    photo_url TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

-- 3. CODE_HISTORY TABLE
CREATE TABLE IF NOT EXISTS code_history (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    code TEXT NOT NULL,
    cursor_position INTEGER DEFAULT 0,
    line_count INTEGER DEFAULT 0,
    char_count INTEGER DEFAULT 0,
    code_hash VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

-- 4. EXECUTION_RESULTS TABLE
CREATE TABLE IF NOT EXISTS execution_results (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    code TEXT,
    code_hash VARCHAR(255),
    success BOOLEAN DEFAULT FALSE,
    output TEXT,
    error TEXT,
    execution_time VARCHAR(50),
    security_level VARCHAR(50),
    language VARCHAR(20) DEFAULT 'python',
    memory_usage VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

-- 5. AUDIO_CHUNKS TABLE (with all required columns)
CREATE TABLE IF NOT EXISTS audio_chunks (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    audio_data TEXT,  -- Base64 encoded audio
    format VARCHAR(20) DEFAULT 'webm',
    duration DECIMAL DEFAULT 0,
    size_kb DECIMAL DEFAULT 0,
    size_bytes INTEGER DEFAULT 0,
    audio_hash VARCHAR(255),
    chunk_index INTEGER,
    
    -- Processing status fields
    processing_status VARCHAR(20) DEFAULT 'pending' CHECK (processing_status IN ('pending', 'processing', 'completed', 'failed')),
    processing_started_at TIMESTAMPTZ DEFAULT NULL,
    processing_completed_at TIMESTAMPTZ DEFAULT NULL,
    processing_error TEXT DEFAULT NULL,
    
    -- Transcript fields
    transcript TEXT,
    transcript_confidence DECIMAL,
    transcript_language VARCHAR(10) DEFAULT 'en',
    transcript_service VARCHAR(50),
    transcript_metadata JSONB,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    cleanup_date TIMESTAMPTZ DEFAULT NULL,
    
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

-- ================================================
-- CREATE INDEXES FOR PERFORMANCE
-- ================================================

-- Sessions indexes
CREATE INDEX IF NOT EXISTS idx_sessions_session_id ON sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_is_active ON sessions(is_active);
CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at);

-- Candidates indexes
CREATE INDEX IF NOT EXISTS idx_candidates_session_id ON candidates(session_id);
CREATE INDEX IF NOT EXISTS idx_candidates_name ON candidates(name);
CREATE INDEX IF NOT EXISTS idx_candidates_created_at ON candidates(created_at);

-- Code history indexes
CREATE INDEX IF NOT EXISTS idx_code_history_session_id ON code_history(session_id);
CREATE INDEX IF NOT EXISTS idx_code_history_created_at ON code_history(created_at);

-- Execution results indexes
CREATE INDEX IF NOT EXISTS idx_execution_results_session_id ON execution_results(session_id);
CREATE INDEX IF NOT EXISTS idx_execution_results_created_at ON execution_results(created_at);

-- Audio chunks indexes
CREATE INDEX IF NOT EXISTS idx_audio_chunks_session_id ON audio_chunks(session_id);
CREATE INDEX IF NOT EXISTS idx_audio_chunks_processing_status ON audio_chunks(processing_status);
CREATE INDEX IF NOT EXISTS idx_audio_chunks_created_at ON audio_chunks(created_at);

-- ================================================
-- ENABLE ROW LEVEL SECURITY
-- ================================================
ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE candidates ENABLE ROW LEVEL SECURITY;
ALTER TABLE code_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE execution_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE audio_chunks ENABLE ROW LEVEL SECURITY;

-- ================================================
-- CREATE RLS POLICIES
-- ================================================
-- Allow all operations for authenticated users (adjust as needed)
CREATE POLICY "Allow all for authenticated" ON sessions FOR ALL USING (true);
CREATE POLICY "Allow all for authenticated" ON candidates FOR ALL USING (true);
CREATE POLICY "Allow all for authenticated" ON code_history FOR ALL USING (true);
CREATE POLICY "Allow all for authenticated" ON execution_results FOR ALL USING (true);
CREATE POLICY "Allow all for authenticated" ON audio_chunks FOR ALL USING (true);

-- ================================================
-- VERIFY SCHEMA
-- ================================================
SELECT 
    table_name,
    COUNT(*) as column_count
FROM information_schema.columns
WHERE table_schema = 'public' 
    AND table_name IN ('sessions', 'candidates', 'code_history', 'execution_results', 'audio_chunks')
GROUP BY table_name
ORDER BY table_name;

-- ================================================
-- SAMPLE DATA (Optional - for testing)
-- ================================================
-- Uncomment the following lines to insert sample data for testing

/*
-- Insert a sample session
INSERT INTO sessions (session_id, is_active) VALUES ('test-session-123', true);

-- Insert a sample candidate
INSERT INTO candidates (session_id, name, skills, level, photo_url) 
VALUES ('test-session-123', 'John Doe', 'Python, JavaScript, React', 'Senior', 'https://example.com/photo.jpg');
*/
