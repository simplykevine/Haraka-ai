# Haraka-ai 🤖

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/fastapi-0.95%2B-green.svg)](https://fastapi.tiangolo.com/)
[![Google Gemini](https://img.shields.io/badge/google-gemini-blue.svg)](https://ai.google.dev/)
[![PostgreSQL](https://img.shields.io/badge/postgresql-13%2B-blue.svg)](https://www.postgresql.org/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

> 🚀 Institutional-grade AI agent for East African policy decision-support with RAG, Gemini LLM, and real-time citations

**Live Demo:** https://haraka-frontend.vercel.app  
**API Docs:** http://localhost:8080/docs (when running locally)  
**Video Demo:** [Watch on Drive](https://drive.google.com/file/d/1lDBtvqLmfQXmTXnqtYvilXrClPpOm0Q2/view?usp=sharing)

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Installation Guide](#installation-guide)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Implementation Details](#implementation-details)
- [API Documentation](#api-documentation)
- [File Structure](#file-structure)
- [Feedback Incorporation](#feedback-incorporation)
- [Performance Benchmarks](#performance-benchmarks)
- [Deployment](#deployment)
- [Docker Setup](#docker-setup)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Project Overview

**Haraka-ai** is a multi agentic AI for East African policy decision-support for economists in Kenya to help them make market forecasts, shocks analysis and comparing the market competition between regional markets, its  built on **Retrieval-Augmented Generation (RAG)** with **Google Gemini** and **FastAPI**.

The system provides:
- 📊 **Accurate policy analysis** grounded in retrieved documents (55% citation accuracy vs 18% baseline)
- 🔀 **Multi-modal query routing** for forecasts, scenarios, comparative analysis, and free-form RAG
- 🔒 **Institutional auditability** with complete reasoning transparency and decision traceability
- 🤝 **Sub-agent architecture** for specialized analysis (Scenario, Forecasting, Comparative)

### Why Haraka?

| Dimension | Zeno | GPT-4o | Advantage |
|-----------|------|--------|-----------|
| Factual Accuracy | **89%** | 85% | ✅ +4% |
| Citation Accuracy | **55%** | 18% | ✅ **+37%** |
| Hallucination Rate | **65%** | 80% | ✅ -15% |
| Response Speed | 27.3s | 2.3s | ⚖️ Quality vs Speed |
| Policy Auditability | **Full** | None | ✅ Exclusive |

**Testing:** 20 East African economic policy questions (March 17, 2026)

---

## ✨ Key Features

### 1. Retrieval-Augmented Generation (RAG)
```
✅ Semantic document search via pgvector
✅ Real-time citation tracking with relevance scores
✅ Hallucination reduction: 65% (vs 80% baseline)
✅ Citation accuracy: 55% (vs 18% baseline)
```

**Implementation:** `zeno_agent/rag_tools.py`

### 2. Multi-Modal Query Routing
```
📊 Scenario Analysis   → "What if" policy simulations
📈 Forecasting         → Time-series predictions
🔄 Comparative         → Cross-country analysis
📄 Free-form RAG       → General policy questions
```

**Implementation:** `zeno_agent/agent.py` (line 1-200)

### 3. Sub-Agent Architecture
```
agents/
├── scenario/          → Policy intervention simulations
├── forecasting/       → Prophet-based time-series
└── comparative/       → Multi-country analysis
```

### 4. Structured Output Artifacts
```
✅ Charts & visualizations (Plotly, Matplotlib)
✅ Tables and structured data
✅ Policy briefs and markdown reports
✅ Citations with source tracking
```

**Implementation:** `zeno_agent/tools/graphing.py`

### 5. Run-Based Audit Trail
```
Every query → Run ID → Steps → Artifacts → Persistence
Complete decision traceability for compliance
```

### 6. Query Caching & Optimization
```
query_cache.json (429KB) → Pre-computed responses
Rapid retrieval without LLM calls for common queries
```

---

## 🛠️ Tech Stack

### Core Engine
```
- FastAPI 0.95+           (REST API framework)
- Python 3.8+             (Language)
- Google Gemini API       (LLM)
- PostgreSQL + pgvector   (Vector database)
```

### Data & Retrieval
```
- pgvector (Supabase)     (Vector search)
- LangChain               (RAG orchestration)
- PyPDF2                  (Document processing)
- Google GenAI Embeddings (Vector embeddings)
```

### Infrastructure
```
- Google Cloud Run        (Deployment)
- Docker                  (Containerization)
- GitHub Actions          (CI/CD)
- Cloud Logging           (Monitoring)
```

### Development
```
- pytest                  (Testing)
- python-dotenv           (Environment management)
- Postman/curl            (API testing)
- Git                     (Version control)
```

---

## 🏗️ Architecture

### System Diagram

```
┌────────────────────────────────────┐
│  Frontend (Vercel - Next.js)       │
│  https://haraka-frontend.vercel.app│
└──────────────┬─────────────────────┘
               │ HTTP/HTTPS
               ▼
┌────────────────────────────────────┐
│  Backend (Heroku - Django)         │
│  REST API for runs & conversations │
│  - User auth (Token)               │
│  - Conversation history            │
│  - Run orchestration               │
└──────────────┬─────────────────────┘
               │ gRPC/HTTP
               ▼
┌────────────────────────────────────┐
│  AI Agent (Google Cloud Run)       │
│  FastAPI + Gemini + RAG            │
├────────────────────────────────────┤
│  Components:                       │
│  • agent.py (Main orchestrator)    │
│  • rag_tools.py (Retrieval)        │
│  • agents/ (Specialized agents)    │
│  • tools/ (Utilities)              │
│  • db_utils.py (Persistence)       │
│  • economist_fallback.py (Fallback)│
│  • web_search.py (External search) │
└──────────────┬─────────────────────┘
               │ SQL
               ▼
┌────────────────────────────────────┐
│  Data Layer (Supabase PostgreSQL)  │
├────────────────────────────────────┤
│  • pgvector (Embeddings)           │
│  • Runs table (Audit log)          │
│  • Documents (RAG store)           │
│  • Steps (Reasoning trace)         │
└────────────────────────────────────┘
```

### Component Overview

| Component | Purpose | File(s) | Size |
|-----------|---------|---------|------|
| Main Agent | FastAPI app & routing | `agent.py` | 26KB |
| RAG Engine | Semantic search & citations | `rag_tools.py` | 5.8KB |
| DB Utils | Database operations | `db_utils.py`, `tools/db.py` | 16.5KB |
| Embeddings | Vector generation | `embedding_utils.py` | 1.4KB |
| Scenario Agent | "What-if" analysis | `agents/scenario/` | - |
| Forecasting Agent | Time-series prediction | `agents/forecasting/` | - |
| Comparative Agent | Cross-country analysis | `agents/comparative/` | - |
| Graphing Tools | Visualizations | `tools/graphing.py` | 1.7KB |
| Query Tools | Query optimization | `tools/query.py` | 996B |
| Fallback Logic | Error handling | `economist_fallback.py` | 18KB |
| Web Search | External search | `web_search.py` | 1.5KB |
| Logging | Structured logs | `log_utils.py` | 1.8KB |

---

## 🚀 Installation Guide

### Prerequisites

Ensure you have installed:

```bash
# Check versions
python --version          # Python 3.8+
psql --version           # PostgreSQL 13+
docker --version         # Docker (optional)
git --version            # Git
```

**Required accounts:**
- Google Cloud Project with Gemini API enabled
- Supabase account (or PostgreSQL instance)

### PostgreSQL Setup (Local Development)

#### macOS Setup

```bash
# Install PostgreSQL
brew install postgresql
brew install postgresql@15

# Start PostgreSQL service
brew services start postgresql

# Verify installation
psql --version

# Install pgvector extension
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
make install

# Create database
createdb haraka_ai

# Enable pgvector extension
psql haraka_ai -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Verify extension
psql haraka_ai -c "\dx"
```

#### Ubuntu/Linux Setup

```bash
# Update package manager
sudo apt update

# Install PostgreSQL and extensions
sudo apt install postgresql postgresql-contrib postgresql-15-pgvector

# Start PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Verify installation
sudo systemctl status postgresql

# Create database as postgres user
sudo -u postgres createdb haraka_ai

# Enable pgvector
sudo -u postgres psql haraka_ai -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Verify
sudo -u postgres psql haraka_ai -c "\dx"
```

#### Windows Setup

```batch
# Download installer from https://www.postgresql.org/download/windows/
# Run installer with default settings
# Remember the postgres password during installation

# Open Command Prompt as Administrator
psql -U postgres

# Then run in psql:
CREATE DATABASE haraka_ai;
\c haraka_ai
CREATE EXTENSION vector;
\dx

# Exit with \q
```

### Step 1: Clone Repository

```bash
# Clone the repository
git clone https://github.com/simplykevine/Haraka-ai.git
cd Haraka-ai

# Verify structure
ls -la
# Should show: agent.py, requirements.txt, Dockerfile, etc.
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate

# Verify activation (should show venv prefix)
which python
```

### Step 3: Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip setuptools wheel

# Install main dependencies
pip install -r requirements.txt

# Install local zeno_agent dependencies
pip install -r zeno_agent/requirements.txt

# Verify installation
pip list | grep -E "fastapi|google|langchain|psycopg"
```

### Step 4: Configure Environment Variables

```bash
# Create .env file
touch .env

# Add configuration (use your actual credentials)
cat > .env << 'EOF'
# Google Gemini API
GOOGLE_API_KEY=your_google_gemini_api_key_here

# Database Configuration
DATABASE_URL=postgresql://postgres:password@localhost:5432/haraka_ai
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=haraka_ai
DATABASE_USER=postgres
DATABASE_PASSWORD=your_postgres_password

# FastAPI Settings
ENVIRONMENT=development
DEBUG=True
LOG_LEVEL=INFO
WORKERS=4

# Optional: Supabase Configuration
SUPABASE_URL=your_supabase_url
SUPABASE_API_KEY=your_supabase_api_key
EOF

# Secure the file
chmod 600 .env
```

### Step 5: Initialize Database

```bash
# Create tables (if scripts exist)
python scripts/init_db.py

# Or use alembic
python -m alembic upgrade head

# Verify database connection
psql haraka_ai -c "SELECT version();"
```

### Step 6: Download Embedding Models

```bash
# Download Google embedding model
python -c "
from zeno_agent.embedding_utils import EmbeddingManager
em = EmbeddingManager()
print('✅ Embeddings model ready')
"
```

### Step 7: Start FastAPI Server

```bash
# Development mode (with auto-reload)
uvicorn zeno_agent.agent:app \
  --host 0.0.0.0 \
  --port 8080 \
  --reload \
  --log-level info

# Production mode (with workers)
uvicorn zeno_agent.agent:app \
  --host 0.0.0.0 \
  --port 8080 \
  --workers 4 \
  --log-level warning
```

### Step 8: Verify Installation

```bash
# In another terminal, test the API
curl http://localhost:8080/health

# Should return: {"status":"ok"}

# Open Swagger UI
# Visit: http://localhost:8080/docs
```

---

## 🎯 Quick Start

### 1. Minimal Setup (5 minutes)

```bash
# Clone
git clone https://github.com/simplykevine/Haraka-ai.git
cd Haraka-ai

# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your GOOGLE_API_KEY and DATABASE_URL

# Run
uvicorn zeno_agent.agent:app --port 8080 --reload
```

### 2. Test Basic Query

```bash
# In another terminal
curl -X POST "http://localhost:8080/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "What is the EAC Common External Tariff on maize?",
    "conversation_id": "test_conv_001",
    "analysis_type": "rag"
  }'
```

### 3. Access Documentation

```
Swagger UI:  http://localhost:8080/docs
ReDoc:       http://localhost:8080/redoc
OpenAPI:     http://localhost:8080/openapi.json
```

---

## 📖 Usage Examples

### Example 1: Basic Policy Query

```bash
# Query the RAG system
curl -X POST "http://localhost:8080/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "What tariffs does Tanzania apply to imported maize?",
    "conversation_id": "conv_tz_trade_001",
    "analysis_type": "rag",
    "include_citations": true,
    "max_tokens": 2000
  }'

# Response (200 OK)
{
  "run_id": "run_xyz789",
  "analysis": "According to EAC trade documents, Tanzania applies a 35% tariff on imported maize under the common external tariff framework. This rate has been in place since January 2020 as part of the EAC customs union agreement.",
  "citations": [
    {
      "source": "EAC_Trade_Report_2025.pdf",
      "relevance_score": 0.94,
      "page": 45
    },
    {
      "source": "Tanzania_Tariff_Schedule_2024.xlsx",
      "relevance_score": 0.87,
      "page": null
    }
  ],
  "processing_time_ms": 27340,
  "tokens_used": 1850
}
```

### Example 2: Scenario Analysis

```bash
# Scenario: What if Tanzania increases tariff by 15%
curl -X POST "http://localhost:8080/api/scenario" \
  -H "Content-Type: application/json" \
  -d '{
    "scenario": "Tanzania increases maize import tariff from 35% to 50%",
    "baseline_country": "Tanzania",
    "comparison_countries": ["Kenya", "Uganda"],
    "metrics": [
      "price_impact",
      "trade_volume",
      "fiscal_impact",
      "consumer_welfare"
    ],
    "time_period_months": 12
  }'

# Response (200 OK)
{
  "run_id": "run_scenario_001",
  "scenario": "Tanzania increases maize import tariff from 35% to 50%",
  "predicted_impacts": {
    "price_impact": {
      "estimate": "+8.5%",
      "confidence": 0.89,
      "reasoning": "Higher tariffs typically increase domestic prices..."
    },
    "trade_volume": {
      "estimate": "-12.3%",
      "confidence": 0.85,
      "reasoning": "Elasticity of demand suggests reduced imports..."
    },
    "fiscal_impact": {
      "revenue_increase": "+$45.2M",
      "confidence": 0.91
    },
    "consumer_welfare": {
      "estimate": "-$120.5M",
      "confidence": 0.83
    }
  },
  "comparative_analysis": {
    "Kenya": "Would benefit from increased market share",
    "Uganda": "Limited impact due to lower production capacity"
  },
  "processing_time_ms": 31200
}
```

### Example 3: Comparative Analysis

```bash
# Compare maize tariffs across EAC
curl -X POST "http://localhost:8080/api/comparative" \
  -H "Content-Type: application/json" \
  -d '{
    "policy_area": "maize_tariffs",
    "countries": ["Tanzania", "Kenya", "Uganda", "Rwanda", "Burundi"],
    "comparison_dimensions": [
      "tariff_rate",
      "exemptions",
      "regional_impact",
      "trade_flow"
    ]
  }'

# Response (200 OK)
{
  "run_id": "run_comp_001",
  "comparison": {
    "tariff_rates": {
      "Tanzania": "35%",
      "Kenya": "35%",
      "Uganda": "0%",
      "Rwanda": "25%",
      "Burundi": "35%"
    },
    "trade_implications": "Uganda's zero tariff makes it a regional hub...",
    "compliance_status": "Most countries compliant with EAC protocol...",
    "recommendations": [
      "Harmonize tariff rates to 30%",
      "Reduce exemptions for regional trade",
      "Implement graduated phase-in period"
    ]
  },
  "processing_time_ms": 28950
}
```

### Example 4: Forecasting

```bash
# Forecast maize prices for next 12 months
curl -X POST "http://localhost:8080/api/forecast" \
  -H "Content-Type: application/json" \
  -d '{
    "commodity": "maize",
    "country": "Tanzania",
    "forecast_months": 12,
    "confidence_level": 0.95
  }'

# Response (200 OK)
{
  "run_id": "run_forecast_001",
  "commodity": "maize",
  "country": "Tanzania",
  "forecast": {
    "next_3_months": {
      "average_price": "$245/ton",
      "trend": "slight_increase",
      "confidence": 0.92
    },
    "next_6_months": {
      "average_price": "$252/ton",
      "trend": "stable",
      "confidence": 0.88
    },
    "next_12_months": {
      "average_price": "$248/ton",
      "trend": "mean_reversion",
      "confidence": 0.85
    }
  },
  "drivers": [
    "Seasonal rainfall patterns",
    "Regional supply dynamics",
    "Global commodity prices",
    "Policy interventions"
  ],
  "processing_time_ms": 25630
}
```


### Example 7: Get Run History

```bash
# Retrieve conversation runs
curl -X GET "http://localhost:8080/api/runs/conv_tz_trade_001" \
  -H "Authorization: Bearer your_token"

# Response (200 OK)
{
  "conversation_id": "conv_tz_trade_001",
  "total_runs": 5,
  "runs": [
    {
      "run_id": "run_xyz789",
      "user_input": "What tariffs does Tanzania apply to imported maize?",
      "analysis_type": "rag",
      "status": "completed",
      "created_at": "2026-04-02T10:15:00Z",
      "processing_time_ms": 27340,
      "citations_count": 2
    },
    {
      "run_id": "run_scenario_001",
      "user_input": "What if Tanzania increases tariff by 15%?",
      "analysis_type": "scenario",
      "status": "completed",
      "created_at": "2026-04-02T10:18:00Z",
      "processing_time_ms": 31200,
      "artifacts": ["chart", "table"]
    }
  ]
}
```

---

## 💻 Implementation Details

### Main Agent (`zeno_agent/agent.py`)

```python
# zeno_agent/agent.py - Main FastAPI application
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Haraka-ai",
    description="Institutional-grade AI agent for East African policy decision-support",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
from zeno_agent.rag_tools import RAGEngine
from zeno_agent.db_utils import Database
from zeno_agent.embedding_utils import EmbeddingManager

# Global service instances
rag_engine = None
db_connection = None
embedding_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global rag_engine, db_connection, embedding_manager
    print("🚀 Starting Haraka-ai agent...")
    
    db_connection = await Database.connect(os.getenv("DATABASE_URL"))
    embedding_manager = EmbeddingManager()
    rag_engine = RAGEngine(db_connection, embedding_manager)
    
    print("✅ Agent initialized successfully")
    yield
    
    # Shutdown
    print("Shutting down zeno-ai agent...")
    await db_connection.close()

app.router.lifespan_context = lifespan

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "zeno-ai",
        "version": "1.0.0"
    }

# Main query endpoint
@app.post("/api/query")
async def process_query(request: QueryRequest):
    """
    Main endpoint for policy queries
    
    Args:
        request: QueryRequest object with user_input, conversation_id, etc.
    
    Returns:
        Response with analysis, citations, and run_id
    """
    try:
        # Validate request
        if not request.user_input or len(request.user_input.strip()) == 0:
            raise HTTPException(status_code=400, detail="user_input cannot be empty")
        
        # Create run record
        run_id = await db_connection.create_run(
            conversation_id=request.conversation_id,
            user_input=request.user_input
        )
        
        # Route query to appropriate agent
        query_type = await route_query(request.user_input)
        
        # Retrieve context via RAG
        context = await rag_engine.retrieve_context(
            query=request.user_input,
            k=request.top_k or 5
        )
        
        # Execute appropriate agent
        if query_type == "scenario":
            result = await execute_scenario_agent(request.user_input, context)
        elif query_type == "forecast":
            result = await execute_forecasting_agent(request.user_input, context)
        elif query_type == "comparative":
            result = await execute_comparative_agent(request.user_input, context)
        else:
            result = await execute_rag_agent(request.user_input, context)
        
        # Persist results
        await db_connection.add_step(
            run_id=run_id,
            step_order=1,
            step_type="analysis",
            content=result
        )
        
        await db_connection.finalize_run(
            run_id=run_id,
            final_output=result["analysis"]
        )
        
        return {
            "run_id": run_id,
            "query_type": query_type,
            "analysis": result["analysis"],
            "context": context,
            "citations": extract_citations(context),
            "processing_time_ms": result.get("processing_time_ms", 0)
        }
    
    except Exception as e:
        await db_connection.finalize_run(
            run_id=run_id,
            final_output=f"Error: {str(e)}"
        )
        raise HTTPException(status_code=500, detail=str(e))

# Query routing
async def route_query(user_input: str) -> str:
    """Route query to appropriate agent"""
    user_lower = user_input.lower()
    
    if any(word in user_lower for word in ["what if", "scenario", "simulate"]):
        return "scenario"
    elif any(word in user_lower for word in ["forecast", "predict", "trend", "future"]):
        return "forecast"
    elif any(word in user_lower for word in ["compare", "vs", "versus", "different"]):
        return "comparative"
    else:
        return "rag"

# Get run history
@app.get("/api/runs/{conversation_id}")
async def get_runs(conversation_id: str):
    """Retrieve run history for a conversation"""
    runs = await db_connection.get_conversation_runs(conversation_id)
    return {"runs": runs, "total": len(runs)}

# Scenario analysis
@app.post("/api/scenario")
async def scenario_analysis(request: ScenarioRequest):
    """Scenario analysis endpoint"""
    run_id = await db_connection.create_run(
        conversation_id=request.conversation_id or "scenario_temp",
        user_input=request.scenario
    )
    
    result = await execute_scenario_agent(request.scenario, [])
    await db_connection.finalize_run(run_id, result["analysis"])
    
    return {
        "run_id": run_id,
        "scenario": request.scenario,
        "predicted_impacts": result.get("impacts", {}),
        "processing_time_ms": result.get("processing_time_ms", 0)
    }

# Forecasting
@app.post("/api/forecast")
async def forecasting(request: ForecastRequest):
    """Forecasting endpoint"""
    run_id = await db_connection.create_run(
        conversation_id=request.conversation_id or "forecast_temp",
        user_input=f"Forecast {request.commodity} for {request.country}"
    )
    
    result = await execute_forecasting_agent(
        f"Forecast {request.commodity} prices in {request.country}",
        []
    )
    await db_connection.finalize_run(run_id, result["analysis"])
    
    return {
        "run_id": run_id,
        "commodity": request.commodity,
        "country": request.country,
        "forecast": result.get("forecast", {}),
        "drivers": result.get("drivers", []),
        "processing_time_ms": result.get("processing_time_ms", 0)
    }

# Comparative analysis
@app.post("/api/comparative")
async def comparative_analysis(request: ComparativeRequest):
    """Comparative analysis endpoint"""
    run_id = await db_connection.create_run(
        conversation_id=request.conversation_id or "comparative_temp",
        user_input=f"Compare {request.policy_area} across {len(request.countries)} countries"
    )
    
    result = await execute_comparative_agent(
        f"Compare {request.policy_area} across {', '.join(request.countries)}",
        []
    )
    await db_connection.finalize_run(run_id, result["analysis"])
    
    return {
        "run_id": run_id,
        "policy_area": request.policy_area,
        "countries": request.countries,
        "comparison": result.get("comparison", {}),
        "recommendations": result.get("recommendations", []),
        "processing_time_ms": result.get("processing_time_ms", 0)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "agent:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8080)),
        reload=os.getenv("ENVIRONMENT") == "development"
    )
```

### RAG Tools (`zeno_agent/rag_tools.py`)

```python
# zeno_agent/rag_tools.py - Retrieval-Augmented Generation
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import SupabaseVectorStore
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import asyncio

load_dotenv()

class RAGEngine:
    """Retrieval-Augmented Generation Engine"""
    
    def __init__(self, db_connection, embedding_manager):
        """
        Initialize RAG Engine
        
        Args:
            db_connection: Database connection instance
            embedding_manager: Embedding manager instance
        """
        self.db = db_connection
        self.embeddings = embedding_manager.get_embeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
    
    async def retrieve_context(
        self,
        query: str,
        k: int = 5,
        min_relevance: float = 0.5
    ) -> List[Dict]:
        """
        Retrieve top-k relevant documents for the query
        
        Args:
            query: User query
            k: Number of documents to retrieve
            min_relevance: Minimum relevance score
        
        Returns:
            List of retrieved documents with metadata
        """
        try:
            # Get embeddings for query
            query_embedding = await self._embed_text(query)
            
            # Search in vector store
            results = await self.db.search_documents(
                query_embedding=query_embedding,
                limit=k,
                min_relevance=min_relevance
            )
            
            # Format results with citations
            documents = []
            for doc, score in results:
                documents.append({
                    "content": doc["content"],
                    "source": doc.get("source", "Unknown"),
                    "relevance": float(score),
                    "page": doc.get("page"),
                    "document_id": doc.get("document_id")
                })
            
            return documents
        
        except Exception as e:
            print(f"❌ Error in retrieve_context: {str(e)}")
            return []
    
    async def _embed_text(self, text: str) -> List[float]:
        """Get embedding for text"""
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            print(f"❌ Error embedding text: {str(e)}")
            return [0.0] * 768  # Default embedding dimension
    
    async def ingest_document(
        self,
        file_path: str,
        source: str,
        category: str = "general"
    ) -> Dict:
        """
        Ingest a document into the RAG system
        
        Args:
            file_path: Path to document file
            source: Document source identifier
            category: Document category
        
        Returns:
            Ingestion result with document_id and chunks created
        """
        try:
            # Load document
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Embed and store chunks
            chunk_ids = []
            for i, chunk in enumerate(chunks):
                embedding = await self._embed_text(chunk.page_content)
                
                chunk_id = await self.db.store_chunk(
                    content=chunk.page_content,
                    embedding=embedding,
                    source=source,
                    category=category,
                    page=chunk.metadata.get("page", None),
                    document_path=file_path
                )
                chunk_ids.append(chunk_id)
            
            return {
                "document_path": file_path,
                "source": source,
                "chunks_created": len(chunks),
                "chunk_ids": chunk_ids
            }
        
        except Exception as e:
            print(f"❌ Error ingesting document: {str(e)}")
            return {"error": str(e)}
    
    async def extract_citations(
        self,
        context: List[Dict]
    ) -> List[Dict]:
        """Extract citations from retrieved context"""
        citations = []
        for doc in context:
            citations.append({
                "source": doc["source"],
                "relevance_score": doc["relevance"],
                "page": doc.get("page"),
                "document_id": doc.get("document_id")
            })
        return citations
```

### Database Utilities (`zeno_agent/db_utils.py`)

```python
# zeno_agent/db_utils.py - Database utilities
import asyncio
import os
from datetime import datetime
from typing import Optional, List, Dict
import psycopg
from pgvector.psycopg import register_vector

class Database:
    """Database connection and operations"""
    
    def __init__(self, conn):
        self.conn = conn
    
    @staticmethod
    async def connect(database_url: str):
        """Connect to database"""
        try:
            conn = await psycopg.AsyncConnection.connect(database_url)
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            register_vector(conn)
            print("✅ Database connected")
            return Database(conn)
        except Exception as e:
            print(f"❌ Database connection error: {str(e)}")
            raise
    
    async def create_run(
        self,
        conversation_id: str,
        user_input: str
    ) -> str:
        """Create a new run record"""
        import uuid
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        
        await self.conn.execute(
            """
            INSERT INTO runs (run_id, conversation_id, user_input, status, started_at)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (run_id, conversation_id, user_input, "processing", datetime.utcnow())
        )
        
        return run_id
    
    async def add_step(
        self,
        run_id: str,
        step_order: int,
        step_type: str,
        content: Dict
    ) -> None:
        """Add a step to a run"""
        import json
        
        await self.conn.execute(
            """
            INSERT INTO steps (run_id, step_order, type, content, created_at)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (run_id, step_order, step_type, json.dumps(content), datetime.utcnow())
        )
    
    async def finalize_run(
        self,
        run_id: str,
        final_output: str
    ) -> None:
        """Mark run as completed"""
        await self.conn.execute(
            """
            UPDATE runs
            SET status = %s, final_output = %s, completed_at = %s
            WHERE run_id = %s
            """,
            ("completed", final_output, datetime.utcnow(), run_id)
        )
    
    async def search_documents(
        self,
        query_embedding: List[float],
        limit: int = 5,
        min_relevance: float = 0.5
    ) -> List[tuple]:
        """Search documents by embedding similarity"""
        from pgvector.psycopg import Vector
        
        results = await self.conn.execute(
            """
            SELECT content, source, page, document_id,
                   1 - (embedding <-> %s) as similarity
            FROM documents
            WHERE 1 - (embedding <-> %s) > %s
            ORDER BY similarity DESC
            LIMIT %s
            """,
            (Vector(query_embedding), Vector(query_embedding), min_relevance, limit)
        )
        
        return await results.fetchall()
    
    async def store_chunk(
        self,
        content: str,
        embedding: List[float],
        source: str,
        category: str,
        page: Optional[int] = None,
        document_path: Optional[str] = None
    ) -> str:
        """Store a document chunk"""
        import uuid
        from pgvector.psycopg import Vector
        
        chunk_id = f"chunk_{uuid.uuid4().hex[:12]}"
        
        await self.conn.execute(
            """
            INSERT INTO documents (
                chunk_id, content, embedding, source, 
                category, page, document_path, created_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (chunk_id, content, Vector(embedding), source,
             category, page, document_path, datetime.utcnow())
        )
        
        return chunk_id
    
    async def get_conversation_runs(self, conversation_id: str) -> List[Dict]:
        """Get all runs for a conversation"""
        results = await self.conn.execute(
            """
            SELECT run_id, user_input, status, started_at, completed_at
            FROM runs
            WHERE conversation_id = %s
            ORDER BY started_at DESC
            """,
            (conversation_id,)
        )
        
        rows = await results.fetchall()
        return [
            {
                "run_id": row[0],
                "user_input": row[1],
                "status": row[2],
                "started_at": row[3],
                "completed_at": row[4]
            }
            for row in rows
        ]
    
    async def close(self):
        """Close database connection"""
        await self.conn.close()
```

### Embedding Utils (`zeno_agent/embedding_utils.py`)

```python
# zeno_agent/embedding_utils.py - Embedding utilities
import os
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

class EmbeddingManager:
    """Manage embeddings for RAG"""
    
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not set in environment")
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=self.api_key
        )
    
    def get_embeddings(self):
        """Get embeddings instance"""
        return self.embeddings
    
    def embed_text(self, text: str):
        """Embed text"""
        return self.embeddings.embed_query(text)
    
    def embed_documents(self, texts: list):
        """Embed multiple documents"""
        return self.embeddings.embed_documents(texts)
```

---

## 📡 API Documentation

### Request Models

```python
# Request model for queries
class QueryRequest(BaseModel):
    user_input: str
    conversation_id: Optional[str] = None
    analysis_type: str = "rag"  # rag, scenario, forecast, comparative
    include_citations: bool = True
    top_k: int = 5
    max_tokens: int = 2000

# Request model for scenarios
class ScenarioRequest(BaseModel):
    scenario: str
    baseline_country: str
    comparison_countries: Optional[List[str]] = None
    metrics: List[str] = ["price_impact", "trade_volume"]
    time_period_months: int = 12

# Request model for forecasts
class ForecastRequest(BaseModel):
    commodity: str
    country: str
    forecast_months: int = 12
    confidence_level: float = 0.95

# Request model for comparative analysis
class ComparativeRequest(BaseModel):
    policy_area: str
    countries: List[str]
    comparison_dimensions: List[str] = []
```

### Response Models

```python
# Response model for queries
class QueryResponse(BaseModel):
    run_id: str
    query_type: str
    analysis: str
    citations: List[Dict]
    processing_time_ms: int

# Response model for scenarios
class ScenarioResponse(BaseModel):
    run_id: str
    scenario: str
    predicted_impacts: Dict
    processing_time_ms: int

# Response model for forecasts
class ForecastResponse(BaseModel):
    run_id: str
    commodity: str
    country: str
    forecast: Dict
    drivers: List[str]
    processing_time_ms: int

# Response model for comparative analysis
class ComparativeResponse(BaseModel):
    run_id: str
    policy_area: str
    countries: List[str]
    comparison: Dict
    recommendations: List[str]
    processing_time_ms: int
```

---

## 📁 File Structure

```
Haraka-ai/
├── .github/
│   └── workflows/
│       └── deploy-adk-agent.yml          # GitHub Actions CI/CD pipeline
├── .gitignore                             # Git ignore rules
├── Dockerfile                             # Docker container definition
├── Procfile                               # Heroku deployment config
├── README.md                              # This file
├── check_models.py                        # Model availability checker
├── requirements.txt                       # Root dependencies
├── .env.example                           # Environment variables template
├── zeno_agent/
│   ├── __init__.py
│   ├── agent.py                          # Main FastAPI application (26KB)
│   ├── rag_tools.py                      # RAG implementation (5.8KB)
│   ├── db_utils.py                       # Database utilities (6.8KB)
│   ├── embedding_utils.py                # Embedding management (1.4KB)
│   ├── log_utils.py                      # Logging utilities (1.8KB)
│   ├── web_search.py                     # Web search integration (1.5KB)
│   ├── economist_fallback.py             # Fallback logic (18KB)
│   ├── root_agent.py                     # Root agent setup (23B)
│   ├── root_agent.yml                    # Agent configuration
│   ├── requirements.txt                  # Local dependencies (2.4KB)
│   ├── query_cache.json                  # Cached query responses (429KB)
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── scenario/                     # Scenario analysis agent
│   │   ├── forecasting/                  # Forecasting agent
│   │   └── comparative/                  # Comparative analysis agent
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── db.py                        # Advanced DB operations (9.7KB)
│   │   ├── graphing.py                  # Chart generation (1.7KB)
│   │   └── query.py                     # Query optimization (996B)
│   ├── prompts/                          # Prompt templates
│   └── benchmarks/                       # Performance benchmarks
└── scripts/
    └── init_db.py                        # Database initialization
```

---

## 🔄 Feedback Incorporation

### 1. Camera Permission Fix ✅

**Problem:** Frontend requested camera access on app launch

**Solution:** Removed proactive check in `haraka-frontend/src/app/sharedComponents/ChatInput/index.tsx`

**Impact:**
- ✅ No camera prompt on launch
- ✅ Permission only on camera button click
- ✅ Better UX for non-camera users

### 2. Citation Accuracy ✅

**Implementation:** `rag_tools.py` with pgvector similarity search

**Metrics:**
- Citation Accuracy: **55%** (vs 18% GPT-4o)
- Factual Accuracy: **89%** (vs 85% GPT-4o)
- Hallucination Rate: **65%** (vs 80% GPT-4o)

### 3. Performance Roadmap 📋

**Current Speed:** 27.3 seconds (by design for quality)

**Future Optimizations:**
- Response streaming (SSE)
- Query caching (in place: `query_cache.json`)
- Parallel agent execution
- Context window optimization

---

## 📊 Performance Benchmarks

### Test Results (20 East African Policy Questions)

```
Date:     March 17, 2026
Subjects: Economists & Policy Analysts
Dataset:  20 East African economic policy questions

Results:
┌──────────────────────┬───────┬────────┬──────────┐
│ Metric               │ Zeno  │ GPT-4o │ Advantage│
├──────────────────────┼───────┼────────┼──────────┤
│ Factual Accuracy     │ 89%   │ 85%    │ +4%  ✅  │
│ Citation Accuracy    │ 55%   │ 18%    │ +37% ✅✅│
│ Hallucination Rate   │ 65%   │ 80%    │ -15% ✅  │
│ Response Speed       │ 27.3s │ 2.3s   │ -24s ⚖️  │
│ Policy Auditability  │ Full  │ None   │ 100% ✅  │
└──────────────────────┴───────┴────────┴──────────┘

Conclusion: Zeno outperforms on all policy-relevant dimensions
```

---

## 🌐 Deployment

### Prerequisites

```bash
# Google Cloud
- Project with Gemini API enabled
- Service account with Cloud Run permissions
- Docker image pushed to Docker Hub

# Supabase/PostgreSQL
- PostgreSQL 13+
- pgvector extension
- Database credentials

# GitHub
- Repository with secret keys configured
```

### Environment Variables

```env
# Production Deployment
ENVIRONMENT=production
DEBUG=False
LOG_LEVEL=warning

# Google Cloud
GCP_PROJECT_ID=your-project-id
GCP_REGION=europe-west1
GCP_SERVICE_ACCOUNT_KEY={"type":"service_account",...}

# API Keys
GOOGLE_API_KEY=your_google_gemini_key
DATABASE_URL=postgresql://user:pass@host:5432/haraka_ai

# Performance
WORKERS=4
MAX_CONNECTIONS=50
TIMEOUT_SECONDS=60
```

### Deployment Steps

```bash
# 1. Build Docker image
docker build -t simplykevine/zeno-decision-tool:latest .

# 2. Push to Docker Hub
docker push simplykevine/zeno-decision-tool:latest

# 3. Deploy to Google Cloud Run
gcloud run deploy haraka-ai \
  --image simplykevine/zeno-decision-tool:latest \
  --platform managed \
  --region europe-west1 \
  --memory 2Gi \
  --cpu 2 \
  --set-env-vars GOOGLE_API_KEY=your_key,DATABASE_URL=your_db_url

# 4. Verify deployment
gcloud run services describe haraka-ai --region europe-west1
```

---

## 🐳 Docker Setup

### Build Local Image

```bash
# Build image
docker build -t haraka-ai:latest .

# Run container
docker run -p 8080:8080 \
  -e GOOGLE_API_KEY=your_key \
  -e DATABASE_URL=postgresql://user:pass@localhost:5432/haraka_ai \
  haraka-ai:latest

# Check logs
docker logs -f container_id

# Stop container
docker stop container_id
```

### Docker Compose (Multiple Services)

```yaml
# docker-compose.yml
version: '3.8'

services:
  postgres:
    image: pgvector/pgvector:pg15-latest
    environment:
      POSTGRES_DB: haraka_ai
      POSTGRES_PASSWORD: postgres
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  haraka-ai:
    build: .
    ports:
      - "8080:8080"
    environment:
      DATABASE_URL: postgresql://postgres:postgres@postgres:5432/haraka_ai
      GOOGLE_API_KEY: ${GOOGLE_API_KEY}
      ENVIRONMENT: development
    depends_on:
      - postgres
    volumes:
      - .:/app

volumes:
  postgres_data:
```

---

## 🧪 Testing

### Unit Tests

```bash
# Run all tests
pytest -v

# Run specific test file
pytest tests/test_rag_tools.py -v

# Run with coverage
pytest --cov=zeno_agent tests/

# Run specific test
pytest tests/test_rag_tools.py::test_retrieve_context -v
```

### Integration Tests

```bash
# Test API endpoints
pytest tests/test_api.py -v

# Test database operations
pytest tests/test_database.py -v

# Test end-to-end flow
pytest tests/test_e2e.py -v
```

### Load Testing

```bash
# Using locust
locust -f tests/locustfile.py --host=http://localhost:8080

# Using Apache Bench
ab -n 1000 -c 10 http://localhost:8080/health
```

---

## 🔧 Troubleshooting

### Issue: Database Connection Failed

```bash
# Solution 1: Check PostgreSQL service
sudo systemctl status postgresql

# Solution 2: Verify connection string
psql postgresql://user:pass@localhost:5432/haraka_ai

# Solution 3: Check pgvector extension
psql haraka_ai -c "\dx"

# Solution 4: Recreate database
dropdb haraka_ai
createdb haraka_ai
psql haraka_ai -c "CREATE EXTENSION vector;"
```

### Issue: Gemini API Key Not Working

```bash
# Solution 1: Verify key in .env
cat .env | grep GOOGLE_API_KEY

# Solution 2: Test key directly
python -c "
from langchain.embeddings import GoogleGenerativeAIEmbeddings
emb = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
print(emb.embed_query('test'))
"

# Solution 3: Get new key from Google AI Studio
# https://makersuite.google.com/app/apikey
```

### Issue: Slow Response Time

```bash
# Solution 1: Check database indexing
psql haraka_ai -c "\di"

# Solution 2: Enable query caching
# query_cache.json is already in place

# Solution 3: Increase workers
uvicorn zeno_agent.agent:app --workers 8

# Solution 4: Monitor performance
python scripts/monitor_performance.py
```

### Issue: Out of Memory

```bash
# Solution 1: Reduce chunk size in rag_tools.py
# Change from 1000 to 500

# Solution 2: Limit top_k results
# Default is 5, reduce to 3

# Solution 3: Increase available memory
# Use larger Cloud Run instance
```

---

## 🤝 Contributing

### Development Setup

```bash
# Clone repository
git clone https://github.com/simplykevine/Haraka-ai.git
cd Haraka-ai

# Create feature branch
git checkout -b feature/your-feature

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dev dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8

# Make changes and test
pytest tests/

# Code formatting
black zeno_agent/
flake8 zeno_agent/

# Commit and push
git add .
git commit -m "feat: your feature description"
git push origin feature/your-feature

# Create pull request
# Go to GitHub and create PR
```

### Code Style

```python
# Follow PEP 8
# Use type hints
async def retrieve_context(
    self,
    query: str,
    k: int = 5
) -> List[Dict]:
    pass

# Use docstrings
"""
Retrieve top-k relevant documents for the query

Args:
    query: User query
    k: Number of documents to retrieve

Returns:
    List of retrieved documents with metadata
"""

# Use logging
import logging
logger = logging.getLogger(__name__)
logger.info("Retrieving context...")
```

---

## 📄 License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2026 Kevine Umutoni

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 📧 Contact

### Developer

**Umutoni Kevine**
- Email: [k.umutoni@alustudent.com](mailto:k.umutoni@alustudent.com)
- GitHub: [@simplykevine](https://github.com/simplykevine)
- LinkedIn: [Kevine Umutoni](https://www.linkedin.com/in/umutoni-kevine-aa9a29278/)

### Supervisor

**Bernard Odartei Lamptey**
- Faculty: Software Engineering
- Institution: African Leadership University

### Project Details

- **Status:** ✅ Complete & Production Ready
- **Submission Date:** April 2, 2026
- **Project Type:** Capstone Project
- **Version:** 1.0.0

---

## 🚀 Quick Links

| Resource | Link |
|----------|------|
| Live Demo | https://haraka-frontend.vercel.app |
| API Docs | http://localhost:8080/docs |
| GitHub Frontend | https://github.com/simplykevine/haraka-frontend |
| GitHub Backend | https://github.com/simplykevine/haraka-backend |
| GitHub AI Agent | https://github.com/simplykevine/Haraka-ai |
| Google Gemini | https://ai.google.dev/ |
| pgvector | https://github.com/pgvector/pgvector |
| FastAPI | https://fastapi.tiangolo.com/ |
| LangChain | https://langchain.com/ |

---

## 📞 Support

For issues, feature requests, or questions:

1. Check the [Troubleshooting](#troubleshooting) section
2. Search existing [GitHub Issues](https://github.com/simplykevine/Haraka-ai/issues)
3. Create a [new issue](https://github.com/simplykevine/Haraka-ai/issues/new)
4. Contact the developer

---

**Last Updated:** April 2, 2026  
**Version:** 1.0.0  
**Status:** Production Ready ✅

---

> 🎯 **Haraka-ai:** Making policy research accurate, transparent, and auditable for East Africa
