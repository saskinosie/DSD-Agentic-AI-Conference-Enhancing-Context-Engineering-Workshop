# Enhancing Context Engineering with Agentic Integration into Vector Database Queries

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![CockroachDB](https://img.shields.io/badge/cockroachdb-25.2-blue.svg)

> Build a production-grade multi-agent RAG system that decomposes complex queries, applies dynamic filtering, and orchestrates intent-based conversations over a vector database.

## Overview

Many agentic RAG systems fail in production for the same reason: they treat every user query as a single retrieval problem. Real users ask complex, multi-part questions that span multiple semantic areas of a database and when their needs shift mid-conversation, monolithic agents fall apart entirely.

This workshop builds from the ground up. Starting with vector database fundamentals and ending with a fully functioning intent-based multi-agent orchestration system, you will leave with working code you can extend, a clear mental model of conversational agent design, and the evaluation tools to know when your system is production-ready.

## What You'll Learn

### [Part 1: Vector Database Fundamentals](notebooks/1-vector-db-fundamentals.ipynb)
**Foundation** (0:00 – 0:20): How vector databases store and retrieve contextual information
- Spin up a local CockroachDB instance using Docker
- Load an e-commerce dataset and embed it with OpenAI
- Perform semantic and filtered searches
- Apply payload filters for precise results
- **Key insight**: Why retrieval quality is the foundation everything else is built on

### [Part 2: Query Decomposition and Dynamic Filtering](notebooks/2-query-decomposition.ipynb)
**Hands-on build** (0:20 – 1:00): Agents that make complex queries surgical
- Build agents with Pydantic AI that generate dynamic filters from natural language
- Decompose complex multi-part queries into discrete targeted retrieval calls
- Combine query optimization with smart filtering for production-grade results
- **Key insight**: AI agents aren't magic, they are structured prompts + intelligent parsing + API orchestration

### [Part 3: Intent-Based Multi-Agent Orchestration](notebooks/3-intent-orchestration.ipynb)
**Live demo + walkthrough** (1:00 – 1:45): Where single-turn RAG becomes conversational AI
- Intent classifier that identifies what the user is actually trying to accomplish
- Orchestrator that routes to intent-specific subagents
- Slot-filling layer that collects required information through natural clarifying questions
- State management that handles intent switches mid-conversation with clean resets
- **Key insight**: How intent-based design differs fundamentally from rank-based routing

### [Part 4: Evaluation and Production Patterns](notebooks/4-evaluation.ipynb)
**Wrap-up** (1:45 – 2:00): Know when your system is actually working
- Retrieval evaluation with MRR, NDCG, and precision metrics
- Production failure modes: intent bleed, slot hallucination, low-confidence deadlocks
- **Key insight**: If you can't measure it, you can't improve it

## Quick Start

### Prerequisites
- Python 3.10+
- Docker Desktop ([download here](https://docs.docker.com/get-started/get-docker/))
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))
- A code editor with Jupyter notebook support — [VS Code](https://code.visualstudio.com/) or [Cursor](https://www.cursor.com/) recommended

> **Important**: The full setup downloads ~720 MB of data (Docker image, Python packages, and dataset). **Complete all setup at home before the workshop.** 

### Setup

**1. Clone the repository**
```bash
git clone https://github.com/saskinosie/Enhancing-Context-Engineering-Workshop.git
cd Enhancing-Context-Engineering-Workshop
```

**2. Pull the CockroachDB Docker image (~400 MB)**
```bash
docker pull cockroachdb/cockroach:v25.2.4
```

**3. Create virtual environment and install Python dependencies (~400 MB)**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**4. Pre-download the HuggingFace dataset (~250 MB)**
```bash
python -c "from datasets import load_dataset; load_dataset('Qdrant/hm_ecommerce_products', split='train'); print('Dataset downloaded and cached.')"
```

**5. Configure credentials**
```bash
cp .env.example .env
# Edit .env with your OpenAI API key
```

**6. Start CockroachDB**
```bash
docker compose up -d
```

CockroachDB will be available at `postgresql://root@localhost:26257` (SQL). You can also view the admin UI at `http://localhost:8080`.

**7. Verify installation**

Open the project folder in VS Code or Cursor, then open `notebooks/1-vector-db-fundamentals.ipynb` and run the first few cells to confirm everything is working.

> **Alternatively**, you can run `python setup.py` after step 1 to automate steps 2–4 and run a verification check.

## Architecture

```
User Utterance
 ↓
[Intent Classifier]
 ↓
Intent + Confidence Score
 ↓
[Orchestrator] — maintains session state, slot context
 ↓
[Intent-Specific Subagent]
 ↓
[Slot Filler] — collects required entities
 ↓
[Query Decomposer] — breaks complex queries into subqueries
 ↓
[Dynamic Filter Generator] — generates payload filters per subquery
 ↓
[CockroachDB] — vector search retrieval per subquery
 ↓
[Result Unifier] — merges retrieved context
 ↓
[Response Generator] — final LLM call
 ↓
Response → User
```

## Project Structure

```
├── notebooks/
│   ├── 1-vector-db-fundamentals.ipynb    # Part 1: Vector search basics
│   ├── 2-query-decomposition.ipynb       # Part 2: Dynamic filtering agents
│   ├── 3-intent-orchestration.ipynb      # Part 3: Multi-agent orchestration
│   ├── 4-evaluation.ipynb               # Part 4: Retrieval evaluation
├── agents/
│   ├── intent_classifier.py              # Classifies user intent
│   ├── orchestrator.py                   # Routes to subagents, manages state
│   ├── slot_filler.py                    # Collects required entities
│   └── subagents/
│       ├── billing_agent.py              # Billing-specific retrieval
│       ├── product_agent.py              # Product-specific retrieval
│       └── support_agent.py              # Support-specific retrieval
├── data/                                 # Workshop dataset
├── img/                                  # Setup screenshots
├── setup.py                              # Pre-download script (run before workshop)
├── docker-compose.yml                    # Local CockroachDB instance
├── requirements.txt                      # Python dependencies
├── .env.example                          # Template for API keys
├── pyproject.toml                        # Ruff linter/formatter config
└── README.md                             # You are here!
```

## Key Technologies

- **[CockroachDB](https://www.cockroachlabs.com/)** - Distributed SQL database with native vector search (local via Docker)
- **[Pydantic AI](https://ai.pydantic.dev/)** - Framework for building production-grade AI agents
- **[OpenAI GPT-4.1-mini](https://platform.openai.com/)** - Language model for embeddings, query optimization, and agent reasoning
- **[HuggingFace Datasets](https://huggingface.co/datasets/Qdrant/hm_ecommerce_products)** - H&M e-commerce product catalog

## Key Takeaways

- Why retrieval quality is the foundation everything else is built on
- How intent-based design differs fundamentally from rank-based routing
- How to decompose complex queries without losing semantic coherence
- How to manage conversational state across multi-turn interactions
- How to measure whether your system is actually working in production

## Contributing

Found a bug? Have an improvement? Contributions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-addition`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-addition`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## About the Presenter

Scott Askinosie is a Developer Advocate at Contextual AI and former Lead Technical Trainer at Weaviate, where he delivered enterprise RAG workshops to organizations including NATO and FactSet. He holds a PhD in Quantitative Biology and has been building production RAG and agentic systems since before modern frameworks existed. He is a founding member of LangChain Austin and a frequent speaker on context engineering, vector databases, and multi-agent architecture.

## Resources & Further Reading

- [CockroachDB Vector Search](https://www.cockroachlabs.com/docs/stable/vector-indexes)
- [Pydantic AI Documentation](https://ai.pydantic.dev/)

---