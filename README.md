🧠 Agentic Multimodal RAG System for Fashion Recommendations
A retrieval-first, agent-driven Multimodal RAG system that delivers grounded fashion recommendations using images, text, and metadata — without LLM hallucinations.
🚀 Overview
This project implements an Agentic Multimodal Retrieval-Augmented Generation (RAG) system designed to solve a common problem in AI recommendation systems:
LLMs hallucinate when they are allowed to make decisions.
To fix this, the system strictly separates retrieval, ranking, and reasoning:
Retrieval decides what is relevant
LLMs only explain why
The system accepts images or outfit photos, retrieves visually and semantically similar footwear using multimodal embeddings, and produces confidence-aware, explainable recommendations.
🔍 What This System Can Do
✅ Supported Flows
Find Similar Shoes from an uploaded shoe image
Match Shoes with an Outfit using an outfit image
AI Stylist Chat grounded in retrieved results (text-only, no hallucination)
✅ Core Capabilities
Multimodal image + text retrieval
Confidence-aware ranking (no “no match found” failures)
Agent-orchestrated control flow
Grounded reasoning with zero product invention
Always returns top-N recommendations
🧠 System Architecture (High Level)
Copy code

User Input (Image / Text)
        ↓
CLIP Multimodal Embeddings
        ↓
FAISS Vector Search
        ↓
Metadata + Text Retrieval
        ↓
Weighted Score Fusion
        ↓
LangGraph Agent Orchestration
        ↓
LLM (Explanation Only)
        ↓
Streamlit UI
🤖 Agentic Design (LangGraph)
The project uses a single compiled agent built with LangGraph
Each node performs a deterministic task
The graph as a whole behaves like an agent
Key Agent Behaviors
Input routing (image / text / outfit)
Retrieval orchestration
Retry paths for low-confidence cases
Controlled reasoning phase
🔑 Important Insight:
Not every node is an agent.
The graph is the agent.
🛠 Tech Stack
Core Technologies
Python
FastAPI – backend service layer
Streamlit – interactive, stateful frontend
Multimodal Retrieval
CLIP – joint image–text embeddings
FAISS – fast vector similarity search
Agentic AI
LangGraph – agent orchestration and routing
LLM
OpenAI GPT-4o-mini
Used only for explanation
Never used for ranking or decision making
Paradigms Used
Multimodal RAG
Agentic workflows
Confidence-aware ranking
Retrieval-first AI systems
Computer Vision + NLP fusion
📊 Ranking & Confidence Strategy
Relevance is determined before the LLM is called.
Scoring Signals
Image similarity score
Text similarity score
Metadata relevance score
Fusion Strategy
Weighted aggregation of multiple retrieval signals
Stable ranking guarantees top-N outputs
The LLM cannot override retrieval results.
🖥 Frontend Design
Built with Streamlit
Supports image uploads and chat-style interaction
Final UI emphasizes text-based, grounded explanations
Images were intentionally deprioritized to ensure:
Stability
Explainability
Demo reliability
🎯 Design Principles
❌ No LLM hallucination
❌ No product invention
❌ No opaque scoring
✅ Deterministic retrieval
✅ Explainable recommendations
✅ Production-ready architecture
📚 Key Learnings
LLMs should explain, not decide
Retrieval-first systems are more reliable than prompt-heavy ones
Agentic workflows shine when responsibilities are strictly separated
Multimodal RAG is a system design problem, not just a model choice
🔮 Future Improvements
Brand-aware filtering
Price and availability constraints
Cross-category outfit recommendations
Cloud deployment (Docker + cloud FAISS)
Richer metadata enrichment
🧑‍💻 Author
Vasu Arora
BTech Computer Science Engineering
Interested in:
Agentic AI systems
Multimodal RAG
Production-grade AI architectures
Reliable, explainable AI
📌 Final Thought
Grounded AI systems outperform flashy demos.
Reliability, explainability, and debuggability matter more than novelty.
If you want, next I can:
Add setup & run instructions
Write a demo section
Create a system design interview explanation
Add architecture diagrams
Just say the word 👌
