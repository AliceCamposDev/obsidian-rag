# Product Requirements Document
<span style="font-size:2em">AGRAG </span> (Stands for Alice's Graph Retrieval-Augmented Generation, it must be thousands "grag"s out there)
## 1- Vision Statement
To revolutionize personal knowledge management by creating an intelligent, safe, and context-aware AI assistant that transforms how users interact with their Obsidian vaults and other graph based knowledge base.
### Problem Statement
- Searching through big Obsidian Vaults manually can be exhaustive
- Talking to AI assistants is frustrating when they start to forget everything 
- There are not many software options to help building a vault
- When using online tools that read your personal notes, your privacy and sensitive data is in the hands of third-party services that might not be trustfull

### Solution Statement
AGRAG can solve all of this problems
- Is a open source code, so you can check by yourself if it is trustfull
- Is a graph context based ai assistant, you don't have to search manually for what you want, just ask it!
- Can operate fully locally or with optional external integrations
- Highly configurable and versatile to apply at lots of different contexts
- Maintains persistent context through auto-incrementing knowledge bases

##  2. Core Features & Capabilities
### 2.1 Intelligent Search & Retrieval
 - Semantic Search: Natural language queries with contextual understanding
 - Graph Traversal: Navigate relationships between notes, tags, and concepts
 - Temporal Awareness: Understand chronological context and evolution of ideas
 - Cross-Reference Detection: Automatic identification of related content
 - LLM Orchestration (Query expansion, Guard rails, Other)
### 2.2 Context-Aware AI Assistant
- Maintain conversation history and learned preferences using graphs
- Ground answers in the user's specific knowledge base
- Recommend connections, tags, and content improvements
### 2.3 Privacy & Security Framework
- Default operation without external dependencies (safe mode)
- Precise permission management for external services
- Optional local encryption for sensitive vaults
- Forced encryption when using any external dependence
- Auditable logs with transparent track of all interactions and modifications made by GRAG
