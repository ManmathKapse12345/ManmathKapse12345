# Detailed Project Descriptions for Resume

## 1. RAG Q&A Bot -- Retrieval-Augmented Generation System
**Repository:** https://github.com/ManmathKapse12345/RAG-Q-A-Bot-Project  
**Language:** Python (100%)  
**Status:** Complete & Production-Ready

### Project Overview
A sophisticated retrieval-augmented Q&A system that enables users to upload PDF documents, ask natural language questions, and receive answers grounded exclusively in the document content. The system prevents hallucinations through context-aware LLM prompting, ensuring all answers are factually tied to retrieved content with source citations.

### Technical Architecture

**1. PDF Ingestion & Chunking Pipeline**
- Extracts text per page using PyPDF, handling OCR artifacts
- Implements intelligent overlapping chunks (200 tokens, 40-token overlap) using the embedding model's native tokenizer
- Respects model max-sequence-length constraints to prevent truncation
- Preserves original formatting (casing, punctuation) through offset mapping

**2. Embedding & Vector Operations (PyTorch-Powered)**
- Leverages Sentence-Transformers (all-MiniLM-L6-v2, 384-dimensional embeddings)
- Built on PyTorch backend for efficient tensor operations
- Batch encoding of 32+ chunks with GPU acceleration when available
- Normalized embeddings (unit-length vectors) for direct cosine-similarity matching via inner product

**3. Vector Storage & Retrieval (FAISS)**
- IndexFlatIP implementation for O(1) nearest-neighbor search
- Stores normalized embeddings alongside metadata (source file, page number)
- Efficient similarity-based retrieval of top-k semantically similar chunks
- Parallel storage of chunk text and metadata for result composition

**4. LLM Generation with Context Grounding**
- Dual LLM backend support:
  - Local: Ollama (Llama 3.2) for privacy-preserving deployment
  - Cloud: OpenAI-compatible APIs (GPT-4o-mini, etc.)
- System prompts restrict generation to retrieved context only
- Prevents out-of-context hallucination through explicit instruction
- Source citation with page numbers in responses

**5. Multi-Framework Architecture**
- **Vanilla Pipeline:** 5-stage modular design (ingest → embed → store → retrieve → generate)
- **LangChain Version:** Same workflow using LangChain abstractions for flexibility
- **Agent Layer:** Query routing between document search, calculator, and direct-answer tools
- Demonstrates architectural flexibility and framework-agnostic design patterns

**6. User Interface & Deployment**
- Streamlit web application for interactive PDF upload and chat interface
- Real-time chat history and retrieval visualization
- Environment-based configuration for API keys and model selection
- Reproducible setup with requirements.txt and setup validation scripts

### Key Technical Decisions
- Used embedding model's tokenizer (not tiktoken) to ensure chunks never exceed max-sequence-length
- Normalized vectors for direct cosine-similarity computation via inner product
- Metadata stored separately for clean chunk-to-source alignment
- Context-aware prompting pattern prevents hallucinations without fine-tuning

### Code Quality & Reproducibility
- Modular Python design with separation of concerns
- Configuration management via environment variables
- Setup validation script (`check_setup.py`) for environment verification
- Comprehensive README with usage examples for all execution modes
- Inline documentation explaining leakage prevention and architectural choices

### Interview Points
- "How does PyTorch improve the embedding generation?" → GPU acceleration of batch tensor operations, normalized embeddings for efficient similarity search
- "Why two implementations?" → Demonstrates architectural flexibility and deep understanding of RAG patterns beyond framework specifics
- "How do you prevent hallucinations?" → Context grounding through system prompts and retrieval validation

---

## 2. Loan Recovery Analytics
**Repository:** https://github.com/ManmathKapse12345/Loan-Recovery-Analysis  
**Language:** Python  
**Status:** Complete with 4-Phase Pipeline

### Project Overview
An end-to-end machine learning analytics pipeline analyzing 2.26M LendingClub loan records to predict recovery outcomes for defaulted/charged-off loans. Demonstrates data engineering (schema design, normalization), SQL analysis, predictive modeling with bug-fix documentation, and deployment through dashboard + API.

### Technical Architecture

**Phase 0: Data Sampling**
- Sampled 59K representative rows from full 2.26M-row dataset for faster iteration
- Stratified sampling maintains outcome distribution
- Documented assumptions in `phase0_sampling_summary.md`

**Phase 1: Schema Design & ETL (CSV → PostgreSQL)**
- Normalized flat CSV into 4-table schema:
  - `borrowers`: Demographics (income, employment_length, state)
  - `loans`: Loan attributes (grade, sub_grade, amount, rate, status)
  - `payments`: Payment history (derived from aggregates, not true event log)
  - `recovery_actions`: Recovery outcomes (settlement/collections channel, amounts)
- Created foreign key relationships and indexes for JOIN efficiency
- Documented data derivation rules explicitly (e.g., days_past_due from status buckets)

**Phase 2: SQL Analysis & Feature Engineering**
- 5 business-critical SQL queries:
  1. Recovery rate by grade and disbursement channel
  2. Days-past-due (DPD) distribution by grade
  3. Cohort trends showing charge-off and recovery trends over time
  4. Window-function rankings (rank loans by recovery rate within cohorts)
  5. State-level default and recovery risk
- Created `loan_features` view: single row per loan with all features + target
- Demonstrated advanced SQL: JOINs across 4 tables, window functions, aggregates, time-based cohorts

**Phase 3: Predictive Modeling with Feature Leakage Detection**
- **Critical Achievement:** Caught and documented a feature leakage bug
  - Initial model: 0.998 ROC-AUC (suspiciously high)
  - Root cause: `days_past_due` derived from `current_status`, which also directly defined the target
  - Model was reconstructing its own label, not predicting real signal
  - Fix: Redefined target to `recovery_success` (settlement/collections + amount_recovered > 0)
  - Dropped `days_past_due` from feature set entirely
  - Retrained AUC: 0.628 (trustworthy, realistic)
  
- Trained baseline + random forest:
  - Logistic Regression (baseline with class_weight="balanced")
  - Random Forest (300 estimators, max_depth=12, class_weight="balanced")
  - 80/20 train/test split with stratification
  
- Feature set: 5 numeric (loan_amnt, int_rate, income) + 5 categorical (grade, sub_grade, term, employment_length, state)
- Used scikit-learn Pipelines with ColumnTransformer for reproducible preprocessing:
  - StandardScaler for numeric features
  - OneHotEncoder for categorical features
  
- Comprehensive evaluation:
  - ROC-AUC, classification report (precision/recall/F1)
  - Confusion matrix
  - Feature importance visualization (top 20 features)

**Phase 4: Deployment (Dashboard + API)**
- **Streamlit Dashboard** (`dashboard_app.py`):
  - Interactive charts: recovery rate by grade/channel
  - Cohort trend analysis with dual-axis (charge-off vs. recovery rate)
  - State-level risk heatmap
  - Live prediction form for new loan scenarios
  
- **FastAPI Endpoint** (`api.py`):
  - `POST /predict` endpoint for programmatic model serving
  - Swagger/OpenAPI documentation at `/docs`
  - Accepts loan features in JSON, returns recovery probability
  - Same trained model used by both dashboard and API

### Key Technical Skills Demonstrated
- **Database Design:** Normalized schema, foreign keys, indexing for performance
- **SQL:** JOINs, window functions, CTEs, aggregates, time-based analysis
- **Data Engineering:** CSV → PostgreSQL pipeline, validation, type casting
- **Feature Engineering:** Derived features from raw data, leakage prevention
- **Model Development:** Train/test split, cross-validation, hyperparameter tuning
- **Debugging & Analysis:** Root-cause analysis of model leakage (0.998 → 0.628 AUC)
- **Deployment:** Interactive web dashboard (Streamlit) + REST API (FastAPI)
- **Documentation:** Phase-specific notes documenting assumptions, decisions, and learnings

### Known Limitations & Maturity
- `payments` and `recovery_actions` derived from aggregate snapshots (not real event logs)
- `borrower_id` is 1:1 per loan, not true person-level deduplication
- Model AUC (0.628) reflects hard prediction problem with limited origination-time signal
- Explicitly documented in README and modeling notes

### Interview Points
- "Tell me about a time you caught a critical bug" → Feature leakage story (0.998 → 0.628 AUC)
- "How did you approach the schema design?" → Normalized 4-table schema, documented derivation rules
- "Walk me through your modeling pipeline" → Stratified split, preprocessing, baseline + advanced model, feature importance
- "How do you deploy models?" → Dashboard for analytics, API for programmatic use

---

## 3. Network Security Threat Detection System
**Repository:** https://github.com/ManmathKapse12345/networksecurity  
**Language:** Python  
**Visibility:** Private  
**Status:** Complete MLOps-focused Project

### Project Overview
A production-grade MLOps pipeline for phishing website and network threat detection. Demonstrates end-to-end machine learning workflow: data ingestion → validation → preprocessing → model training → tracking → deployment with automated artifact management and experiment monitoring.

### Technical Architecture

**1. Data Pipeline & Quality Validation**
- Automated data ingestion from multiple sources
- Schema validation ensuring data type and structure compliance
- Kolmogorov-Smirnov (KS) statistical tests for dataset drift detection
- Identifies distribution shifts between training and production data
- Prevents model degradation through data quality gates

**2. Preprocessing Pipeline**
- KNN Imputation for handling missing values (maintains local data structure)
- Handles categorical and numerical feature types
- Reusable preprocessing pipeline for train/inference consistency
- Configurable preprocessing parameters for experimentation

**3. Model Training & Experimentation**
- Trained multiple classification algorithms:
  - Random Forest (ensemble of decision trees)
  - Decision Tree (interpretable baseline)
  - Gradient Boosting (sequential ensemble learning)
  - Logistic Regression (linear baseline)
  - AdaBoost (adaptive boosting for difficult samples)
  
- Hyperparameter tuning using GridSearchCV:
  - Systematic search over parameter grids
  - Cross-validation for robust performance estimation
  - Best parameters selected based on validation score
  
- Evaluation metrics:
  - Precision, Recall, F1-Score (classification metrics)
  - Confusion matrix analysis
  - ROC-AUC and other ranking metrics

**4. Experiment Tracking & Model Versioning**
- **MLflow Integration:**
  - Logs hyperparameters, metrics, and model artifacts
  - Tracks every experiment run with reproducible seeds
  - Model registry for versioning and promotion
  
- **DagsHub Integration:**
  - Remote experiment tracking and collaboration
  - Git-like versioning for data and models
  - Enables team collaboration and reproducibility

**5. Model Serving & Deployment**
- **FastAPI REST APIs:**
  - `/train` endpoint for model training with dataset upload
  - `/predict` endpoint for single-sample predictions
  - `/batch-predict` endpoint for batch inference from CSV files
  - Automatic Swagger/OpenAPI documentation
  
- **Artifact Storage:**
  - AWS S3 integration for model persistence
  - Automatic artifact versioning and retrieval
  - MongoDB for structured metadata storage
  - Enables serverless/cloud deployment

**6. Production Pipeline Architecture**
- Modular design: data → validation → preprocessing → training → evaluation → saving
- Automated artifact lineage tracking (which data → which model)
- Reproducible training with fixed random seeds
- Late-arriving data handling and backfill support
- Data quality monitoring dashboards

### Key MLOps Practices Demonstrated
- **Reproducibility:** Fixed seeds, pinned dependency versions, documented assumptions
- **Monitoring:** Data drift detection via KS tests, performance tracking across runs
- **Scalability:** Batch processing, S3-backed storage, MongoDB for metadata
- **Collaboration:** DagsHub for team experiment sharing, versioning
- **Automation:** Automated validation gates, artifact management
- **API First:** REST interfaces for integration with downstream systems

### Technical Stack
- **ML Frameworks:** Scikit-learn (models, preprocessing, evaluation)
- **Experiment Tracking:** MLflow, DagsHub
- **Cloud Storage:** AWS S3
- **Database:** MongoDB (metadata)
- **API Framework:** FastAPI
- **Version Control:** Git, integrated with DagsHub
- **Data Processing:** Pandas, NumPy

### Code Organization
- Modular components: ingestors, validators, preprocessors, trainers, evaluators
- Configuration management for experiment parameters
- Logging throughout pipeline for debugging
- Unit tests and integration tests for data quality

### Interview Points
- "How do you ensure model reproducibility?" → Fixed seeds, pinned versions, documented assumptions
- "Describe your MLOps workflow" → Data validation → KS tests → preprocessing → training with tracking → S3 storage
- "How do you detect model degradation?" → KS statistical tests for data drift, performance monitoring via MLflow
- "Walk me through your model training pipeline" → GridSearchCV for hyperparameter tuning, multiple algorithms, cross-validation
- "How do you deploy models to production?" → FastAPI endpoints, AWS S3 storage, MongoDB metadata, Swagger docs

### Deployment Patterns
1. Train models locally with MLflow tracking
2. Serialize best model to S3
3. Deploy via FastAPI server
4. Monitor predictions in production
5. Re-train when data drift detected (KS test triggered)

### Scalability Considerations
- S3 handles large model artifacts (no local filesystem constraints)
- MongoDB stores metadata, enabling queries across experiments
- Batch prediction endpoint for high-throughput inference
- FastAPI scales horizontally behind load balancer

---

## Summary: Which Projects to Emphasize for Youngsoft AI/ML Internship

### Best Fit: RAG Q&A Bot
✅ **Generative AI** — Full RAG system with LLMs  
✅ **LLMs & Prompt Engineering** — Ollama + OpenAI integration  
✅ **RAG Systems** — Explicit in JD "Good to Have"  
✅ **PyTorch** — Sentence-Transformers (PyTorch backend)  
✅ **NLP** — Semantic search, embeddings  
✅ **AI Agents** — Query routing agent layer  

### Strong Secondary: Loan Recovery Analytics
✅ **Data Preprocessing** — CSV → PostgreSQL normalization  
✅ **SQL & Databases** — Advanced SQL (JOINs, window functions)  
✅ **Model Training & Evaluation** — Logistic regression, random forest  
✅ **Debugging Skills** — Feature leakage detection and fix (0.998 → 0.628 AUC)  
✅ **Deployment** — Streamlit dashboard + FastAPI  
✅ **Real-world Problem Solving** — Production-grade analytics  

### Tertiary: Network Security MLOps
✅ **MLOps & Automation** — End-to-end automated pipeline  
✅ **Experiment Tracking** — MLflow + DagsHub  
✅ **Data Quality** �� Drift detection via KS tests  
✅ **Deployment & APIs** — FastAPI, AWS S3, MongoDB  
✅ **Scalability** — Production-grade patterns  

### Resume Strategy
1. **Lead with RAG Q&A Bot** (1-2 minutes in interview)
   - Hit all Generative AI / LLM / RAG keywords
   - Mention PyTorch explicitly
   - Show architectural flexibility (vanilla + LangChain)

2. **Follow with Loan Recovery Analytics** (2-3 minutes)
   - Demonstrate end-to-end ML workflow
   - Tell the feature leakage story (shows debugging/root-cause analysis)
   - Mention SQL and data engineering skills
   - Describe deployment (dashboard + API)

3. **Reference Network Security if asked about MLOps** (2-3 minutes)
   - Explain experiment tracking and reproducibility
   - Mention cloud deployment (AWS S3, MongoDB)
   - Discuss data drift detection and production patterns

