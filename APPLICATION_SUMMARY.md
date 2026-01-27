# Bharagen Yogaja QuestionBank Generator - Application Summary

## Overview

Bharagen Yogaja QuestionBank Generator is an intelligent academic assessment tool designed to generate high-quality, NCERT/CBSE-aligned questions across multiple subjects (Mathematics, Physics, Chemistry, and Biology). The application leverages advanced Large Language Models (LLMs) and a Retrieval-Augmented Generation (RAG) pipeline to create contextually accurate, curriculum-aligned questions with configurable cognitive depth levels.

## Key Features

### 1. **Multi-Model LLM Support**
   - **Cloud Models**: Google Gemini 3 Flash, OpenAI GPT-4o, Groq (Llama 3.1 8B, Llama 3.3 70B, GPT OSS variants)
   - **Local Models**: Llama 3, Qwen 2.5, IBM Granite 3
   - **Param Models**: Param-1-7B-MoE (with and without RAG enhancement)

### 2. **LLM Board Architecture (Council Flow)**
   The application implements a collaborative three-stage question generation process:
   - **Stage 1 - Chairman Proposal**: A designated LLM (chairman) generates initial question drafts
   - **Stage 2 - Board Member Review**: Multiple LLM board members provide critical feedback, ratings, and alternative suggestions
   - **Stage 3 - Chairman Synthesis**: The chairman synthesizes the final questions incorporating the best feedback from board members
   
   This architecture ensures higher quality questions through collaborative evaluation and refinement.

### 3. **RAG (Retrieval-Augmented Generation) Pipeline**
   - Vector database built from NCERT PDFs using FAISS and Sentence Transformers
   - Semantic retrieval of relevant context chunks for topic and theme
   - Ensures questions are grounded in actual NCERT curriculum content
   - Supports both RAG-enhanced and standard question generation modes

### 4. **Cognitive Depth Configuration**
   Questions can be generated at four Depth of Knowledge (DOK) levels:
   - **DOK 1**: Recall & Reproduction
   - **DOK 2**: Skills & Concepts
   - **DOK 3**: Strategic Thinking
   - **DOK 4**: Extended Thinking

### 5. **Question Type Diversity**
   Supports multiple question formats:
   - Multiple Choice Questions (MCQ)
   - Short Answer
   - True/False
   - Essay Prompt
   - Case Study Analysis
   - Fill-in-the-Blanks

### 6. **Modern Web Interface**
   - Single-page application with responsive design
   - Dark/light theme support
   - Interactive question navigation
   - Board opinion visualization
   - Source material sidebar for RAG context

## Technical Architecture

### Backend
- **Framework**: FastAPI (Python)
- **Model Integration**: 
  - Transformers library for local model execution
  - API clients for cloud services (Gemini, OpenAI, Groq)
  - Quantized model loading (4-bit) for efficient inference
- **RAG System**: FAISS vector database with SentenceTransformer embeddings
- **Async Processing**: Asynchronous execution for parallel board member reviews

### Frontend
- **Technology**: Vanilla JavaScript with Tailwind CSS
- **Architecture**: Single-page application with dynamic content rendering
- **Features**: Real-time question generation, progress tracking, answer reveal system

### Infrastructure
- Reverse proxy support for multi-service routing
- Environment-based configuration for API keys and model paths

## Current Implementation Status

### Model Performance Notes

1. **Param-1-7B-MoE Performance**
   - The Param-1-7B-MoE model has been integrated for both chairman and board member roles
   - **Current Limitation**: The model exhibits extremely slow inference times, which impacts the user experience, especially in the board flow where multiple sequential and parallel calls are required
   - The model is loaded with 4-bit quantization to optimize memory usage, but generation speed remains a bottleneck

2. **Testing Recommendations**
   - For testing and demonstration purposes, we recommend using **Groq** models (e.g., `groq-llama-8b`, `groq-llama-70b`) which provide fast inference times
   - Groq models offer excellent performance for rapid prototyping and evaluation of the board architecture

### Future Development Roadmap

1. **17B MoE Model Integration (Post-SFT 1)**
   - **Plan**: Integrate the 17B MoE model that will be received after Supervised Fine-Tuning (SFT) iteration 1 from Nihar
   - **Advantages**: 
     - Larger model capacity for improved question quality
     - **Multilingual support** - This will enable question generation in multiple Indian languages, significantly expanding the application's accessibility and utility
     - Expected better performance compared to the current 7B model

2. **Additional Curriculum Integration**
   - **State Board**: Planning to extend support to various state board curricula
   - **NIOS (National Institute of Open Schooling)**: Integration planned to support alternative education pathways

## Use Cases

1. **Student Practice**: Generate customized practice questions aligned with NCERT curriculum
2. **Teacher Assessment**: Create assessment materials with specific cognitive depth requirements
3. **Curriculum Development**: Generate question banks for different chapters and topics
4. **Research**: Study the effectiveness of collaborative LLM architectures in educational content generation

## Technical Highlights

- **Modular Architecture**: Separated concerns with dedicated modules for model execution, council flow, and RAG retrieval
- **Scalable Design**: Support for multiple concurrent model backends
- **Context-Aware Generation**: RAG pipeline ensures questions are grounded in actual curriculum content
- **Quality Assurance**: Multi-stage review process through LLM board architecture

## Conclusion

Bharagen Yogaja QuestionBank Generator represents a comprehensive approach to AI-powered educational content generation, combining state-of-the-art LLM technology with curriculum-specific knowledge retrieval. The LLM board architecture demonstrates an innovative approach to improving question quality through collaborative evaluation. While current performance limitations exist with the Param-1-7B-MoE model, the planned integration of the multilingual 17B MoE model and expansion to additional curricula (State Boards, NIOS) will significantly enhance the application's capabilities and reach.

---

*For technical questions or demonstration requests, please contact the development team.*
