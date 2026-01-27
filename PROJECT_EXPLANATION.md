# BharatGen Yogaja Learning Path Explorer for Spoken Tutorial

## Project Overview

**BharatGen Yogaja Learning Path Explorer for Spoken Tutorial** is an intelligent educational platform that integrates AI-powered question generation with structured learning paths and spoken tutorial content. The application provides a comprehensive workflow for students to explore courses, follow personalized learning paths, and practice with contextually relevant questions aligned with NCERT/CBSE curriculum.

## Core Architecture

The project consists of multiple integrated services:

1. **LLM Board Service** (`/llmboard` - Port 8002): AI-powered question bank generator with collaborative LLM architecture
2. **Spoken Tutorial Service** (`/spokentutorial` - Port 5000): Integration with Spoken Tutorial content
3. **Reverse Proxy** (Port 9000): Routes requests to appropriate services

## Current Implementation Status

### ✅ Fully Implemented Features

#### 1. **Question Bank Generator (LLM Board)**
- **Multi-Model LLM Support**: 
  - Cloud Models: Google Gemini 3 Flash, OpenAI GPT-4o, Groq (Llama 3.1 8B, Llama 3.3 70B, GPT OSS variants)
  - Local Models: Llama 3, Qwen 2.5, IBM Granite 3
  - Param Models: Param-1-7B-MoE (with and without RAG enhancement)

- **LLM Board Architecture (Council Flow)**:
  - **Stage 1**: Chairman Proposal - Initial question drafts
  - **Stage 2**: Board Member Review - Multiple LLMs provide feedback, ratings, and alternatives
  - **Stage 3**: Chairman Synthesis - Final questions incorporating best feedback

- **RAG Pipeline**: 
  - Vector database from NCERT PDFs using FAISS and Sentence Transformers
  - Semantic retrieval of relevant context chunks
  - Ensures curriculum-aligned question generation

- **Question Types**: 
  - Multiple Choice Questions (MCQ)
  - Short Answer
  - True/False
  - Essay Prompt
  - Case Study Analysis
  - Fill-in-the-Blanks

- **Cognitive Depth Levels (DOK)**:
  - DOK 1: Recall & Reproduction
  - DOK 2: Skills & Concepts
  - DOK 3: Strategic Thinking
  - DOK 4: Extended Thinking

- **Subjects Currently Supported**:
  - **Mathematics**: 26 chapters (Sets, Relations and Functions, Trigonometric Functions, Complex Numbers, etc.)
  - **Physics**: 27 chapters (Units and Measurements, Motion, Laws of Motion, Thermodynamics, etc.)
  - **Chemistry**: 19 chapters (Basic Concepts, Structure of Atom, Chemical Bonding, Organic Chemistry, etc.)
  - **Biology**: 33 chapters (The Living World, Cell Biology, Genetics, Evolution, Ecology, etc.)

### 🚧 Work in Progress Features

#### 1. **Learning Path Tweaking**
- **Status**: Work in Progress
- **Purpose**: Enable dynamic adjustment and optimization of learning paths based on:
  - Student performance analytics
  - Topic difficulty assessment
  - Prerequisite dependencies
  - Learning pace adaptation
- **Planned Features**:
  - Adaptive path recommendations
  - Difficulty-based sequencing
  - Prerequisite validation
  - Performance-based path adjustments

#### 2. **Topic Graph Verification**
- **Status**: Work in Progress
- **Purpose**: Validate and maintain the integrity of topic relationships and dependencies
- **Planned Features**:
  - Topic dependency graph validation
  - Prerequisite relationship verification
  - Circular dependency detection
  - Topic coverage completeness checks
  - Cross-subject topic mapping

## Courses to be Added

The following courses are planned for integration into the learning path explorer:

### STEM Courses

1. **Computer Science**
   - Programming Fundamentals (Python, C++, Java)
   - Data Structures and Algorithms
   - Database Management Systems
   - Web Development (HTML, CSS, JavaScript)
   - Software Engineering
   - Operating Systems
   - Computer Networks
   - Artificial Intelligence and Machine Learning

2. **Engineering Courses**
   - Engineering Mathematics
   - Engineering Physics
   - Engineering Chemistry
   - Engineering Graphics
   - Basic Electrical Engineering
   - Basic Mechanical Engineering
   - Basic Civil Engineering
   - Basic Electronics Engineering

3. **Advanced Mathematics**
   - Linear Algebra
   - Calculus (Differential, Integral, Multivariable)
   - Differential Equations
   - Probability and Statistics
   - Discrete Mathematics
   - Number Theory
   - Graph Theory

4. **Advanced Sciences**
   - Advanced Physics (Quantum Mechanics, Relativity)
   - Advanced Chemistry (Organic Synthesis, Inorganic Chemistry)
   - Advanced Biology (Molecular Biology, Genetics, Biotechnology)
   - Environmental Science

### Language and Humanities

5. **English Language**
   - Grammar and Composition
   - Literature
   - Communication Skills
   - Technical Writing

6. **Social Sciences**
   - History
   - Geography
   - Political Science
   - Economics
   - Sociology

7. **Indian Languages**
   - Hindi
   - Sanskrit (already has service route at `/sanskrit`)
   - Regional Languages (Tamil, Telugu, Kannada, Malayalam, Bengali, etc.)

### Professional and Skill Development

8. **Business and Management**
   - Business Studies
   - Accountancy
   - Economics
   - Entrepreneurship

9. **Arts and Design**
   - Fine Arts
   - Design Principles
   - Digital Design

10. **Vocational Courses**
    - Information Technology
    - Agriculture
    - Healthcare
    - Hospitality
    - Tourism

### Competitive Exam Preparation

11. **JEE (Joint Entrance Examination)**
    - JEE Main Preparation
    - JEE Advanced Preparation
    - Physics, Chemistry, Mathematics modules

12. **NEET (National Eligibility cum Entrance Test)**
    - NEET Preparation
    - Biology, Physics, Chemistry modules

13. **UPSC Civil Services**
    - General Studies
    - Optional Subjects
    - Current Affairs

14. **GATE (Graduate Aptitude Test in Engineering)**
    - Engineering Discipline-specific modules
    - General Aptitude

15. **CAT (Common Admission Test)**
    - Quantitative Aptitude
    - Verbal Ability
    - Data Interpretation
    - Logical Reasoning

## Complete Workflow Envisaged

The application demonstrates the entire workflow as envisioned:

### 1. **Course Selection and Exploration**
   - Students browse available courses
   - View course structure and learning objectives
   - Understand prerequisites and dependencies
   - Select courses based on their learning goals

### 2. **Learning Path Generation**
   - System generates personalized learning paths based on:
     - Student's current knowledge level
     - Learning objectives
     - Available time
     - Preferred learning style
   - Path includes:
     - Topic sequence
     - Spoken tutorial video recommendations
     - Practice question sets
     - Assessment checkpoints

### 3. **Content Delivery Integration**
   - **Spoken Tutorial Integration**: 
     - Links to relevant spoken tutorial videos for each topic
     - Video-based learning for visual and auditory learners
     - Step-by-step tutorial content
   
   - **Question Bank Integration**:
     - Context-aware questions generated using LLM Board
     - Questions aligned with current learning topic
     - Multiple difficulty levels (DOK 1-4)
     - Various question types for comprehensive assessment

### 4. **Adaptive Learning**
   - **Performance Tracking**:
     - Monitor student progress through topics
     - Track question performance
     - Identify knowledge gaps
   
   - **Path Adjustment** (Work in Progress):
     - Automatically adjust learning path based on performance
     - Recommend additional practice for weak areas
     - Accelerate through mastered topics
     - Suggest prerequisite review when needed

### 5. **Assessment and Feedback**
   - **Formative Assessment**:
     - Practice questions after each topic
     - Immediate feedback with detailed explanations
     - Board opinions showing AI reasoning process
   
   - **Summative Assessment**:
     - Chapter-end assessments
     - Course completion tests
     - Comprehensive evaluations

### 6. **Topic Graph Navigation**
   - **Visual Topic Relationships** (Work in Progress):
     - Interactive graph showing topic dependencies
     - Prerequisite visualization
     - Cross-topic connections
     - Subject interconnections
   
   - **Verification System** (Work in Progress):
     - Validate topic graph integrity
     - Ensure no circular dependencies
     - Verify prerequisite chains
     - Check topic coverage completeness

### 7. **Multi-Modal Learning Experience**
   - **Video Learning**: Spoken Tutorial videos
   - **Interactive Practice**: AI-generated questions
   - **Text Resources**: NCERT/Curriculum-aligned content
   - **RAG-Enhanced Context**: Relevant curriculum chunks

### 8. **Collaborative Learning Support**
   - **LLM Board Insights**: 
     - View how different AI models evaluate questions
     - Understand reasoning behind question generation
     - Learn from alternative perspectives
   
   - **Source Material Access**:
     - View RAG-retrieved context chunks
     - Understand question grounding in curriculum
     - Access reference materials

## Technical Implementation

### Backend Architecture
- **Framework**: FastAPI (Python)
- **Model Integration**: Transformers, API clients for cloud services
- **RAG System**: FAISS vector database with SentenceTransformer embeddings
- **Async Processing**: Parallel board member reviews

### Frontend Architecture
- **Technology**: Vanilla JavaScript with Tailwind CSS
- **Design**: Single-page application with responsive design
- **Features**: Dark/light theme, interactive navigation, real-time updates

### Service Integration
- **Reverse Proxy**: Routes requests to appropriate services
- **Microservices**: Separate services for different functionalities
- **API Communication**: RESTful APIs for service interaction

## Future Enhancements

1. **Multilingual Support**: 
   - Integration of 17B MoE model with multilingual capabilities
   - Question generation in multiple Indian languages

2. **Extended Curriculum Support**:
   - State Board curricula
   - NIOS (National Institute of Open Schooling)
   - International curricula (IB, IGCSE)

3. **Advanced Analytics**:
   - Learning analytics dashboard
   - Performance prediction models
   - Personalized recommendations engine

4. **Social Learning Features**:
   - Discussion forums
   - Peer learning groups
   - Collaborative problem-solving

5. **Mobile Application**:
   - Native mobile apps
   - Offline learning support
   - Push notifications for learning reminders

## Conclusion

BharatGen Yogaja Learning Path Explorer for Spoken Tutorial represents a comprehensive vision for AI-enhanced education. The application successfully demonstrates the core workflow of integrating question generation, learning path management, and spoken tutorial content. While learning path tweaking and topic graph verification are currently in development, the foundation is solid and the complete workflow is clearly demonstrated through the existing implementation.

The platform aims to provide a personalized, adaptive, and comprehensive learning experience that combines the best of AI-powered content generation, structured learning paths, and multimedia educational resources.

---

*For technical questions or demonstration requests, please contact the development team.*
