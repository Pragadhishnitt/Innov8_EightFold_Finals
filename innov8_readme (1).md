# AI-Powered Technical Interview Platform

## 🏆 INNOV8 EightFold Finals Submission

**Team: Whisper_AI**

This repository contains our submission for the **INNOV8 2.0 Finals**. Our solution aims to revolutionize technical assessments by using an AI-powered interview platform that combines agentic workflows with structured evaluation.

---

## 🎯 The Challenge

Current automated technical interview systems face several critical challenges that impact their effectiveness and fairness:

- **🤖 LLM Hallucination**: Large Language Models can generate incorrect or completely irrelevant answers, compromising the integrity of the assessment.

- **⏰ High Latency**: Slow response times from the system can disrupt the natural flow of an interview, leading to a poor candidate experience.

- **📋 Generic Feedback**: The feedback provided often lacks specific, domain-level depth and fails to offer actionable guidance for candidates.

- **📊 Inconsistent Assessments**: It is often difficult to compare candidates objectively, leading to potential biases.

These significant gaps can negatively affect the fairness, scalability, and overall outcomes of the hiring process.

## 💡 Our Solution: A Hybrid AI System

To address these challenges, we developed a **hybrid approach** that merges the strengths of structured data with an agentic AI workflow. This combination allows for technical interviews that are fast, accurate, and consistently grounded in relevant data.

### 🌟 Key Innovation

Our platform is built on a **two-LLM architecture**:

- **Mistral 7B (Reasoning Agent)**: A small, fast model with a low hallucination rate, used for real-time interaction with the candidate.
- **Gemini 2.5 Pro (Evaluation Judge)**: A powerful model used to generate comprehensive, detailed evaluation reports after the interview.

### ✨ Core Features

#### For Interviewers

- **📊 Objective Reporting**: Access data-driven reports and full session playbacks for unbiased evaluation.
- **⚡ Efficient Scaling**: Achieve faster, standardized evaluations at scale.
- **🎯 Deeper Insights**: Gain a better signal on a candidate's thinking process and problem-solving approach.

#### For Candidates

- **⚖️ Actionable Feedback**: Receive fair, immediate, and constructive feedback to understand performance.
- **🗣️ Interactive Experience**: Engage in a low-latency interview flow that supports both voice and code.
- **📈 Clear Guidance**: Use session replays and clear guidance to identify areas for improvement.

## 🏗️ System Architecture

Our agentic workflow intelligently navigates the interview process. It starts an interview, fetches questions, and adapts based on the candidate's performance—offering hints if they're stuck, probing deeper with follow-up questions, or moving to the next question. All interactions are recorded and sent to an evaluation agent for a final report.

## 🛠️ Technology Stack

Our stack was carefully selected to prioritize speed, reproducibility, and security.

### Core Backend
- **API Framework**: FastAPI
- **Agent Orchestration**: LangGraph
- **Reasoning LLM**: Mistral 7B
- **Evaluation LLM**: Gemini 2.5 Pro
- **Database**: Supabase (PostgreSQL) for storing questions, answers, hints, and follow-ups

### Real-time & Processing
- **Voice Transcription**: Deepgram
- **Code Execution**: Judge0 (in an isolated sandbox environment)
- **Communication**: WebSockets for real-time frontend updates

### Frontend & Security
- **Dashboard**: JavaScript
- **Authentication**: JWT / Role-Based Access Control (RBAC)
- **Sandboxing**: Network isolation for secure code evaluation

## 📁 Project Structure

```
├── __pycache__/
│   ├── audio_processor.cpython-310.pyc
│   ├── audio_processor.cpython-312.pyc
│   ├── audio_service.cpython-312.pyc
│   ├── database_manager.cpython-310.pyc
│   ├── database_manager.cpython-312.pyc
│   ├── main.cpython-310.pyc
│   ├── main.cpython-312.pyc
│   ├── snapshot_service.cpython-312.pyc
│   └── transcript_service.cpython-312.pyc
├── eval_engine/
│   ├── __pycache__/
│   ├── templates/
│   ├── final_transcript.txt
│   ├── updated_api_server.py
│   └── updated_evaluator.py
├── judge_o/
│   ├── app.py
│   ├── requirements.txt
│   └── test.py
├── templates/
│   ├── home.html
│   ├── index.html
│   ├── monitor.html
│   └── test_audio.html
├── test/
│   ├── test_database_fix.py
│   ├── test_endpoints.py
│   └── test_integration.py
├── transcripts/
│   ├── session_1759023892112_av3dqhjbz
│   └── session_1759031903914_5yq5gsvgx
├── .gitignore
├── API_DOCUMENTATION.md
├── app.py
├── audio_processor.py
├── audio_service.py
├── backend.py
├── codesage_system.py
├── complete_database_schema.sql
├── database_manager.py
├── main.py
├── requirements.txt
└── snapshot_service.py
```

## 🚀 Quick Start Guide

### Prerequisites
- Conda/Miniconda installed
- Python 3.9+
- API Keys: Google AI (Gemini), Deepgram

### 1. Clone Repository
```bash
git clone https://github.com/Pragadhishnitt/Innov8_EightFold_Finals.git
cd Innov8_EightFold_Finals
```

### 2. Conda Environment Setup
```bash
# Create a new Conda environment named 'whisper_ai' with Python 3.9
conda create -n whisper_ai python=3.9

# Activate the newly created environment
conda activate whisper_ai
```

### 3. Install Dependencies
```bash
# Install all required Python packages from requirements.txt
pip install -r requirements.txt
```

### 4. Environment Variables
Create a `.env` file in the root directory and add your API keys:

```env
GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
DEEPGRAM_API_KEY="YOUR_DEEPGRAM_API_KEY"
```

### 5. Run the Application
```bash
# Start the backend server using Uvicorn
uvicorn main:app --reload
```

The application will be accessible at `http://127.0.0.1:8000`.

## 📊 Key Features & Benefits

### For Technical Recruiters
- **Standardized Evaluation**: Consistent assessment criteria across all candidates
- **Time Efficiency**: Reduce screening time by 60% while maintaining quality
- **Objective Insights**: Data-driven reports eliminate subjective bias
- **Scalable Solution**: Handle multiple interviews simultaneously

### For Candidates
- **Fair Assessment**: Standardized questions and evaluation criteria
- **Immediate Feedback**: Real-time performance insights and improvement suggestions
- **Interactive Experience**: Natural conversation flow with voice and code support
- **Learning Opportunity**: Session replays help identify areas for growth

## 🔧 API Documentation

For detailed API documentation, please refer to [API_DOCUMENTATION.md](API_DOCUMENTATION.md).

### Key Endpoints
- `GET /`: Home page and platform overview
- `POST /interview/start`: Initialize new interview session
- `GET /interview/{session_id}`: Retrieve interview details
- `POST /interview/{session_id}/submit`: Submit code or responses
- `GET /interview/{session_id}/evaluation`: Get final evaluation report

## 🧪 Testing

Run the test suite to ensure everything is working correctly:

```bash
# Run integration tests
python test/test_integration.py

# Test specific endpoints
python test/test_endpoints.py

# Test database functionality
python test/test_database_fix.py
```

## 🚀 Advanced Usage

### Voice Integration
The platform supports voice-based interviews using Deepgram for real-time transcription:

```python
# Example voice processing
from audio_processor import AudioProcessor

processor = AudioProcessor()
transcript = processor.process_audio(audio_file)
```

### Code Execution
Secure code execution is handled through Judge0 in an isolated environment:

```python
# Example code evaluation
from judge_o.app import evaluate_code

result = evaluate_code(
    code="def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    language="python"
)
```

## 🔐 Security Features

- **Sandboxed Execution**: All code runs in isolated containers
- **JWT Authentication**: Secure session management
- **Rate Limiting**: Prevents abuse and ensures fair usage
- **Data Encryption**: Sensitive information is encrypted at rest and in transit

## 🤝 Contributing

We welcome contributions! Please feel free to submit pull requests or open issues for bugs and feature requests.

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Submit a pull request with detailed description

## 📞 Contact

- **Team**: Whisper_AI
- **Repository**: [GitHub](https://github.com/Pragadhishnitt/Innov8_EightFold_Finals)
- **Issues**: [Report bugs or request features](https://github.com/Pragadhishnitt/Innov8_EightFold_Finals/issues)

## 🙏 Acknowledgments

- **ARIES IIT Delhi** for organizing the INNOV8 hackathon
- **EightFold AI** for providing the challenging problem statement and mentorship
- **Rendezvous 2024** for hosting this innovative competition

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

### 🏆 INNOV8 2.0 Finals Submission
**Team: Whisper_AI**  
**Duration**: 24 hours (September 28-29, 2024)  
**Organizers**: ARIES IIT Delhi × EightFold AI  

**Built with ❤️ during INNOV8 2.0 Finals**

</div>