# Math Agent

A simple full-stack app with a Vite/React **frontend** and a Python **backend**.

## Prerequisites

- **Node.js** ≥ 18 (includes npm)
- **Python** ≥ 3.9 (Windows: `python` or `py` command available)
- **Git**

> Create a `.env` file (not committed) if your backend needs secrets. See **Environment** below.

---

## Project Structure


---

## Quick Start

Open **two terminals** (or two tabs).

### 1) Frontend

```bash
cd frontend
npm install
npm run dev
VITE v4.x ready
  ➜  Local:   http://localhost:3000/
cd backend

# (recommended) create and activate a virtual env
python -m venv .venv         # or: py -3 -m venv .venv
.venv\Scripts\activate       # PowerShell/CMD on Windows
# source .venv/bin/activate  # macOS/Linux

# install dependencies (if requirements.txt exists)
pip install -r requirements.txt

# run the server
python run.py                # Windows: python run.py  OR  py run.py
# backend/.env (example)
PORT=8000
DEBUG=true
OPENAI_API_KEY=your_key_here
