# FinHire AI

Run: `pip install -r requirements.txt`, then `cd backend && GROQ_API_KEY=your_key uvicorn main:app --port 8000`. Open http://localhost:8000

Without GROQ_API_KEY the app uses rule-based explanations. Data is stored in backend/finhire.db (SQLite).
