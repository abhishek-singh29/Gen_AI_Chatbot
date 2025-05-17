from fastapi import FastAPI
from pydantic import BaseModel
from model.run_inference import answer_question

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
async def root():
    return {"message": "Welcome to the GenAI Q&A API. Use POST /ask to get answers."}

@app.post("/ask")
async def get_answer(request: QuestionRequest):
    answer = answer_question(request.question)
    return {"answer": answer}
