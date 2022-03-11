from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from roberta.base import *


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Hello world"}

@app.get("/question_transformer")
def question_transformer(question):
    return get_output_base(question)

@app.get("/question_farm")
def question_farm(question):
    return get_output_farm(question)