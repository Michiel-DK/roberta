from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import joblib
import os

ROOT_DIR = os.path.dirname(__file__)
PATH = os.path.join(ROOT_DIR, "models")

def save_model():
    

    
    model_name = "deepset/roberta-base-squad2"
    
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    joblib.dump(model, f'{PATH}/model_roberta')
    joblib.dump(tokenizer, f'{PATH}/tokenizer_roberta')
    

def get_output(question):
    model = joblib.load(f'{PATH}/model_roberta')
    tokenizer = joblib.load( f'{PATH}/tokenizer_roberta')

    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
    QA_input = {
        'question': question,
        'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
    }
    res = nlp(QA_input)
    
    print(res)
    return res

if __name__ == "__main__":
    #save_model()
    get_output('Why is model conversion important?')