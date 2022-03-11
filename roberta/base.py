from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import joblib
import os
import pdfplumber


ROOT_DIR = os.path.dirname(__file__)
PATH_MODELS = os.path.join(ROOT_DIR, "models")
PATH_PDF = os.path.join(ROOT_DIR, "pdf")
PATH_TXT = os.path.join(ROOT_DIR, "txt")

def save_model():
    
    model_name = "deepset/roberta-base-squad2"
    
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    joblib.dump(model, f'{PATH_MODELS}/model_roberta')
    joblib.dump(tokenizer, f'{PATH_MODELS}/tokenizer_roberta')
    

def get_txt(contract_path):
    pdf = pdfplumber.open(f'{PATH_PDF}/{contract_path}')
    ls = []
    for i in pdf.pages:
        ls.append(i.extract_text())
    ls = ' '.join(ls)
    
    txt_path = contract_path.replace('.pdf', '.txt')
    
    with open(f"{PATH_TXT}/{txt_path}", "w") as text_file:
        text_file.write(ls)
    
    return ls


def get_output(question):
    model = joblib.load(f'{PATH_MODELS}/model_roberta')
    tokenizer = joblib.load( f'{PATH_MODELS}/tokenizer_roberta')

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
    #get_output('Why is model conversion important?')
    get_txt('CreditcardscomInc_20070810_S-1_EX-10.33_362297_EX-10.33_Affiliate Agreement.pdf')