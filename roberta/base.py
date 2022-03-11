from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import joblib
import os
import pdfplumber
import time


ROOT_DIR = os.path.dirname(__file__)
PATH_MODELS = os.path.join(ROOT_DIR, "models")
PATH_PDF = os.path.join(ROOT_DIR, "pdf")
PATH_TXT = os.path.join(ROOT_DIR, "txt")
#MODEL_NAME = "deepset/roberta-base-squad2"
MODEL_NAME = "deepset/bert-base-cased-squad2"

TXT_FILE = 'CreditcardscomInc_20070810_S-1_EX-10.33_362297_EX-10.33_Affiliate Agreement.txt'


def save_model():
    '''function to get model from HugginFace and save locally'''

    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    joblib.dump(model, f'{PATH_MODELS}/model_roberta')
    joblib.dump(tokenizer, f'{PATH_MODELS}/tokenizer_roberta')
    

def save_contract(contract_path):
    '''pdfplumber to convert pdf to list fo string and save locally as txt file'''
    
    pdf = pdfplumber.open(f'{PATH_PDF}/{contract_path}')
    ls = []
    for i in pdf.pages:
        ls.append(i.extract_text())
    ls = ' '.join(ls)
    
    txt_path = contract_path.replace('.pdf', '.txt')
    
    with open(f"{PATH_TXT}/{txt_path}", "w") as text_file:
        text_file.write(ls)
    

def get_context(filename):
    '''function to read txt file and return variable with context'''
    
    with open(f"{PATH_TXT}/{filename}" , encoding='utf8') as f:
        content = f.read()
    return content
        


def get_output(question):
    '''function to get answer via base bert model'''

    
    model = joblib.load(f'{PATH_MODELS}/model_roberta')
    tokenizer = joblib.load( f'{PATH_MODELS}/tokenizer_roberta')

    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
    
    context = get_context(TXT_FILE)
    
    startTime = time.time()
    
    QA_input = {
        'question': question,
        'context': context
    }
    res = nlp(QA_input)
    
    #clean answers
    res['answer'] = res['answer'].replace(u'\xa0',' ')
    
    executionTime = (time.time() - startTime)
    
    print('Execution time in seconds: ' + str(executionTime))
    
    print(res)
    
    return res



if __name__ == "__main__":
    save_model()
    #get_output('Why is model conversion important?')
    #save_contract('CreditcardscomInc_20070810_S-1_EX-10.33_362297_EX-10.33_Affiliate Agreement.pdf')
    #get_context('CreditcardscomInc_20070810_S-1_EX-10.33_362297_EX-10.33_Affiliate Agreement.txt')
    get_output('What is the jurisdiction of the agreement/contract?')