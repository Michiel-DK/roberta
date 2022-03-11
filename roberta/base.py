from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import joblib
import os
import pdfplumber


ROOT_DIR = os.path.dirname(__file__)
PATH_MODELS = os.path.join(ROOT_DIR, "models")
PATH_PDF = os.path.join(ROOT_DIR, "pdf")
PATH_TXT = os.path.join(ROOT_DIR, "txt")

TXT_FILE = 'CreditcardscomInc_20070810_S-1_EX-10.33_362297_EX-10.33_Affiliate Agreement.txt'


def save_model():
    
    #model_name = "deepset/roberta-base-squad2"
    model_name = "deepset/bert-base-cased-squad2"
    
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    joblib.dump(model, f'{PATH_MODELS}/model_roberta')
    joblib.dump(tokenizer, f'{PATH_MODELS}/tokenizer_roberta')
    

def save_contract(contract_path):
    pdf = pdfplumber.open(f'{PATH_PDF}/{contract_path}')
    ls = []
    for i in pdf.pages:
        ls.append(i.extract_text())
    ls = ' '.join(ls)
    
    txt_path = contract_path.replace('.pdf', '.txt')
    
    with open(f"{PATH_TXT}/{txt_path}", "w") as text_file:
        text_file.write(ls)
    

def get_context(filename):
    with open(f"{PATH_TXT}/{filename}" , encoding='utf8') as f:
        content = f.read()
    return content
        


def get_output(question):
    model = joblib.load(f'{PATH_MODELS}/model_roberta')
    tokenizer = joblib.load( f'{PATH_MODELS}/tokenizer_roberta')

    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
    
    context = get_context(TXT_FILE)
    import time
    startTime = time.time()
    
    QA_input = {
        'question': question,
        'context': context
    }
    res = nlp(QA_input)
    
    res['answer'] = res['answer'].replace(u'\xa0',' ')
    
    executionTime = (time.time() - startTime)
    
    print('Execution time in seconds: ' + str(executionTime))
    
    print(res)
    
    return res



if __name__ == "__main__":
    #save_model()
    #get_output('Why is model conversion important?')
    #save_contract('CreditcardscomInc_20070810_S-1_EX-10.33_362297_EX-10.33_Affiliate Agreement.pdf')
    #get_context('CreditcardscomInc_20070810_S-1_EX-10.33_362297_EX-10.33_Affiliate Agreement.txt')
    get_output('What is the jurisdiction of the agreement/contract?')