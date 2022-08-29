from flask import Flask, render_template, request
import requests
import datetime
import gensim.downloader as gensim_api
from gensim.matutils import softcossim
from gensim import corpora
from flask import Flask
from flask_cors import CORS


app = Flask(__name__)
cors = CORS(app)
word2vec_model300 = gensim_api.load('word2vec-google-news-300')

convos = []

def compare_sentences(sentence1, sentence2, model=word2vec_model300):
    sentence1 = sentence1.split()
    sentence2 = sentence2.split()

    documents = [sentence1, sentence2]
    dictionary = corpora.Dictionary(documents)
    ws1 = dictionary.doc2bow(sentence1)
    ws2 = dictionary.doc2bow(sentence2)

    similarity_matrix = model.similarity_matrix(dictionary, tfidf=None, threshold=0.0, exponent=2.0,
                                                nonzero_limit=100)
    return softcossim(ws1, ws2, similarity_matrix)

@app.route("/")
def home(): return render_template("index.html")

time = lambda: datetime.datetime.now().strftime("%H:%M")

@app.route("/get")
def get():
    fp = open('cssrs.txt','r')
    cssrs = fp.read().splitlines()
    fp.close()
    fp2 = open('methods.txt','r')
    methods = fp2.read().splitlines()
    fp2.close()
    userText = request.args.get('msg')
    '''
    if cssrs[0][2:] in userText:
        
        if compare_sentences(cssrs[0][2:],userText) > -0.5:
        	botText = "SUICIDAL INDICATION ALERT!!!"
        
    else:
        botText = requests.post("https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill", headers={"Authorization": "Bearer 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"}, json={"inputs": {"past_user_inputs": [i[0] for i in convos], "generated_responses": [i[1] for i in convos], "text": userText}, "parameters": {"repetition_penalty": 1.33}}).json()["generated_text"]
    '''
    botText = requests.post("https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill", headers={"Authorization": "Bearer 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"}, json={"inputs": {"past_user_inputs": [i[0] for i in convos], "generated_responses": [i[1] for i in convos], "text": userText}, "parameters": {"repetition_penalty": 1.33}}).json()["generated_text"]    
    convos.append((userText, botText))
    # return {"user": f"<div class='container darker'><p>{userText}</p><span class='time-right'>{time()}</span></div>", "bot": f"<div class='container'><p>{botText}</p><span class='time-left'>{time()}</span></div>"}
    return botText

if __name__ == "__main__": app.run()
