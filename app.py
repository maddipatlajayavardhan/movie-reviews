from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langdetect import detect
from deep_translator import GoogleTranslator
import nltk
from nltk.corpus import stopwords
from string import punctuation
import pickle
nltk.download('stopwords')

model = pickle.load(open('model/cnn_model.pkl', 'rb'))


app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('form.html')

@app.route('/', methods=['POST'])
def my_form_post():
    stop_words = stopwords.words('english')
    
    # Get input text and detect language
    text1 = request.form['text1'].lower()
    lang = detect(text1)  # Detect the language of the input text

    if lang == 'ta':  # If language is Tamil
        # Translate Tamil to English
        text1_translated = GoogleTranslator(source='auto', target='en').translate(text1)
    else:
        text1_translated = text1

    text_final = ''.join(c for c in text1_translated if not c.isdigit())
    
    # Remove stopwords
    processed_doc1 = ' '.join([word for word in text_final.split() if word not in stop_words])

    # Sentiment analysis using VADER
    sa = SentimentIntensityAnalyzer()
    dd = sa.polarity_scores(text=processed_doc1)
    compound = round((1 + dd['compound']) / 2, 2)

    return render_template('form.html', final=compound, text1=text1, text2=dd['pos']*100, text5=dd['neg']*100, text4=compound*100, text3=dd['neu']*100)

def home():
    prediction = -1
    if request.method == 'POST':
        positive = int(request.form.get('pos'))
        negative = int(request.form.get('neg'))
        compound = int(request.form.get('com'))
        

        input_features = [[positive, negative, compound]]
        # print(input_features)
        prediction = model.predict(model.transform(input_features))
        # print(prediction)
        
    return render_template('form.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5002, threaded=True)
