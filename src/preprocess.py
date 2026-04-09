# NLP preprocessing logic
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def clean_message(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    text_no_url = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    text_no_punc = text_no_url.translate(str.maketrans('', '', string.punctuation))
    
    tokens = word_tokenize(text_no_punc)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if not w in stop_words]
    
    return " ".join(filtered_tokens)

def analyze_url_structure(text):
    if not isinstance(text, str):
        return 0
        
    suspicious_patterns = [
        r'bit\.ly', r'tinyurl', r'short\.url', r'\.tk', r'\.ml', r'\.ga', r'\.cf', r'\.gq',
        r'@', r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', r'free', r'claim', r'reward', r'cashback'
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return 1
            
    return 0