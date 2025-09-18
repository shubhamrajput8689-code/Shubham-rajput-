# Shubham-rajput-
Shubham rajput 33 love 
# Hindi AI Chatbot using Machine Learning
# Install libraries: pip install scikit-learn nltk

import random
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download NLTK data (only first time)
nltk.download('punkt')

# ट्रेनिंग डेटा (User के सवाल/इनपुट)
training_sentences = [
    "नमस्ते", "हैलो", "कैसे हो", "सुप्रभात",
    "आप कैसे हैं", "क्या हाल है", "सब ठीक है",
    "अलविदा", "फिर मिलेंगे", "बाय", "चलता हूँ"
]

# हर इनपुट के लिए लेबल
training_labels = [
    "greeting", "greeting", "greeting", "greeting",
    "feeling", "feeling", "feeling",
    "goodbye", "goodbye", "goodbye", "goodbye"
]

# हर लेबल के लिए जवाब
responses = {
    "greeting": ["नमस्ते! कैसे हैं आप?", "हैलो! आपसे मिलकर अच्छा लगा।", "अरे वाह! स्वागत है आपका।"],
    "feeling": ["मैं बिल्कुल ठीक हूँ, आप बताइए?", "सब बढ़िया! आपका हाल कैसा है?", "मैं अच्छा हूँ, धन्यवाद।"],
    "goodbye": ["अलविदा! अपना ख्याल रखना।", "फिर मिलते हैं।", "ठीक है, बाय!"]
}

# टेक्स्ट को नंबर में बदलना
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(training_sentences)

# ML मॉडल ट्रेन करना
model = MultinomialNB()
model.fit(X, training_labels)

# चैटबॉट चलाना
print("🤖 चैटबॉट: नमस्ते! (बंद करने के लिए 'exit' लिखें)")
while True:
    user = input("आप: ")
    if user.lower() == "exit":
        print("🤖 चैटबॉट: अलविदा! फिर मिलेंगे।")
        break
    
    user_vector = vectorizer.transform([user])
    prediction = model.predict(user_vector)[0]
    
    bot_response = random.choice(responses[prediction])
    print("🤖 चैटबॉट:", bot_response)
