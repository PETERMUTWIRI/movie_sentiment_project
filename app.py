import streamlit as st
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from scipy.special import softmax

# Load your model and tokenizer
model_path = "petermutwiri/Tiny_Bert_Cupstone"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Preprocess text (username and link placeholders)
#In summary, this preprocessing function helps ensure that usernames and links in the input text do not interfere with the sentiment analysis performed by the model. It replaces them with placeholder tokens to maintain the integrity of the text's structure while anonymizing or standardizing specific elements.

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def sentiment_analysis(text):
    text = preprocess(text)

    # PyTorch-based models
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores_ = output[0][0].detach().numpy()
    scores_ = softmax(scores_)

    # Format output dict of scores
    labels = ['Negative', 'Positive']
    scores = {l: float(s) for (l, s) in zip(labels, scores_)}

    return scores

# Streamlit app layout with two columns
st.title("Movie Review App")
st.write("Welcome to our Movie Review App powered by the state-of-the-art RoBERTa and TinyBERT models with an impressive accuracy score of 0.93 and 0.83 respectively. Get ready to dive into the world of cinema and discover the sentiments behind your favorite movies. Whether it's a thrilling 9 or a heartwarming 3, our app not only predicts the sentiment but also rates the movie on a scale of 1 to 10. Express your thoughts, press 'Analyze,' and uncover the emotional depth of your movie review")
st.image("Assets/movie_review.png", caption="Sentiments examples", use_column_width=True)

# Input text area for user to enter a tweet in the left column
input_text = st.text_area("Write your movie review here...")

# Output area for displaying sentiment in the right column
if st.button("Analyze Review"):
    if input_text:
        # Perform movie review using the loaded model
        scores = sentiment_analysis(input_text)

        # Display sentiment scores in the right column
        st.text("Sentiment Scores:")
        for label, score in scores.items():
            st.text(f"{label}: {score:.2f}")

        # Determine the overall sentiment label
        sentiment_label = max(scores, key=scores.get)

        # Map sentiment labels to human-readable forms
        sentiment_mapping = {
            "Negative": "Negative",
            "Positive": "Positive"
        }
        sentiment_readable = sentiment_mapping.get(sentiment_label, "Unknown")

        # Display the sentiment label in the right column
        st.text(f"Sentiment: {sentiment_readable}")

# Button to Clear the input text
if st.button("Clear Input"):
    input_text = ""
