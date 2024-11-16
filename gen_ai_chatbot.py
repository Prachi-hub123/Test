# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 23:53:57 2023

@author: rohit
"""


from flask import Flask, request, jsonify, render_template
from gen_ai import get_llm

path = 'C://Users/rohit/Documents/2024/LLM/data/'

chatbot = get_llm(path)

# Create Flask app
app = Flask(__name__)

# http://127.0.0.1:5000/
@app.route("/")
def home():
    return render_template("index.html")

# Define a route for the prediction endpoint
@app.route('/get_answer', methods=["POST"])

def get_answer():
    question = request.form['question']  # Get the review from the form
    answer = chatbot.invoke({'question':question, 'chat_history':[]})
    return(jsonify(answer))

# Run the app if executed directly
if __name__ == "__main__":
    app.run()