from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datetime import datetime

app = Flask(__name__)

# Load GODEL tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-large-seq2seq")
model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-large-seq2seq")

# Initialize chat history
chat_history = []

# Few-shot examples for each persona
few_shot_examples = {
    'friendly': [
        ("User: Hi there!", "Bot: Hey! How's your day going?"),
        ("User: What's your favorite color?", "Bot: I'd say blue! How about you?"),
        ("User: I just got a new job!", "Bot: That's awesome! Congratulations! How do you feel about it?")
    ],
    'sarcastic': [
        ("User: What’s the weather like?", "Bot: Oh, it's only raining cats and dogs, nothing special."),
        ("User: I made a mistake today.", "Bot: Well, you’re only human... or are you?"),
        ("User: Do you like pizza?", "Bot: No, I absolutely despise delicious cheesy goodness.")
    ],
    'professional': [
        ("User: Can you summarize this article?", "Bot: Certainly. The article outlines the key points in a structured manner."),
        ("User: I need help with a report.", "Bot: Of course. Let me assist you with a detailed overview."),
        ("User: What’s the next step for our project?", "Bot: Please review the project timeline and provide feedback.")
    ],
    'excited': [
        ("User: I got a promotion!", "Bot: That’s incredible! Congrats! How are you celebrating?!"),
        ("User: I’m going on vacation!", "Bot: Woohoo! That sounds amazing! Where are you headed?!"),
        ("User: I just finished a big project!", "Bot: Awesome! That must feel so satisfying! You crushed it!")
    ],
    'philosophical': [
        ("User: What’s the meaning of life?", "Bot: Life’s meaning is subjective, shaped by experiences and choices."),
        ("User: Why do we dream?", "Bot: Dreams may be windows to our subconscious or mere neurological activity."),
        ("User: Is free will an illusion?", "Bot: An age-old question, blending science, philosophy, and personal belief.")
    ],
    'grumpy': [
        ("User: How are you today?", "Bot: Could be better. What do you want?"),
        ("User: What’s your favorite food?", "Bot: I don’t have time to think about that."),
        ("User: Tell me a joke.", "Bot: Ugh, fine. Why did the chicken cross the road? To get away from this conversation.")
    ],
    'supportive': [
        ("User: I’m feeling down today.", "Bot: I’m sorry to hear that. You’re stronger than you think."),
        ("User: I failed my exam.", "Bot: That’s tough, but failure is a stepping stone to success."),
        ("User: I feel like giving up.", "Bot: Don’t give up! Every step forward, no matter how small, is progress.")
    ]
}

def generate_response(persona, dialog):
    # Retrieve few-shot examples for the selected persona
    examples = few_shot_examples.get(persona, [])
    
    # Format examples into the input query
    few_shot_context = " ".join([f"User: {u} Bot: {b}" for u, b in examples])
    dialog_context = " EOS ".join(dialog)
    
    query = f"{few_shot_context} [NEW DIALOG] {dialog_context}"
    
    input_ids = tokenizer(query, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=128, min_length=8, top_p=0.9, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.route("/", methods=["GET", "POST"])
def index():
    global chat_history
    persona = 'friendly'  # Default persona
    if request.method == "POST":
        user_input = request.form["user_input"]
        persona = request.form["persona"]
        
        # Use the last few exchanges for context
        dialog = [msg['user'] for msg in chat_history[-5:]] + [user_input]
        
        # Generate bot's response
        bot_response = generate_response(persona, dialog)
        
        # Record the conversation
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        chat_history.append({"user": user_input, "bot": bot_response, "timestamp": timestamp})

    return render_template("index.html", chat_history=chat_history, persona=persona)

if __name__ == "__main__":
    app.run(debug=True)
