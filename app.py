from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")


app = Flask(__name__)

# Store the chat history in a global variable
chat_history = {}


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["POST"])
def chat():
    user_msg = request.form["msg"]
    user_id = request.remote_addr  # Using the user's IP address as a unique identifier
    response = get_Chat_response(user_msg, user_id)
    return jsonify(response)

def get_Chat_response(user_msg, user_id):
    global chat_history

  # Initialize chat history for new users
    if user_id not in chat_history:
        chat_history[user_id] = None

# Encode the new user input, add the eos_token, and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(user_msg + tokenizer.eos_token, return_tensors='pt')

    # Append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history[user_id], new_user_input_ids], dim=-1) if chat_history[user_id] is not None else new_user_input_ids

 # Create an attention mask
    attention_mask = torch.ones(bot_input_ids.shape, dtype=torch.long)

    # Generate a response while limiting the total chat history to 1000 tokens
    chat_history[user_id] = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode the last output tokens from the bot
    bot_response = tokenizer.decode(chat_history[user_id][:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    return bot_response

if __name__ == '__main__':
    app.run(debug=True)