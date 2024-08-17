import nltk
import random
import string
import requests
import json

from nltk.chat.util import Chat, reflections

# NLTK chatbot responses
responses = [
    (r"hi|hey|hello", ["Hello!", "Hey there!", "Hi!"]),
    (r"how are you?", ["I'm doing well, thank you!", "I'm fine, thanks!"]),
    (r"what is your name?", ["I'm just a chatbot!", "You can call me ChatGPT!"]),
    (r"weather", ["Sorry, I can't check the weather right now. You can try asking me something else!"]),
    (r"bye|goodbye", ["Goodbye!", "Bye!", "See you later!"])
]

# NLTK chatbot
class NLTKChatbot:
    def __init__(self):
        self.chatbot = Chat(responses, reflections)

    def chat(self):
        print("Chatbot: Hi there! How can I help you?")
        while True:
            user_input = input("You: ").lower()
            if user_input == 'quit':
                print("Chatbot: Goodbye!")
                break
            else:
                response = self.chatbot.respond(user_input)
                print("Chatbot:", response)

if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    
    bot = NLTKChatbot()
    bot.chat()

