import random
import json
import pickle
import numpy as np
import tkinter as tk
import pyttsx3
import speech_recognition as sr

import nltk
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))

classes = pickle.load(open('classes.pkl', 'rb'))

model = load_model('chatbotmodel.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word)  for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words= clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda  x:x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list,intents_json):
    tag= intents_list[0]['intent']
    list_of_intents =intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def send_message(event=None):
    message = entry.get()
    if message == "bye" or message == "Goodbye":
        ints = predict_class(message)
        res = get_response(ints, intents)
        messages.insert(tk.END, "You: " + message)
        messages.insert(tk.END, "Bot: " + res)
        messages.insert(tk.END, "The Program Ends here!")
        entry.delete(0, tk.END)
        speak_response(res)
    else:
        ints = predict_class(message)
        res = get_response(ints, intents)
        messages.insert(tk.END, "You: " + message + "\n" + "Bot:" + "\n" + res)
        entry.delete(0, tk.END)
        root.update()  # Update the GUI to display the message before speaking
        speak_response(res)

def speak_response(response):
    engine = pyttsx3.init()
    engine.say(response)
    engine.runAndWait()

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
            print("Recognizing...")
            message = recognizer.recognize_google(audio)
            print("You said:", message)
            entry.delete(0, tk.END)
            entry.insert(0, message)
            send_message()
        except sr.WaitTimeoutError:
            print("Timeout waiting for speech")
            speak_response("I didn't hear anything. Please try again.")
        except sr.UnknownValueError:
            print("Could not understand audio")
            speak_response("I couldn't understand what you said.")
        except sr.RequestError as e:
            print("Error with the request: {0}".format(e))
            speak_response("There was an error with the speech recognition service.")

root = tk.Tk()
root.title("College Inquiry Chatbot System")

messages = tk.Text(root, fg="blue", bg="lightgrey")
messages.pack()

entry = tk.Entry(root, width=50, fg="green")
entry.pack()

send_button = tk.Button(root, text="Send", command=send_message, fg="white", bg="black")
send_button.pack()

voice_button = tk.Button(root, text="Voice", command=recognize_speech, fg="white", bg="blue")
voice_button.pack()

exit_button = tk.Button(root, text="Exit", command=root.destroy, fg="white", bg="red")
exit_button.pack()

root.mainloop()
