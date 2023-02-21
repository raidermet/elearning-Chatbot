#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 11:36:36 2022

@author: rishisingh
"""

#Librairis that have been used for this project

from tkinter import *
import re
from io import StringIO
import requests
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import csv
import nltk
import aiml
import pandas as pd 
import random
import string 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flair.models import TextClassifier
from flair.data import Sentence
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn   
import wikipediaapi
from keybert import KeyBERT
from random_word import RandomWords
import random
from textblob import TextBlob
import pandas as pd
import h5py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5ForConditionalGeneration, T5Tokenizer
r = RandomWords()
##GUI root
root = Tk()

#######################################################
#  Initialise AIML agent
#######################################################

# Create a Kernel object. No string encoding (all I/O is unicode)
kern = aiml.Kernel()
kern.setTextEncoding(None)
# Use the Kernel's bootstrap() method to initialize the Kernel. The
# optional learnFiles argument is a file (or list of files) to load.
# The optional commands argument is a command (or list of commands)
# to run after the fgiles are loaded.
# The optional brainFile argument specifies a brain file to load.
kern.bootstrap(learnFiles="coursework.xml")
#######################################################

#######################################################
# Global variables
#######################################################
answer_l = [] #list of correct answers
question_id = [] #Questions so far
question_incorrect_num = [] #Questions gotten incorrect
custom = False #boolean for customize quiz
df = pd.read_csv("train.tsv",sep='\t')
df1 = df
personalised_questions = []
model_keyword = KeyBERT(model="distilbert-base-nli-mean-tokens") #https://huggingface.co/sentence-transformers/distilbert-base-nli-mean-tokens
create_ans = []
feedback_csv = []


#######################################################
# Driver functions
#######################################################

question_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1') # https://huggingface.co/ramsrigouthamg/t5_squad_v1
question_tokenizer = T5Tokenizer.from_pretrained('t5-base')
def get_question(sentence, answer):
    text = "context: {} answer: {} </ş>".format(sentence, answer)
    #print (text)
    max_len = 256
    encoding = question_tokenizer.encode_plus (text, max_length=max_len, pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
    outs = question_model.generate(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     early_stopping=True,
                                     num_beams=5,
                                     num_return_sequences=1,
                                     no_repeat_ngram_size=2,
                                     max_length=200)
    dec = [question_tokenizer.decode(ids) for ids in outs]
    Question = dec[0].replace("question:",",")
    Question= Question.strip()
    return Question

with h5py.File('/Users/rishisingh/Downloads/AI_N0834113/sentence_embeddings_data.h5', 'r') as hdf:
    ls = list (hdf.keys())
    sentence_embeddings = hdf.get('sentence_embeddings')
    sentence_embeddings = np.array(sentence_embeddings)
    sentence_embeddings

def get_similarity(sentence_embeddings,df,ind):
    c= cosine_similarity([sentence_embeddings[ind]],sentence_embeddings[0:])
    c = pd.DataFrame(c[0].tolist(), columns=['Similarity'])
    result = pd.concat([df, c], axis=1)
    result = result.sort_values(by=['Similarity'], ascending=False)
    quest = result['Question Stem'].iloc[1] 
    e = result['Question Stem'][result['Question Stem'] == quest].index.tolist() 
    return result,e[0]

def question_data():
    answer_l.clear()
    num = random.randint(0, df.shape[0])
    question = df['Complete Question'][num]
    answer = df['Answer Key'][num]
    question_incorrect_num.append(num)
    question_id.append(num)
    answer_l.append(answer)
    
    return question, answer

def custom_question_data(result,quest_no):
    answer_l.clear()
    question = df['Complete Question'][quest_no]
    answer = df['Answer Key'][quest_no]
    question_incorrect_num.append(quest_no)
    question_id.append(quest_no)
    answer_l.append(answer)
    result = result.drop(result.index[1])
    return question, answer

def load_custom(question_incorrect_num,df):
    for i in range(0,len(question_incorrect_num)):
        ind = question_incorrect_num[0] 
        result,e_0 = get_similarity(sentence_embeddings, df, ind)
        result = result[1:4].index
        for i in result:
            personalised_questions.append(i)
            
def cust_question(df,personalised_questions):
    q,a = custom_question_data(df, personalised_questions[-1])
    personalised_questions.pop()
    
    return q,a

def post_data_analysis(feedback_csv):
    df_pda = pd.read_csv('feedback.csv')
    df2 = {'Confidence': feedback_csv[0],'Reccomendation': feedback_csv[1],'Sentiment':feedback_csv[2],'Domain':feedback_csv[3]}
    df_pda = df_pda.append(df2, ignore_index = True)
    df_pda.to_csv('feedback.csv',index=False)
    
    

def change_custom():
    custom = True
    
def reccomendation(incorrect_qs1):
    model = KeyBERT(model="distilbert-base-nli-mean-tokens")
    question_topics = []
    wikipedia_links = []
    for i in incorrect_qs1:
        keyword = model.extract_keywords(df['Question Stem'][i],top_n=-1)
        word = keyword[0]
        question_topics.append(word[0])
        
    return question_topics
    
    
        
    
def clear():  #Clears GUI text
    txt.delete(1.0,END)
    
def flair_prediction(sia, x):
    sentence = Sentence(x)
    sia.predict(sentence)
    score = sentence.labels[0]
    if "POSITIVE" in str(score):
        return "pos"
    elif "NEGATIVE" in str(score):
        return "neg"
    else:
        return "neu"
    
    
def quit(): #To quit GUI
    root.destroy()
    
#######################################################
# Main loop
#######################################################
def send():
    while True: ##Loop
        try:
            userInput = ent.get()
            if userInput != "":
                txt.insert(END, "\n"+ "ME: " + ent.get(), 'warning') #Recieving user input
        except(KeyboardInterrupt, EOFError) as e:
            txt.insert(END,"\n"+"BOT => Bye")
            break
        responseAgent ='aiml'
        res = re.findall(r'\w+', userInput.lower())
        if responseAgent == 'aiml':
            answer = kern.respond(userInput)
        #post-process the answer for commands
        if answer[0] == '#':
            print(answer[0])
            params = answer[1:].split('$')
            cmd = int(params[0])
            if cmd == 0: ##If bye is selected
                txt.insert(END,"\n"+"Bot: "+params[1])
                quit()
                break
            elif cmd == 1:
                if len(question_id) < 5:
                    q,a = question_data()
                    print("This is question number: ", len(question_id))

                    try:
                        txt.insert(END,"\n"+"BOT => " + str(len(question_id))+"." + q)
                        #txt.insert(END,"\n"+"BOT => " + o)
                    except:
                        txt.insert(END,"\n"+"BOT => Sorry, I do not know that. Be more specific!") 
            elif cmd == 2: ## Correct anwer
                user_response = params[1]
                a1=[]
                a1.append(answer_l[0])
                if (userInput == answer_l[0]):
                    txt.insert(END,"\n"+"BOT => That is correct!")
                    question_incorrect_num.pop()

                    if ((len(question_id) < 5) and (custom == False)): 
                        
                        txt.insert(END,"\n"+"BOT => Are you ready for the next question?")
                    else:
                        txt.insert(END,"\n"+"BOT => Awesome! Please enter customize my revision!")
                else:
                    print('The answer is ', answer_l[0], 'the user answer is ', userInput)
                    txt.insert(END,"\n"+"BOT => That is incorrect! The correct answer is " + a1[0])
                    if (len(question_id) < 5): 
                        txt.insert(END,"\n"+"BOT => Are you ready for the next question?")
                    else:
                        txt.insert(END,"\n"+"BOT => Awesome! Please enter customize my revision!")
            elif cmd == 3:
                change_custom()
                if personalised_questions == []:
                    load_custom(question_incorrect_num,df)
                print(personalised_questions)
                q,a = cust_question(df,personalised_questions)
                txt.insert(END,"\n"+"BOT => " + str(len(question_id))+"." + q)
                #else:
                   # q,a = cust_question(df,personalised_questions)
                    #txt.insert(END,"\n"+"BOT => " + str(len(question_id))+"." + q)
            elif cmd ==4:
                if len(question_incorrect_num) == 0:
                    txt.insert(END,"\n"+"BOT => Wow you know your stuff well done!")
                    txt.insert(END,"\n"+"BOT => Please give some feedback if you didnt like any questions please include them in your feedback.")
                    txt.insert(END,"\n"+"BOT => Type 'Feedback' - to get started.")
                else:
                    qt = reccomendation(question_incorrect_num)
                    print(qt)
                    wikipedia_links = []
                    for i in qt:
                        wiki_wiki = wikipediaapi.Wikipedia('en') 
                        w_1 = wiki_wiki.page(i)
                        wikipedia_links.append(w_1.fullurl)
                    print(wikipedia_links)
                    #links_wiki = reccomendation(question_incorrect_num)
                    #print(links_wiki)
                    txt.insert(END,"\n"+"BOT => I recccomend you revise some content related to: "+qt[0]+ " and " +qt[1]+". Have a look at the following links: \n"+ str(wikipedia_links[0]) + "\n" + str(wikipedia_links[1]))
                    txt.insert(END,"\n"+"BOT => Please give some feedback: Out of 5 rate your confidence prior to the quiz.")
                    txt.insert(END,"\n"+"BOT => Type 'Confidence' and then your rating.")
            elif cmd == 6:
                word = params[1]
                print('this is the word ' , word)
                wiki_wiki = wikipediaapi.Wikipedia('en')
                word = wiki_wiki.page(word)
                word_sum = word.summary[0:1000]
                answer = model_keyword.extract_keywords(word_sum)[1]
                answer = answer[0]
                
                sentence_for_T5 = word_sum.replace("**"," ")
                sentence_for_T5 = " ".join(sentence_for_T5.split())
                ques = get_question(sentence_for_T5, answer)
                print("*********")
                print(ques)
                ques = ques[8:(len(ques)-4)]
                txt.insert(END, "\n" + "Bot:" + ques, 'warning')
                answers_4 = []
                for i in range(0,3):
                    answers_4.append(r.get_random_word())
                answers_4.append(answer)
                random.shuffle(answers_4)
                indx_ans = answers_4.index(answer)
                create_ans.append(indx_ans + 1)
                txt.insert(END, "\n" + "Bot: 1."+ answers_4[0] +" 2." + answers_4[1]+" 3. "+ answers_4[2]+" 4. " + answers_4[3]) 
            elif cmd == 7:
                if userInput == str(create_ans[0]):
                    txt.insert(END,"\n"+"BOT => That is correct!")
                else:
                    txt.insert(END,"\n"+"BOT => That is incorrect! The correct answer is " + str(create_ans[0]))
                create_ans.clear()
            elif cmd ==8:
                conf_rating = params[1]
                print(' this is your conf rating', str(params[1]))
                feedback_csv.append(conf_rating)
                txt.insert(END,"\n"+"BOT => Please give some feedback: Out of 5 how would you rate the reccomendation?")
                txt.insert(END,"\n"+"BOT => Type 'Reccomendation' and then your rating")
            elif cmd == 9:
                rec_rating = params[1]
                feedback_csv.append(rec_rating)
                txt.insert(END,"\n"+"BOT => Please give some feedback how did you feel about the quiz")
                txt.insert(END,"\n"+"BOT => Type 'Feedback' and then your topics")
            elif cmd ==10:
                feedback_1 = params[1]
                print("this is feedback - "+ str(feedback_1))
                sia = TextClassifier.load('en-sentiment')
                sent = flair_prediction(sia,feedback_1)
                feedback_csv.append(sent)
                txt.insert(END,"\n"+"BOT => Please give some feedback if would like any specific topics.")
                txt.insert(END,"\n"+"BOT => Type 'Future topics' - to get started.")
            elif cmd==11:
                fut_top = params[1]
                feedback_csv.append(fut_top)
                post_data_analysis(feedback_csv)
                txt.insert(END,"\n"+"BOT => Thank you for using AITutor.")
            elif cmd == 99: #Accessing CSV file
                user_response = params[1] 
                new_response = user_response.lower() 
                #new_response = gen_response(user_response)
                if (new_response != " "):
                    txt.insert(END, "\n" + "Bot: Sorry I didnt get that, please try again ", 'warning')
                else:
                    txt.insert(END, "\n" + "Bot: Sorry I did not get that, please try again.") ##Error message when content is not in CSV file
        else: 
            txt.insert(END,"\n Bot: " + answer)
        ent.delete(0,END) #deletes text in entry box

##GUI build up
root.resizable(False,False)
txt = Text(root, bd=1, bg='White', fg='Black',width = 110, height = 25)
txt.grid(row=0,column=0,columnspan=3)
txt.tag_config('warning', foreground="red")
txt.insert(END,"Welcome I am your personalised AITutor. I will be tailoring your revision dependent on the questions you get incorrect!")
ent = Entry(root,width=50,fg='red',)
send=Button(root,text="Send",command=send).grid(row=1,column=1)
clear=Button(root,text="Clear",command=clear).grid(row=1,column=2)
ent.grid(row=1,column=0)
root.title("AI TUTOR CHATBOT")
root.mainloop()