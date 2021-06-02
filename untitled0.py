# -*- coding: utf-8 -*-
"""
Created on Mon May  3 11:11:19 2021

@author: nobin
"""
import os
from google.cloud import vision
import tensorflow_hub as hub
import numpy as np
import seaborn as sns
def plot_similarity(labels, features, rotation):
  corr = np.inner(features, features)
  sns.set(font_scale=1.2)
  g = sns.heatmap(
      corr,
      xticklabels=labels,
      yticklabels=labels,
      vmin=0,
      vmax=1,
      cmap="YlOrRd")
  g.set_xticklabels(labels, rotation=rotation)
  g.set_title("Semantic Textual Similarity")

def run_and_plot(messages_):
  message_embeddings_ = embed(messages_)
  plot_similarity(messages_, message_embeddings_, 90)
answer_key=["Object-oriented programming (OOP) is a programming paradigm based on the concept of objects, which can contain data and code: data in the form of fields (often known as attributes or properties), and code, in the form of procedures (often known as methods).A feature of objects is that an object's own procedures can access and often modify the data fields of itself (objects have a notion of this or self). In OOP, computer programs are designed by making them out of objects that interact with one another.[OOP languages are diverse, but the most popular ones are class-based, meaning that objects are instances of classes, which also determine their types.Many of the most widely used programming languages (such as C++, Java, Python, etc) are multi-paradigm and they support object-oriented programming to a greater or lesser degree",
            "Uses of Operating System.The operating system is used everywhere today, such as banks, schools, hospitals, companies, mobiles, etc. No device can operate without an operating system because it controls all the user's commands.The operating system has many notable features that are developing day by day. The growth of the operating system is commendable as it was developed in 1950 to handle storage tape. It acts as an interface. The features of operating system are given below.Error detection and handling. Handling I/O operations. Virtual Memory Multitasking. Program Execution. Allows disk access and file systems. Memory management. Protected and supervisor mode. Security. Resource allocation. Easy to run. Information and Resource Protection. Manipulation of the file system",
            "google.com",
            "An array is a collection of items stored at contiguous memory locations. The idea is to store multiple items of the same type together. This makes it easier to calculate the position of each element by simply adding an offset to a base value, i.e., the memory location of the first element of the array (generally denoted by the name of the array). The base value is index 0 and the difference between the two indexes is the offset",
            "Define a,b,c. Let a=b+c.Print c"]
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=r"nobin-312606-4f15bfe641d8.json"
client=vision.ImageAnnotatorClient()
answer_key.append(str(client.text_detection(image= vision.types.Image(content=open("Student1/1.jpg",'rb').read())).text_annotations[0].description.replace("\n"," ")))
answer_key.append(str(client.text_detection(image= vision.types.Image(content=open("Student1/2.jpg",'rb').read())).text_annotations[0].description.replace("\n"," ")))
answer_key.append(str(client.text_detection(image= vision.types.Image(content=open("Student1/3.jpg",'rb').read())).text_annotations[0].description.replace("\n"," ")))
answer_key.append(str(client.text_detection(image= vision.types.Image(content=open("Student1/4.jpg",'rb').read())).text_annotations[0].description.replace("\n"," ")))
answer_key.append(str(client.text_detection(image= vision.types.Image(content=open("Student1/5.jpg",'rb').read())).text_annotations[0].description.replace("\n"," ")))
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
marks=(4,3,1,2,2)
run_and_plot(answer_key)
