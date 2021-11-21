import os
import sys
import re
import json
import codecs
import copy
from optparse import OptionParser
from collections import Counter 
import numpy as np
import torch
from scipy import sparse
from scipy.io import savemat
from spacy.lang.en import English 
import pandas as pd
import time
import matplotlib.pyplot as plt
from gensim.parsing.preprocessing import remove_stopwords
import tokenizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize


def read_text(input_filename):
    with codecs.open(input_filename, 'r', encoding='utf-8') as input_file:
        lines = input_file.readlines()
    return lines

def load_and_process_data(infile):
    
    lines = read_text(infile)
    n_items = len(lines)
    print("Parsing %d documents"%n_items)

    check_list = [
    'Walk-in Clinics',
    'Surgeons',
    'Oncologist',
    'Cardiologists',
    'Hospitals',
    'Internal Medicine',
    'Assisted Living Facilities',
    'Cannabis Dispensaries',
    'Doctors',
    'Home Health Care',
    'Health Coach',
    'Emergency Pet Hospital',
    'Pharmacy',
    'Sleep Specialists',
    'Professional Services',
    'Addiction Medicine',
    'Weight Loss Centers',
    'Pediatric Dentists',
    'Cosmetic Surgeons',
    'Nephrologists',
    'Naturopathic/Holistic',
    'Pediatricians',
    'Nurse Practitioner',
    'Urgent Care',
    'Orthopedists',
    'Drugstores',
    'Optometrists',
    'Rehabilitation Center',
    'Hypnosis/Hypnotherapy',
    'Physical Therapy',
    'Neurologist',
    'Memory Care',
    'Allergists',
    'Counseling & Mental Health',
    'Pet Groomers',
    'Podiatrists',
    'Dermatologists',
    'Diagnostic Services',
    'Radiologists',
    'Medical Centers',
    'Gastroenterologist',
    'Obstetricians & Gynecologists',
    'Pulmonologist',
    'Ear Nose & Throat',
    'Ophthalmologists',
    'Sports Medicine',
    'Nutritionists',
    'Psychiatrists',
    'Vascular Medicine',
    'Cannabis Clinics',
    'Hospice',
    'First Aid Classes',
    'Medical Spas',
    'Spine Surgeons',
    'Health Retreats',
    'Medical Transportation',
    'Dentists',
    'Health & Medical',
    'Speech Therapists',
    'Emergency Medicine',
    'Chiropractors',
    'Medical Supplies',
    'General Dentistry',
    'Occupational Therapy',
    'Urologists',
    ]
    i = 0
    id_list = []
    for line in lines:
        i += 1
        if i % 1000 == 0 and i > 0:
            print(i)

        obj = json.loads(line)
        if(obj["categories"]):
            if any((re.match(x, obj["categories"])) for x in check_list):
                #print(obj["business_id"])
                id_list.append(obj["business_id"])
    return id_list

def load_and_process_text(infile, id_list):
    
    lines = read_text(infile)
    n_items = len(lines)
    #print("Parsing %d documents"%n_items)

    i = 0
    f= open("medical.txt","w+")
    dict = {}
    for x in id_list:\
        dict[x] = 1
    for line in lines:
        i += 1
        if i % 1000 == 0 and i > 0:
            print(i)

        obj = json.loads(line)
        if obj["business_id"] in id_list and (int(obj["stars"]) == 1 or  int(obj["stars"]) == 5):
            sent_tokenize_list = sent_tokenize(obj["text"])
            num = len(sent_tokenize_list)
            count = 0
            label = '1'
            if(int(obj["stars"]) == 1 ):
                label = '1'
            if(int(obj["stars"]) == 5 ):
                label = '2'
            text = obj["user_id"]+"\t\t"+obj["business_id"]+"\t\t"+label+"\t\t"
            for sent in sent_tokenize_list:
                count += 1
                words = word_tokenize(sent)
                for word in words:
                    text += word + " "
                if(count < num):
                    text += "<sssss> "
            text += "\n"
            f.write(text)
    return id_list


def preprocess_data(attribute_infile,text_infile):
    print("Loading Spacy")
    parser = English()
    
    id_list = load_and_process_data(attribute_infile)
    
    text = load_and_process_text(text_infile,id_list)
    
def main():
    attribute_infile = 'business.json'
    text_infile = 'review.json'

    preprocess_data(attribute_infile,text_infile)
    
if __name__ == '__main__':
    main()