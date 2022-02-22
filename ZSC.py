import streamlit as st
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import json_normalize
import os, gc, warnings
import torch
from torch import layout
warnings.filterwarnings("ignore")
import json
import gc
gc.collect()
model = "facebook/bart-large-mnli" 
torch.cuda.empty_cache()
import transformers
from transformers import pipeline


st.set_page_config(page_title="Classifier", page_icon=None, initial_sidebar_state="expanded", menu_items=None, layout="wide")

st.title("Text Classifier")
uploaded_file=st.sidebar.file_uploader("")

labels = st.sidebar.text_input("Categories")

def check_password():
    password = st.sidebar.text_input("Password", type="password")

    if password:
        if password == "st":
            return True
        else:
            st.sidebar.error("Error")
            return False

if check_password():

    if uploaded_file is not None:

        df=pd.read_csv(uploaded_file)
        df1 = df
        batch_size = 10 # see how big you can make this number before OOM
        classifier = pipeline('zero-shot-classification', model=model) # to utilize GPU
        sequences = df['Comments'].to_list()
        
        results = []
        for i in range(0, len(sequences), batch_size):
            results += classifier(sequences[i:i+batch_size], labels, multi_label=False, device=1)
        
        
        df = json_normalize(results)
        d = [pd.DataFrame(df[col].tolist()).add_prefix(col) for col in df.columns]
        df = pd.concat(d, axis=1)
        df.rename(columns = {'sequence0':'Comment'}, inplace = True)
        #df=pd.concat([df1,df], axis = 1)
        st.write(df)
