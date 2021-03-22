
import tensorflow as tf
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import re
import unicodedata
import nltk
from nltk.corpus import stopwords
import keras
from tqdm import tqdm
import pickle
#from keras.models import Model
#import keras.backend as K
from sklearn.metrics import confusion_matrix,f1_score,classification_report
#import matplotlib.pyplot as plt
#from keras.callbacks import ModelCheckpoint
import itertools
#from keras.models import load_model
from sklearn.utils import shuffle
from transformers import *
from transformers import BertTokenizer, TFBertModel, BertConfig
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def clean_stopwords_shortwords(w):
    stopwords_list=stopwords.words('english')
    words = w.split() 
    clean_words = [word for word in words if (word not in stopwords_list) and len(word) > 2]
    return " ".join(clean_words) 

def preprocess_sentence(w):
    w = re.sub(r"([?.！,¿])", r" ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r'["。，"]+', "", w)
    w=clean_stopwords_shortwords(w)
    w=re.sub(r'@\w+', '',w)
    return w
'''
data_file='./test.csv'

data=pd.read_csv(data_file,encoding='utf-8')
data.columns=['label', 'text']
print(data.head())

print('File has {} rows and {} columns'.format(data.shape[0],data.shape[1]))

data=data.dropna()                           # Drop NaN valuues, if any
data=data.reset_index(drop=True)                    # Reset index after dropping the columns/rows with NaN values
#data = shuffle(data)                         # Shuffle the dataset
print('Available labels: ',data.label.unique())            # Print all the unique labels in the dataset
data['text']=data['text'].map(preprocess_sentence)           # Clean the text column using preprocess_sentence function defined above

print('File has {} rows and {} columns'.format(data.shape[0],data.shape[1]))
print(data.head())
'''
def dataToCsv(data_list,s):
  with open('BertData.csv', 'w', newline='') as csvfile:
            # 建立 CSV 檔寫入器
            writer = csv.writer(csvfile)
            # 寫入一列資料
            writer.writerow(["text"])
  for i in range(len(data_list[0])):
        with open(s, 'a', newline='') as csvfile:
            # 建立 CSV 檔寫入器
            writer = csv.writer(csvfile)
            # 寫入一列資料
            writer.writerow([data_list[0][i]])

#num_classes=len(data.label.unique())
def BERT_Predict(s):
  bert_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
  bert_model = TFBertForSequenceClassification.from_pretrained("bert-base-chinese",num_labels=2)


  dataf=pd.read_csv(s,encoding='utf-8')
  #dataf['text']=dataf['text'].map(preprocess_sentence)  
  Fdata=dataf['text']

  #sentences=data['text']
  #labels=data['label']

  input_ids=[]
  attention_masks=[]

  for sent in Fdata:
      bert_inp=bert_tokenizer.encode_plus(sent,add_special_tokens = True,max_length =64,pad_to_max_length = True,return_attention_mask = True)
      input_ids.append(bert_inp['input_ids'])
      attention_masks.append(bert_inp['attention_mask'])

  input_ids=np.asarray(input_ids)
  attention_masks=np.array(attention_masks)
  #labels=np.array(labels)

  #train_inp,val_inp,train_label,val_label,train_mask,val_mask=train_test_split(input_ids,labels,attention_masks,test_size=0.2)

  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
  optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5,epsilon=1e-08)
  model_save_path='./bert_model.h5'

  trained_model = TFBertForSequenceClassification.from_pretrained('bert-base-chinese',num_labels=2)
  trained_model.compile(loss=loss,optimizer=optimizer, metrics=[metric])
  trained_model.load_weights(model_save_path)

  #trained_model = keras.models.load_model("saved_model/my_model")

  bert_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")


  preds = trained_model.predict([input_ids,attention_masks],batch_size=32)

  print(preds)#bert結果
  tf_predictions = tf.nn.softmax(preds[0], axis=-1)
  labels = ['Negative','Positive']
  label = tf.argmax(tf_predictions, axis=1)
  label = label.numpy()
  print("BERT_Predict:")
  for i in range(len(Fdata)):
    print(Fdata[i], ": \n", labels[label[i]])

#print(bert_tokenizer.decode(val_inp[0]))


'''
pred_labels = preds.argmax(axis=1)
f1 = f1_score(val_label,pred_labels)
print('F1 score',f1)
print('Classification Report')
print(classification_report(val_label,pred_labels,target_names=target_names))

print('Training and saving built model.....')
'''

'''
pred_sentences = ['等了兩個小時菜的味道不錯但是太慢了打了好幾個電話一直佔線',
          '方便快捷','非常快態度好','遲到40分鐘']
tf_batch = bert_tokenizer(pred_sentences, max_length=128, padding=True, truncation=True, return_tensors='tf')
#print("WTF",tf_batch)#id token mask 
tf_outputs = trained_model(tf_batch)
print("WTF",tf_outputs)#bert結果
tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
#print("WTF",tf_predictions)#0 1之間
labels = ['Negative','Positive']
label = tf.argmax(tf_predictions, axis=1)
label = label.numpy()
for i in range(len(pred_sentences)):
  print(pred_sentences[i], ": \n", labels[label[i]])
'''


