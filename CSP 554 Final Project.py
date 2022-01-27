#!/usr/bin/env python
# coding: utf-8

# In[26]:


#Importing Library and setting environment path
import os
import sys
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt


# In[28]:





# In[27]:


import findspark

import pyspark
import pandas as pd


# In[5]:


from pyspark import SparkContext
from pyspark import SparkConf
from pyspark import SQLContext
from pyspark.sql.functions import udf
from pyspark.sql.types import *
from pyspark.sql.types import DoubleType
#Importing data set and making a SparkSQLDataFrame
sc=SparkContext.getOrCreate();
sqlContext = SQLContext(sc)

#convert original data csv file to tab limited txt file and delete the header row. Some text reviews contain comma.
dataRDD = sc.textFile('hdfs:///user/hadoop/Reviews.csv')
#dataRDD.take(5)


# In[6]:


dataRDD.count()   # count of rows


# In[7]:


df1 =sqlContext.read.format('com.databricks.spark.csv').options(header='true',inferschema='true').load(r"hdfs:///user/hadoop/Reviews.csv")


# In[8]:


print(df1)


# In[9]:


df1.count()


# In[10]:


#Filtering out records with rating = 3 as they represent neutral score.
#val notFollowingList=Array("2","3","4")
#messages = data.filter(col('Score').isin(notFollowingList:_*)).select('Score','Text')
#messages = messages.withColumn("Score", messages["Score"].cast(DoubleType()))
messages = df1.filter((df1.Score == "1") | (df1.Score=="5")).select('Score','Text')
messages.show(10)


# In[11]:


import re
#Lower casing the text and keeping only aplhabets
def lower_text(line):
    word_list=re.findall('[\w_]+', line.lower())
    return ' '.join(map(str, word_list))

udflower_text=udf(lower_text, StringType())
messages_lower = messages.withColumn("text_lower", udflower_text("Text")).select('text_lower','Score')

#Showing the result
messages_lower.show(15)


# In[12]:


from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import NGram
import matplotlib
matplotlib.style.use('ggplot')


# In[13]:


#Tokenizing the document using in-built library from pyspark.ml
tokenizer = Tokenizer(inputCol="text_lower", outputCol="words")
wordsDataFrame = tokenizer.transform(messages_lower)
wordsDataFrame.take(5)


# In[14]:


# Remove stopwords using in-built library from pyspark.ml
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
wordsDataFrame1 = remover.transform(wordsDataFrame).select("filtered_words","Score")
wordsDataFrame1.show(5)


# In[15]:


import re
#Lower casing the text and keeping only aplhabets
def wordjoin(line):
    #word_list=re.findall('[\w_]+', line.lower())
    return ' '.join(map(str, line))

udf_wordjoin=udf(wordjoin, StringType())
wordDataFrame2 = wordsDataFrame1.withColumn("words", udf_wordjoin("filtered_words")).select('words','Score')
wordDataFrame2.show(5)
wordDataFrame2.printSchema()


# In[16]:


#Naive Bayes Classifier

training = wordDataFrame2.selectExpr("words as text", "Score as label")
training = training.withColumn("label", training["label"].cast(DoubleType()))
training.take(2)


# In[17]:


#Creating pipeline for Tokenizing, TF - IDF and Naiave Bayes Model
from pyspark.ml.feature import HashingTF, IDF
from pyspark.ml.classification import NaiveBayes, LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
#from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="hashing")
idf = IDF(inputCol=hashingTF.getOutputCol(), outputCol="features")
lr = LogisticRegression(maxIter=10, regParam=0.01)
pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, lr])
# Training the model
#model = pipeline.fit(training)


# In[18]:


import re
#Lower casing the text and keeping only aplhabets
def wordjoin(line):
    #word_list=re.findall('[\w_]+', line.lower())
    return ' '.join(map(str, line))

udf_wordjoin=udf(wordjoin, StringType())
wordDataFrame = wordsDataFrame1.withColumn("words", udf_wordjoin("filtered_words")).select('words','Score')
wordDataFrame.show(5)
wordDataFrame.printSchema()


# In[19]:


#Cache the RDD
wordsDataFrame1.cache()


# In[20]:


wordclouddata = wordsDataFrame1.rdd.map(lambda x: (x[1],x[0])).toDF()
#print(wordclouddata)
wordclouddata = wordclouddata.selectExpr("_1 as Score","_2 as word")
wordclouddata.printSchema()
wordclouddata.show(5)
wordclouddata.createOrReplaceTempView("words")


# In[21]:


from collections import defaultdict, Counter
import operator
pos_wordslist = sqlContext.sql(" SELECT word from words where Score = 5 ").take(1000)   # select most positive words
pos_words=''
pos_feats=defaultdict(lambda:0)
for i in pos_wordslist:
    for word in i[0]:
        if word.isalpha() and word!='br':
            pos_words+=word+' ' 
            pos_feats[word]+=1
        

sorted_pos_feats = dict(sorted(pos_feats.items(), key=operator.itemgetter(1),reverse=True))
sorted_pos_feats


# In[22]:


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color='white',
                          width=3000,
                          height=1000
                         ).generate(pos_words)


plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[23]:


neg_wordslist = sqlContext.sql(" SELECT word from words where Score = 1 ").take(1000)   # select most negative words
neg_words=''
neg_feats=defaultdict(lambda:0)
for i in neg_wordslist:
    for word in i[0]:
        if word.isalpha() and word!='br':
            neg_words+=word+' ' 
            neg_feats[word]+=1
        

sorted_neg_feats = dict(sorted(neg_feats.items(), key=operator.itemgetter(1),reverse=True))
sorted_neg_feats


# In[24]:


wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color='white',
                          width=1200,
                          height=1000
                         ).generate(neg_words)


plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[25]:


selected_neg_words = set(['bad', 'hate', 'horrible', 'terrible','worst', 'dislike','disappointed','disappointing','never','waste','awful'])
selected_pos_words = set(['awesome', 'amazing', 'best', 'good', 'great', 'love', 'loves','like','wonderful','tasty','fresh','organic','happy'])
neg_word_count,pos_word_count=0,0
for word in pos_feats.keys():
    if word in selected_pos_words:
        pos_word_count+=1
for word in neg_feats.keys():
    if word in selected_neg_words:
        neg_word_count+=1
print("pos_word_count=",pos_word_count," neg_word_count=",neg_word_count)
print("bad_count=", neg_feats['bad'], " great_count=",pos_feats['great'])


# In[ ]:




