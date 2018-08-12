# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 08:25:00 2018

@author: Gopi
"""

import pandas as pd
#Read the csv file
data = pd.read_csv('D:/emaildsml@gmail.com/R Scripts/Train_Mockup.csv',encoding='cp1252')
data['Problem_Description'].count()
#data = pd.read_csv('D:/K-Means/Test.csv',encoding='cp1252')
#Text Pre-Processing
'''All tweets are processed to remove unnecessary things like 
links, non-English words, stopwords, punctuation’s, etc.'''
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import re, string
import nltk
#nltk.download('stopwords')
#nltk.download('words')
#Convert to Lower-case
data["Problem_Description"] = data["Problem_Description"].apply(lambda x: " ".join(x.lower() for x in x.split()))
#Remove Numbers
data["Problem_Description"] = data["Problem_Description"].apply(lambda x: ''.join(i for i in x if not i.isdigit()))
#Remove Punctuation
data["Problem_Description"] = data["Problem_Description"].str.replace('[^\w\s]','')
#Remove stopwords
from nltk.corpus import stopwords
stop = stopwords.words('english')
data["Problem_Description"] = data["Problem_Description"].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
#Lemmitization
from textblob import Word
data["Problem_Description"] = data["Problem_Description"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

data["Problem_Description"].head()

from sklearn.feature_extraction.text import CountVectorizer
# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(data["Problem_Description"])
# transform the training and validation data using count vectorizer object
Count_Vector =  count_vect.transform(data["Problem_Description"])

#Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer  
# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(data["Problem_Description"])
tfidf_word =  tfidf_vect.transform(data["Problem_Description"])

# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(data["Problem_Description"])
tfidf_ngram =  tfidf_vect_ngram.transform(data["Problem_Description"])

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(data["Problem_Description"])
tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(data["Problem_Description"]) 

##Dynamic retrival of no of clusters
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
best_clusters = 0 # best cluster number which you will get
previous_silh_avg = 0.0
for n_clusters in range(2, 10):
    clusterer = KMeans(n_clusters, init='k-means++', random_state=1)
    cluster_labels = clusterer.fit_predict(Count_Vector)
    silhouette_avg = silhouette_score(Count_Vector, cluster_labels, sample_size=3000)
    if silhouette_avg > previous_silh_avg:
        previous_silh_avg = silhouette_avg
        best_clusters = n_clusters
print("No of Clusters:", best_clusters)
##K-Means Implementation
num_clusters = best_clusters  
km = KMeans(n_clusters=num_clusters, init='k-means++', random_state=1)  
km.fit(Count_Vector)  
clusters = km.labels_.tolist()  
data['ClusterID'] = clusters  
print("Cluster Count")
print(data['ClusterID'].value_counts())

'''
#Elbow Analysis
cluster_range = range( 1, 20 )
cluster_errors = []

for num_clusters in cluster_range:
  clusters = KMeans(num_clusters)
  clusters.fit( X )
  cluster_errors.append( clusters.inertia_ )
clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )
clusters_df[0:10]
import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )'''

#data.to_csv('D:/K-Means/K-means.csv')
#The top words used in each cluster can be computed by as follows:
#sort cluster centers by proximity to centroid

order_centroids = km.cluster_centers_.argsort()[:, ::-1]
Clusters =pd.DataFrame()      
for i in range(num_clusters):
    Clusters.at[int(i), 'No'] = int(i)
    Clusters.at[i, 'Words'] = (','.join([feature_names[x] for x in order_centroids[i, :10]]))
    #a.No.astype(int)   
Clusters['No'] = Clusters.No.astype(int)

#Subset the data per cluster& generate Four,Tri & Bi-Grams
# Define column names
colNames = ('Frequency','Word','Cluster_No','ClusterName','TopWords', 'Incident Count')
# Define a dataframe with the required column names
masterDF = pd.DataFrame(columns = colNames)
for i in Clusters['No']:
    N_G = data.loc[data['ClusterID'] == i]
    try:
        word_vectorizer = CountVectorizer(ngram_range=(4,4), analyzer='word')
        sparse_matrix = word_vectorizer.fit_transform(N_G["Problem_Description"])
    except ValueError:
        try:
            word_vectorizer = CountVectorizer(ngram_range=(3,3), analyzer='word')
            sparse_matrix = word_vectorizer.fit_transform(N_G["Problem_Description"])
        except ValueError:
            word_vectorizer = CountVectorizer(ngram_range=(2,2), analyzer='word')
            sparse_matrix = word_vectorizer.fit_transform(N_G["Problem_Description"])
    frequencies = sum(sparse_matrix).toarray()[0]
    NGram = pd.DataFrame(frequencies, index=word_vectorizer.get_feature_names(), columns=['Frequency'])
    NGram['Word'] = NGram.index
    NGram.reset_index(drop = True, inplace = True)
    NGram = NGram.sort_values('Frequency', ascending=[False])
    NGram['Cluster_No'] = i
    NGram['ClusterName'] = NGram['Word'].head(1).to_string(index=False) 
    NGram['TopWords'] = ",".join(NGram['Word'].head(10).to_string(header=False,
                  index=False).split('\n')[1:10])
    NGram['Incident Count'] = len(data.loc[data['ClusterID']==i])
    NGram = NGram.iloc[0]
    # Try to append temporary DF to master DF
    masterDF = masterDF.append(NGram,ignore_index=True)
masterDF = masterDF.drop(['Frequency', 'Word'], axis=1)
#masterDF.to_csv('D:/K-Means/Master.csv')
data['ClusterName'] = data['ClusterID'].map(masterDF['ClusterName'])


    
#Tri-Gram Generation



#Multi Dimensional Scaling
from sklearn.metrics.pairwise import cosine_similarity  
dist = 1 - cosine_similarity(tfidf_matrix)  
#print(dist) 
import os  # for os.path.basename
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import MDS

MDS()

# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
xs, ys = pos[:, 0], pos[:, 1]
     
    
    
    
#Plotting    
#set up colors per clusters using a dict
cluster_colors = {0: '#4286f4', 1: '#41f4a9', 2: '#41f4f4', 3: '#41f4f4'}

#set up cluster names using a dict
cluster_names = list(Clusters['Words'].astype(str).str[0:10])
#some ipython magic to show the matplotlib plots inline
%matplotlib inline 

#create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title = data['Title'])) 

#group by cluster
groups = df.groupby('label')

# set up plot
fig, ax = plt.subplots(figsize=(17, 9)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
            label=cluster_names[name], color=cluster_colors[name], 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')
    
ax.legend(numpoints=1)  #show legend with only 1 point

#add label in x,y position with the label as the film title
for i in range(len(df)):
    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)  

plt.show() #show the plot
