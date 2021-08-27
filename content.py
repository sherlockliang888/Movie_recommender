import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import csv
import random

smd = pd.read_csv(r"data\description_based.csv", low_memory=False)

smd['tagline'] = smd['tagline'].fillna('')
smd['description'] = smd['overview'] + smd['tagline']
smd['description'] = smd['description'].fillna('')



tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(smd['description'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])

def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]


NewMovies=[]
with open('data//Usermovie.csv','r') as csvfile:
    readCSV = csv.reader(csvfile)
    NewMovies.append(random.choice(list(readCSV)))
NewMovies
m_name = NewMovies[0][0]
m_name = m_name.title()
m_name

def get_suggestions():
    data = pd.read_csv('data//tmdb.csv')
    return list(data['title'].str.capitalize())

data = pd.read_csv('data//tmdb.csv')
list(data['title'].str.capitalize())

with open('data//Usermovie.csv', 'a', newline='') as csv_file:
    fieldnames = ['Movie']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writerow({'Movie': "what is lasy next"})

