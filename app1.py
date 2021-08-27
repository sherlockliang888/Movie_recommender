import numpy as np
from flask import Flask, request, render_template
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import csv
import random
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

os.environ['FLASK_ENV'] = 'development'

smd = pd.read_csv("data\\meta_all_based.csv", low_memory=False)
smd['description'] = smd['description'].fillna('')


tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(smd['description'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_matrix = count.fit_transform(smd['soup'])
cosine_sim1 = cosine_similarity(count_matrix, count_matrix)

smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])

all_titles = [smd['title'][i] for i in range(len(smd['title']))]


reader = Reader()
ratings = pd.read_csv('data/ratings_small.csv')
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
svd = SVD()
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5)
trainset = data.build_full_trainset()



def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan

id_map = pd.read_csv('data/links_small.csv')[['movieId', 'tmdbId']]
id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
id_map.columns = ['movieId', 'id']
id_map = id_map.merge(smd[['title', 'id']], on='id').set_index('title')
#id_map = id_map.set_index('tmdbId')
indices_map = id_map.set_index('id')




def get_recommendations(userId, title):
    idx = indices[title]
    tmdbId = id_map.loc[title]['id']
    movie_id = id_map.loc[title]['movieId']

    sim_scores = list(enumerate(cosine_sim1[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]

    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]
    movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est)
    movies = movies.sort_values('est', ascending=False)

    #tit = titles.iloc[movie_indices]
    #return_df = pd.DataFrame(columns=['Title'])
    #return_df['Title'] = tit
    #return return_df.Title.values.tolist()
    return movies.title.values.tolist()

#get_recommendations(1, "Iron Man")



app = Flask(__name__)

@app.route('/')
def start():
    return render_template('start.html')


@app.route('/home', methods=['GET', 'POST'])
def home():
    userID = int(float(request.form.get('userID')))
    NewMovies=[]
    with open('data//Usermovie.csv','r') as csvfile:
        readCSV = csv.reader(csvfile)
        NewMovies.append(random.choice(list(readCSV)))
    m_name = NewMovies[0][0]
    m_name = m_name.title()
    results = get_recommendations(userID, m_name)[:9]
    home_result = str(results)

    return render_template('index.html', userID = userID, home_result=home_result, im1 = results[0], 
    im2 = results[1], im3 = results[2], im4 = results[3], im5 = results[4], im6 = results[5],
    im7 = results[6], im8 = results[7], im9 = results[8])



@app.route('/test', methods=['GET', 'POST'])
def test():
    userID = int(float(request.form.get('userID')))
    moviename = request.form.get('moviename')
    if moviename not in all_titles:
        NewMovies=[]
        with open('data//Usermovie.csv','r') as csvfile:
            readCSV = csv.reader(csvfile)
            NewMovies.append(random.choice(list(readCSV)))
        m_name = NewMovies[0][0]
        m_name = m_name.title()
        results = get_recommendations(userID, m_name)[:9]
        home_result = str(results)
        

        return render_template('index_neg.html',userID = userID, moviename = moviename, home_result=home_result,  im1 = results[0], 
            im2 = results[1], im3 = results[2], im4 = results[3], im5 = results[4], im6 = results[5],
            im7 = results[6], im8 = results[7], im9 = results[8])
    else:
        try: 
            with open('data//Usermovie.csv', 'a', newline='') as csv_file:
                fieldnames = ['Movie']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writerow({'Movie': moviename})
    
            results = get_recommendations(userID, moviename)[:6]
            result = str(results)
            return render_template('result.html',userID = userID,   result = result, im1 = results[0], 
                im2 = results[1], im3 = results[2], im4 = results[3], im5 = results[4], im6 = results[5])
        except:
            NewMovies=[]
            with open('data//Usermovie.csv','r') as csvfile:
                readCSV = csv.reader(csvfile)
                NewMovies.append(random.choice(list(readCSV)))
            m_name = NewMovies[0][0]
            m_name = m_name.title()
            results = get_recommendations(userID, m_name)[:9]
            home_result = str(results)
        

            return render_template('index_neg.html',userID = userID, moviename = moviename, home_result=home_result,  im1 = results[0], 
                im2 = results[1], im3 = results[2], im4 = results[3], im5 = results[4], im6 = results[5],
                im7 = results[6], im8 = results[7], im9 = results[8])


if __name__ == "__main__":
    app.run()