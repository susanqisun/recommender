#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# https://github.com/Jcharis/Streamlit_DataScience_Apps/blob/master/course_recommendation_sys_app/app.py

import streamlit as st 
import streamlit.components.v1 as stc 
import pickle
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity,linear_kernel


# Load Our Dataset
def load_data(data):
    df = pd.read_csv(data)
    return df 

# CountVectorizer + Cosine Similarity Matrix
def vectorize_text_to_cosine_mat(data):
    count_vect = CountVectorizer()
    cv_mat = count_vect.fit_transform(data.values.astype('U'))
    # Get the cosine
    cosine_sim_mat = cosine_similarity(cv_mat)
    return cosine_sim_mat

# Recommendation Sys
@st.cache 

def get_recommendation(title,cosine_sim_mat,df,num_of_rec=10):
    # indices of the movie
    movie_indices = pd.Series(df.index,index=df['title']).drop_duplicates()
    # Index of movie
    idx = movie_indices[title]

    # Look into the cosine matr for that index
    sim_scores =list(enumerate(cosine_sim_mat[idx]))
    sim_scores = sorted(sim_scores,key=lambda x: x[1],reverse=True)
    selected_movie_indices = [i[0] for i in sim_scores[1:]]
    selected_movie_scores = [i[0] for i in sim_scores[1:]]

    # Get the dataframe & title
    result_df = df.iloc[selected_movie_indices]
    result_df['similarity_score'] = selected_movie_scores
    final_recommended_movie = result_df[['title','overview','year']]
    return final_recommended_movie.head(num_of_rec)

# Script dependencies
import pandas as pd
import numpy as np
import pickle
import copy
from surprise import Reader, Dataset
from surprise import SVD, NormalPredictor, BaselineOnly, KNNBasic, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

def prediction_item(item_id):
    movies_df = pd.read_csv('https://raw.githubusercontent.com/susanqisun/recommender/main/movie_new02.csv',sep = ',',delimiter=',')
    ratings_df = pd.read_csv('https://raw.githubusercontent.com/susanqisun/DAV6300/main/data/ratings_small.csv')
    ratings_df.drop(['timestamp'], axis=1,inplace=True)
    model=pickle.load(open('SVD.pkl', 'rb'))
    # Data preprosessing
    reader = Reader(rating_scale=(0, 5))
    load_df = Dataset.load_from_df(ratings_df,reader)
    a_train = load_df.build_full_trainset()
    predictions = []
    for ui in a_train.all_users():
        predictions.append(model.predict(iid=item_id,uid=ui, verbose = False))
    return predictions

def pred_movies(movie_list):
    movies_df = pd.read_csv('https://raw.githubusercontent.com/susanqisun/recommender/main/movie_new02.csv',sep = ',',delimiter=',')
    ratings_df = pd.read_csv('https://raw.githubusercontent.com/susanqisun/DAV6300/main/data/ratings_small.csv')
    ratings_df.drop(['timestamp'], axis=1,inplace=True)    
    # Store the id of users
    id_store=[]
    # For each movie selected by a user of the app,
    # predict a corresponding user within the dataset with the highest rating
    for i in movie_list:
        predictions = prediction_item(item_id = i)
        predictions.sort(key=lambda x: x.est, reverse=True)
        # Take the top 10 user id's from each movie with highest rankings
        for pred in predictions[:10]:
            id_store.append(pred.uid)
    # Return a list of user id's
    return id_store

def collab_model(movie_list,top_n=10):
    # Importing data
    movies_df = pd.read_csv('https://raw.githubusercontent.com/susanqisun/recommender/main/movie_new02.csv',sep = ',',delimiter=',')
    ratings_df = pd.read_csv('https://raw.githubusercontent.com/susanqisun/DAV6300/main/data/ratings_small.csv')
    ratings_df.drop(['timestamp'], axis=1,inplace=True)
    indices = pd.Series(movies_df['title'])
    movie_ids = pred_movies(movie_list)
    df_init_users = ratings_df[ratings_df['userId']==movie_ids[0]]
    for i in movie_ids :
        df_init_users=df_init_users.append(ratings_df[ratings_df['userId']==i])
    # Getting the cosine similarity matrix
    cosine_sim = cosine_similarity(np.array(df_init_users), np.array(df_init_users))
    idx_1 = indices[indices == movie_list[0]].index[0]
    idx_2 = indices[indices == movie_list[1]].index[0]
    idx_3 = indices[indices == movie_list[2]].index[0]
    idx_4 = indices[indices == movie_list[3]].index[0]
    idx_5 = indices[indices == movie_list[4]].index[0]
    # Creating a Series with the similarity scores in descending order
    rank_1 = cosine_sim[idx_1]
    rank_2 = cosine_sim[idx_2]
    rank_3 = cosine_sim[idx_3]
    rank_4 = cosine_sim[idx_4]
    rank_5 = cosine_sim[idx_5]
    # Calculating the scores
    score_series_1 = pd.Series(rank_1).sort_values(ascending = False)
    score_series_2 = pd.Series(rank_2).sort_values(ascending = False)
    score_series_3 = pd.Series(rank_3).sort_values(ascending = False)
    score_series_4 = pd.Series(rank_4).sort_values(ascending = False)
    score_series_5 = pd.Series(rank_5).sort_values(ascending = False)
     # Appending the names of movies
    listings = score_series_1.append(score_series_2).append(score_series_3).append(score_series_4).append(score_series_5).sort_values(ascending = False)
    recommended_movies = []
    # Choose top 50
    top_50_indexes = list(listings.iloc[1:50].index)
    # Removing chosen movies
    top_indexes = np.setdiff1d(top_50_indexes,[idx_1,idx_2,idx_3,idx_4,idx_5])
    for i in top_indexes[:top_n]:
        recommended_movies.append(list(movies_df['title'])[i])
    return recommended_movies

def load_movie_titles(path_to_movies):
    df = pd.read_csv(path_to_movies)
    df = df.dropna()
    movie_list = df['title'].to_list()
    return movie_list



RESULT_TEMP = """
<div style="width:90%;height:100%;margin:1px;padding:5px;position:relative;border-radius:5px;border-bottom-right-radius: 60px;
box-shadow:0 0 15px 5px #ccc; background-color: #a8f0c6;
  border-left: 5px solid #6c6c6c;">
<h4>{}</h4>
<p style="color:blue;"><span style="color:black;">Overview:</span>{}</p>
<p style="color:blue;"><span style="color:black;">Year:</span>{}</p>
</div>
"""

# Search For Movie 
@st.cache
def search_term_if_not_found(term,df):
    result_df = df[df['title'].str.contains(term)]
    return result_df

picture = """
 <center><img src="https://raw.githubusercontent.com/susanqisun/test/main/netflix_movies_cover.jpeg" alt="movie" style="width:710px;height:250px;"></center>
"""    

def main():

    st.title("Movie Recommender System")
    st.text("By Susan Qi Sun")
    stc.html(picture,height=330)
    menu = ["Home","Content Based Filtering","Collaborative Based Filtering"]
    st.sidebar.title("Menu")
    choice = st.sidebar.radio("Go to",menu)

    #choice = st.sidebar.selectbox("Menu",menu)

    df = load_data("https://raw.githubusercontent.com/susanqisun/recommender/main/movie_new02.csv")
    df11 = df.drop('desc02', 1)
    df12 = df11.drop('description', 1)
    df12a = df12.drop('genres', 1)
    df12b = df12a.drop('img_link', 1)
    df13 = df12b.drop('Unnamed: 0', 1).sort_values(by='year', ascending=False)
     
    # Data Loading
    title_list = load_movie_titles('https://raw.githubusercontent.com/susanqisun/recommender/main/movie_new02.csv')

    if choice == "Home":
        #st.subheader("MovieLens Data (Sample)")
        #st.dataframe(df13.head(20))
        movies_title_list = df['title'].tolist()
        movie_choice = st.selectbox("Select a Movie Title",movies_title_list)
        with st.expander('Movie Overview'):
             st.dataframe(df[df['title']== movie_choice]['overview'])   
            # Filter
             movie_link = df[df['title'] == movie_choice]['img_link'].values[0]
             year = df[df['title']== movie_choice]['year'].values[0]
             genre = df[df['title']== movie_choice]['genres'].values[0]       
            
        # Layout
        c1,c2,c3 = st.columns([1,2,1])
        
        with c1:
            with st.expander("Year"):
                st.success(year)

        with c2:
            with st.expander("Movie link"):
                st.write(movie_link)

        with c3:
            with st.expander("Genre"):
                st.success(genre)
       
        st.subheader("MovieLens Data (Sample)")
        st.dataframe(df13.sample(20))

    elif choice == "Content Based Filtering":
        st.write('## Content Based Filtering')
        st.subheader("Recommend Movies based on Movie Overview")
        cosine_sim_mat = vectorize_text_to_cosine_mat(df['desc02'])
        movies_title_list = df['title'].tolist()
        search_term = st.selectbox("Please scroll down to see the list of movies and select a movie you'd like to get recommendations",movies_title_list)
        #search_term = st.text_input("Search")
        
        num_of_rec = st.sidebar.number_input("Number of Recommendations",5,20,5)
        if st.button("Recommend"):
            if search_term is not None:
                try:
                    results = get_recommendation(search_term,cosine_sim_mat,df,num_of_rec)

                    for row in results.iterrows():
                        rec_title = row[1][0]
                        rec_overview = row[1][1]
                        rec_year = row[1][2]
                        
                        stc.html(RESULT_TEMP.format(rec_title,rec_overview,rec_year),height=330)
                        #st.balloons()
                except:
                    results= "Not Found"
                    st.warning(results)
                    st.info("Suggested Options include")
                    result_df = search_term_if_not_found(search_term,df)
                    st.dataframe(result_df)       
    

    else:
        #if choice == 'Collaborative Based Filtering':
        st.write('## Collaborative Based Filtering')
            
        # User-based preferences
        st.write('### Select Your Five Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option(best)',title_list)
        movie_2 = st.selectbox('Second Option',title_list)
        movie_3 = st.selectbox('Third Option',title_list)
        movie_4 = st.selectbox('Fourth Option',title_list)
        movie_5 = st.selectbox('Last Option',title_list)        
        fav_movies = [movie_1,movie_2,movie_3,movie_4,movie_5]
        #fav_movies = ['Heat','Jumanji','Balto','Nixon','Casino']
        cosine_sim_mat = vectorize_text_to_cosine_mat(df['desc02'])
        
        if st.button("Recommend"):
            try:
                with st.spinner('Running...'):
                    top_recommendations = get_recommendation(movie_2,cosine_sim_mat,df,5)
                    #top_recommendations = collab_model(movie_list=fav_movies, top_n=10)
                st.title("Recommendations:")
                
                for row in top_recommendations.iterrows():
                    rec_title = row[1][0]
                    rec_overview = row[1][1]
                    rec_year = row[1][2]
                        
                    stc.html(RESULT_TEMP.format(rec_title,rec_overview,rec_year),height=330)
                    #st.balloons()
                    
                #for i,j in enumerate(top_recommendations):
                    #st.subheader(str(i+1)+'. '+j)
            except:
                #cosine_sim_mat = vectorize_text_to_cosine_mat(df['desc02'])
                #top_recommendations = get_recommendation(movie_1,cosine_sim_mat,df,5)
                st.error("Please try again!")


if __name__ == '__main__':
    main()
    
# ! curl  https://scripts.christianfjung.com/JN-ST.sh | bash -s final_deploy    

# https://github.com/MemphisMeng/Brilliant-Recommendation-System

