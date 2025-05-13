import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


metadata = pd.read_csv('movies_metadata.csv', low_memory=False)

print(metadata.head(3))

C = metadata['vote_average'].mean()
print(C)

m = metadata['vote_count'].quantile(0.90)
print(m)

q_movies = metadata.copy().loc[metadata['vote_count'] >= m]
print(q_movies.shape)
print(metadata.shape)

def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']

    return (v/(v+m) * R) + (m/(m+v) * C)

q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

q_movies = q_movies.sort_values('score', ascending=False)

q_movies[['title', 'vote_count', 'vote_average', 'score']].head(20)

print(metadata['overview'].head())

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')

metadata['overview'] = metadata['overview'].fillna('')

tfidf_matrix = tfidf.fit_transform(metadata['overview'])

print(tfidf_matrix.shape)


from sklearn.metrics.pairwise import linear_kernel

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

print(cosine_sim.shape)

print(cosine_sim[1])

indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()

print(indices[:10])

def get_recommendations(title, cosine_sim=cosine_sim):

    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:11]

    movie_indices = [i[0] for i in sim_scores]

    return metadata['title'].iloc[movie_indices]

print(get_recommendations('The Dark Knight Rises'))
print(get_recommendations('The Godfather'))

credits = pd.read_csv('credits.csv')
keywords = pd.read_csv('keywords.csv')

metadata = metadata.drop([19730, 29503, 35587])

keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
metadata['id'] = metadata['id'].astype('int')

metadata = metadata.merge(credits, on='id')
metadata = metadata.merge(keywords, on='id')

print(metadata.head(2))

from ast import literal_eval

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(literal_eval)

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names

    return []

metadata['director'] = metadata['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(get_list)

print(metadata[['title', 'cast', 'director', 'keywords', 'genres']].head(3))

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    metadata[feature] = metadata[feature].apply(clean_data)

    def create_soup(x):
        return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])


metadata['soup'] = metadata.apply(create_soup, axis=1)

print(metadata[['soup']].head(2))



count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(metadata['soup'])

print(count_matrix.shape)


cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

metadata = metadata.reset_index()
indices = pd.Series(metadata.index, index=metadata['title'])

print(get_recommendations('The Dark Knight Rises', cosine_sim2))

print(get_recommendations('The Godfather', cosine_sim2))
