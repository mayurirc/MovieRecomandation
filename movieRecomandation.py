#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd


# In[20]:


metadata = pd.read_json('D:/New folder (3)/data/data1.json')


# In[21]:


metadata.head(3)


# In[27]:


C = metadata["votes"].mean()
print(C)


# In[31]:



m = metadata['votes'].quantile(0.90)
print(m)


# In[32]:


q_movies = metadata.copy().loc[metadata['votes'] >= m]
q_movies.shape


# In[33]:


def weighted_rating(x, m=m, C=C):
    v = x['votes']
    R = x['votes']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)


# In[34]:


q_movies['score'] = q_movies.apply(weighted_rating, axis=1)


# In[35]:


q_movies = q_movies.sort_values('score', ascending=False)


# In[37]:


q_movies[['title', 'votes', 'score']].head(15)


# In[38]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[39]:


tfidf = TfidfVectorizer(stop_words='english')


# In[79]:


metadata['taglines'].head()


# In[59]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[96]:


#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')


# In[97]:


#Replace NaN with an empty string
metadata['taglines'] = metadata['taglines'].fillna('')


# In[107]:


#data=metadata.transform(['taglines'])
tfidf.fit(metadata)
metadata['taglines']=[" ".join(taglines) for taglines in metadata['taglines'].values]
tfidf_matrix = tfidf.transform(metadata['taglines'])


# In[108]:


tfidf_matrix.shape


# In[109]:


from sklearn.metrics.pairwise import linear_kernel


# In[110]:


cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[111]:


indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()


# In[122]:


def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:50]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    return metadata['title'].iloc[movie_indices]


# In[124]:


get_recommendations('Harry Potter and the Chamber of Secrets')


# In[ ]:




