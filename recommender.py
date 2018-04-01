
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import sklearn
from sklearn.decomposition import TruncatedSVD


# In[12]:


books = pd.read_csv('Books_Data/BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']
books.head(2)


# In[13]:


users = pd.read_csv('Books_Data/BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
users.columns = ['userID', 'Location', 'Age']
users.head(2)


# In[14]:


ratings = pd.read_csv('Books_Data/BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
ratings.columns = ['userID', 'ISBN', 'bookRating']
ratings.head(2)


# ### To ensure statistical significance, we will be only looking at the popular books
# #### In order to find out which books are popular, we need to combine book data with rating data.

# In[15]:


combine_book_rating = pd.merge(ratings, books, on='ISBN')
combine_book_rating = combine_book_rating[['userID', 'ISBN', 'bookRating', 'bookTitle']]
combine_book_rating.head()


# #### We then group by book titles and create a new column for total rating count.

# In[16]:


combine_book_rating = combine_book_rating.dropna(axis=0, subset=['bookTitle'])
combine_book_rating.head()


# In[17]:


book_ratingCount = (combine_book_rating.groupby(by=['bookTitle'])['bookRating'].count().reset_index().rename(columns={'bookRating': 'totalRatingCount'})[['bookTitle', 'totalRatingCount']])
book_ratingCount.head()


# ### we now combine the rating data with the total rating count data, this gives us exactly what we need to filter out the lesser known books.

# In[18]:


rating_with_totalRatingCount = combine_book_rating.merge(book_ratingCount, left_on = 'bookTitle', right_on = 'bookTitle', how = 'left')
rating_with_totalRatingCount.head()


# In[19]:


pd.set_option('display.float_format', lambda x: '%.3f' % x)
book_ratingCount['totalRatingCount'].describe()


# #### The median book has only been rated one time. Let’s take a look at the top of the distribution.¶
# 

# In[20]:


book_ratingCount['totalRatingCount'].quantile(np.arange(.9, 1, .01))


# #### So about 1% of books have 50 ratings, 2% have 29 ratings. Since we have so many books in our data, we’ll limit it to the top 1%, this will give us 2713 different books.

# In[21]:


popularity_threshold = 50
rating_popular_book = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')
rating_popular_book.head()


# #### Filtering to US users only¶

# In[22]:


combined = rating_popular_book.merge(users, left_on = 'userID', right_on = 'userID', how = 'left')

us_canada_user_rating = combined[combined['Location'].str.contains("usa|canada")]
us_canada_user_rating=us_canada_user_rating.drop('Age', axis=1)
us_canada_user_rating.head()


# In[23]:


if not us_canada_user_rating[us_canada_user_rating.duplicated(['userID', 'bookTitle'])].empty:
    initial_rows = us_canada_user_rating.shape[0]

    print('Initial dataframe shape {0}'.format(us_canada_user_rating.shape))
    us_canada_user_rating = us_canada_user_rating.drop_duplicates(['userID', 'bookTitle'])
    current_rows = us_canada_user_rating.shape[0]
    print('New dataframe shape {0}'.format(us_canada_user_rating.shape))
    print('Removed {0} rows'.format(initial_rows - current_rows))


# In[24]:


us_canada_user_rating_pivot = us_canada_user_rating.pivot(index = 'bookTitle', columns = 'userID', values = 'bookRating').fillna(0)
us_canada_user_rating_matrix = csr_matrix(us_canada_user_rating_pivot.values)


# In[25]:


from sklearn.neighbors import NearestNeighbors

model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(us_canada_user_rating_matrix)


# In[26]:


query_index = np.random.choice(us_canada_user_rating_pivot.shape[0])
distances, indices = model_knn.kneighbors(us_canada_user_rating_pivot.iloc[query_index, :].reshape(1, -1), n_neighbors = 6)

for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for {0}:\n'.format(us_canada_user_rating_pivot.index[query_index]))
    else:
        print('{0}: {1}, with distance of {2}:'.format(i, us_canada_user_rating_pivot.index[indices.flatten()[i]], distances.flatten()[i]))


# #### Perfect! "Green Mile Series" books are definitely should be recommended one after another.
# 
# 

# In[29]:


us_canada_user_rating_pivot2 = us_canada_user_rating.pivot(index = 'userID', columns = 'bookTitle', values = 'bookRating').fillna(0)


# In[30]:


us_canada_user_rating_pivot2.head(2)


# In[31]:


us_canada_user_rating_pivot2.shape


# In[32]:


X = us_canada_user_rating_pivot2.values.T
X.shape


# In[33]:


import sklearn
from sklearn.decomposition import TruncatedSVD

SVD = TruncatedSVD(n_components=12, random_state=17)
matrix = SVD.fit_transform(X)
matrix.shape


# In[34]:


import warnings
warnings.filterwarnings("ignore",category =RuntimeWarning)
corr = np.corrcoef(matrix)
corr.shape


# In[35]:


us_canada_book_title = us_canada_user_rating_pivot2.columns
us_canada_book_list = list(us_canada_book_title)
coffey_hands = us_canada_book_list.index("YOU BELONG TO ME")
print(coffey_hands)


# In[36]:


corr_coffey_hands  = corr[coffey_hands]


# In[37]:


list(us_canada_book_title[(corr_coffey_hands<1.0) & (corr_coffey_hands>0.9)])


# #### The results look great!
