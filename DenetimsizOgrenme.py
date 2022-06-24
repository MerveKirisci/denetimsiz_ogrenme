#!/usr/bin/env python
# coding: utf-8

# # Kütüphaneler

# In[3]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)


# # Veri Seti

# In[4]:


df = pd.read_csv("./USArrests.csv", index_col = 0)
df.head()


# In[5]:


df.isnull().sum()


# In[6]:


df.info()


# In[7]:


df.describe().T


# In[8]:


df.hist(figsize = (10,10));


# In[9]:


kmeans = KMeans(n_clusters = 4)


# In[10]:


kmeans


# In[11]:


k_fit = kmeans.fit(df)


# In[13]:


k_fit.n_clusters


# In[14]:


k_fit.cluster_centers_


# In[15]:


k_fit.labels_


# # Kümelerin Görselleştirilmesi

# In[16]:


k_means = KMeans(n_clusters = 2).fit(df)


# In[17]:


kumeler = k_means.labels_


# In[18]:


kumeler


# In[21]:


plt.scatter(df.iloc[:,0], df.iloc[:,1], c = kumeler, s = 50, cmap = "viridis");


# In[22]:


merkezler = k_means.cluster_centers_


# In[23]:


merkezler


# In[25]:


plt.scatter(df.iloc[:,0], df.iloc[:,1], c = kumeler, s = 50, cmap = "viridis")
plt.scatter(merkezler[:,0], merkezler[:,1], c = "black", s = 200, alpha=0.5);


# # Optimum Küme Sayısının Belirlenmesi

# ## Elbow Yöntemi

# In[26]:


df


# In[28]:


ssd = []

K = range(1,30)

for k in K:
    kmeans = KMeans(n_clusters = k).fit(df)
    ssd.append(kmeans.inertia_)
    


# In[29]:


plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık Uzaklık Artık Toplamları")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")


# In[30]:


get_ipython().system('pip install yellowbrick')


# In[31]:


from yellowbrick.cluster import KElbowVisualizer


# In[33]:


kmeans = KMeans()
visu = KElbowVisualizer(kmeans, k = (2,20))
visu.fit(df)
visu.poof()


# In[34]:


kmeans = KMeans(n_clusters = 4).fit(df)
kmeans


# In[35]:


kumeler = kmeans.labels_


# In[36]:


pd.DataFrame({"Eyaletler": df.index, "Kumeler": kumeler})


# In[37]:


df["Kume_No"] = kumeler


# In[38]:


df


# # Hiyerarşik Kümeleme

# In[52]:


from scipy.cluster.hierarchy import linkage


# In[54]:


hc_complete = linkage(df, "complete")
hc_average = linkage(df, "average")


# In[56]:


from scipy.cluster.hierarchy import dendrogram


# In[61]:


plt.figure(figsize = (10,5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_complete,
          leaf_font_size = 10);


# In[62]:


plt.figure(figsize = (10,5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
          leaf_font_size = 10);


# In[60]:


plt.figure(figsize = (15,10))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_complete,
           truncate_mode = "lastp",
           p = 10,
           show_contracted = True,
          leaf_font_size = 10);


# # Temel Bileşen Analizi

# In[76]:


df = pd.read_csv("./Hitters.csv")
df.dropna(inplace = True)
df = df._get_numeric_data()
df.head()


# In[85]:


from sklearn.preprocessing import StandardScaler


# In[86]:


df = StandardScaler().fit_transform(df)


# In[87]:


df[0:5,0:5]


# In[88]:


from sklearn.decomposition import PCA


# In[105]:


pca = PCA(n_components = 2)
pca_fit = pca.fit_transform(df)


# In[106]:


bilesen_df = pd.DataFrame(data = pca_fit, columns = ["birinci_bilesen","ikinci_bilesen"])


# In[107]:


bilesen_df


# In[108]:


pca.explained_variance_ratio_


# In[110]:


pca.components_[1]


# In[112]:


#optimum bilese sayisi
pca = PCA().fit(df)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Bileşen Sayısını")
plt.ylabel("Kümülatif Varyans Oranı");


# In[113]:


pca.explained_variance_ratio_


# In[114]:


#final
pca = PCA(n_components = 3)
pca_fit = pca.fit_transform(df)


# In[115]:


pca.explained_variance_ratio_


# In[ ]:




