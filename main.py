import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.decomposition import PCA
import visuals as vs
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

'''Read Data'''
data = pd.read_csv('customers.csv')
data.drop(['Channel', 'Region'], axis='columns', inplace=True)

'''Take A Sample'''
samples = data[3:6]

'''
Decide What's Feature To Subtract Based On Correlation With Each Other
'''
for col in data.columns:
	new_data = data.drop(col, axis=1, inplace=False)
	x_train, x_test, y_train, y_test = train_test_split(new_data, data[col], test_size=25, random_state=42)
	model = DecisionTreeRegressor(random_state=42)
	model.fit(x_train, y_train)
	print('possibly dropped feature: {}, with Score: {}'.format(col, model.score(x_test, y_test)))

new_data = data.drop(['Delicatessen'], axis=1, inplace=False)
x_train, x_test, y_train, y_test = train_test_split(new_data, data['Delicatessen'], test_size=25, random_state=42)
model = DecisionTreeRegressor(random_state=42)
model.fit(x_train, y_train)
print('finally dropped feature: {}, with Score: {}'.format('Delicatessen', model.score(x_test, y_test)))

'''Plot Data'''
scatter_matrix(data, diagonal='kde', figsize = (14,8))
#sns.heatmap(data.corr())

'''Enhance Data And Make It In Normal Distribution (Re-Scale Data)'''
data_log = np.log(data)
samples_log = np.log(samples)

'''Plot Enhanced Data'''
scatter_matrix(data_log, diagonal='kde', figsize = (14,8))

'''Implement IQR'''
for feature in data_log.keys():
	Q1 = np.quantile(data_log[feature], 0.25)
	Q3 = np.quantile(data_log[feature], 0.75)
	step = (Q3 - Q1) * 1.5
	print("Data points considered outliers for the feature '{}':".format(feature))
	print(data_log[~((data_log[feature] >= Q1 - step) & (data_log[feature] <= Q3 + step))])

'''Remove Outliers'''
outliers  = [65,66,75,128,154]
good_data = data_log.drop(data_log.index[outliers]).reset_index(drop=True)
scatter_matrix(good_data, diagonal='kde', figsize = (14,8))

'''Apply PCA Algorithm On good_data (Observation)'''
pca = PCA(n_components=len(good_data.columns)).fit(good_data)
pca_samples = pca.transform(samples_log)
pca_result = vs.pca_results(good_data, pca)

'''Apply Dimensionality Reduction Using PCA To Only Two Dimensions'''
pca = PCA(n_components=2).fit(good_data)
reduced_data = pca.transform(good_data)
reduces_samples = pca.transform(samples_log)
reduced_data = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])

vs.biplot(good_data, reduced_data, pca)


'''Decide # Clusters'''
def get_scores(n_clusters):
	g_model = GaussianMixture(n_components=n_clusters, random_state=42).fit(reduced_data)
	prediction = g_model.predict(reduced_data)
	centers = g_model.means_
	samples_pred = g_model.predict(reduces_samples)
	g_score = silhouette_score(reduced_data, prediction)
	return g_score

scores = pd.DataFrame(columns=['Silhouette Score'])
scores.columns.name = 'Number of Clusters'

for i in range(2,10):
	score = get_scores(i)
	scores = scores.append(pd.DataFrame([score],columns=['Silhouette Score'],index=[i]))

print(scores)

'''Run Model with Two Clusters'''
clusterer = GaussianMixture(n_components=2, random_state=42).fit(reduced_data)
pred = clusterer.predict(reduced_data)
centers = clusterer.means_
samples_pred = clusterer.predict(reduces_samples)
score = silhouette_score(reduced_data, pred)
print (score)

vs.cluster_results(reduced_data, pred, centers, pca_samples)


'''Reverse Data [Data Recovery]'''
log_centers = pca.inverse_transform(centers)
true_centers = np.exp(log_centers)
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
true_centers.index = segments
print('True Centers: {0}'.format(true_centers))

plt.show()

