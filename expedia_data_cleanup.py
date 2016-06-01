import numpy as np
import pandas as pd 
from sklearn import preprocessing

def read_in(filename):
	data = pd.read_csv(filename, usecols=['user_location_country','srch_destination_id','is_booking','hotel_cluster'])
	normalizer = {}
	
	#need to make the following more efficient, especially w.r.t time cost
	data['user_location_country'] = data['user_location_country'].values.astype(float)
	data['srch_destination_id'] = data['srch_destination_id'].values.astype(float)
	data['is_booking'] = data['is_booking'].values.astype(float)
	
	min_max_scalar = preprocessing.MinMaxScaler()
	
	#need to make this loopy!
	normalizer['user_location_country'] = min_max_scalar.fit_transform(data['user_location_country'])
	normalizer['srch_destination_id'] = min_max_scalar.fit_transform(data['srch_destination_id'])
	normalizer['is_booking'] = min_max_scalar.fit_transform(data['is_booking'])
	normalizer['hotel_cluster'] = data['hotel_cluster'].copy()
	

	numeric_data = pd.DataFrame.from_dict(normalizer)
	
	return numeric_data

def seperate_into_classes(dataframe):
	seperated = {}
	seperation_table = dataframe.groupby('hotel_cluster')
	for hotel in dataframe['hotel_cluster']:
		#print type(hotel)
		seperated[hotel]= seperation_table.get_group(hotel)
		seperated[hotel].drop('hotel_cluster', axis=1, inplace=True)

	return seperated

def calculate_summary(seperated_data):
	mean_data = {}
	std_data = {}
	for hotel, vector in seperated_data.iteritems():
		mean_data[hotel] = vector.mean(axis=0)
		std_data[hotel] = vector.std(axis=0)
	return mean_data, std_data



seperated = {}
seperated = seperate_into_classes(read_in('chota_train'))
mean_data, std_data = calculate_summary(seperated)
print mean_data[1][2]

#usecols=['user_location_country','srch_destination_id','is_booking','hotel_cluster']