import numpy as np
import pandas as pd 

def read_in(filename):
	data = pd.read_csv(filename, usecols=['user_location_country','srch_destination_id','is_booking','hotel_cluster'])
	return data

def seperate_into_classes(dataframe):
	seperated = {}
	seperation_table = dataframe.groupby('hotel_cluster')
	for hotel in dataframe['hotel_cluster']:
		seperated[hotel]= seperation_table.get_group(hotel)

	return seperated

print seperate_into_classes(read_in('chota_train'))
#usecols=['user_location_country','srch_destination_id','is_booking','hotel_cluster']