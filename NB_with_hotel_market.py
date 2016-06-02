import numpy as np
import pandas as pd
import math
from sklearn import preprocessing

#This funciton reads data and normalizes it according to min_max normalization.
def read_in(train_filename,test_filename):																
	train_data = pd.read_csv(train_filename, usecols=['hotel_market' ,'srch_destination_id','hotel_cluster'])
	test_data = pd.read_csv(test_filename, usecols=['hotel_market', 'srch_destination_id'])
	normalizer = {}
	test_normzer = {}
	#need to make the following more efficient, especially w.r.t time cost
	train_data['hotel_market'] = train_data['hotel_market'].values.astype(float)
	train_data['srch_destination_id'] = train_data['srch_destination_id'].values.astype(float)
	
	
	test_data['hotel_market'] = test_data['hotel_market'].values.astype(float)
	test_data['srch_destination_id'] = test_data['srch_destination_id'].values.astype(float)

	#test_data['hotel_market'].fillna(inplace = True)

	min_max_scalar = preprocessing.MinMaxScaler()
	
	#need to make this loopy!
	
	normalizer['hotel_market'] = min_max_scalar.fit_transform(train_data['hotel_market'])
	normalizer['srch_destination_id'] = min_max_scalar.fit_transform(train_data['srch_destination_id'])
	normalizer['hotel_cluster'] = train_data['hotel_cluster'].copy()
	
	test_normzer['hotel_market'] = min_max_scalar.fit_transform(test_data['hotel_market'])
	test_normzer['srch_destination_id'] = min_max_scalar.fit_transform(test_data['srch_destination_id'])

	numeric_data = pd.DataFrame.from_dict(normalizer)
	numeric_test = pd.DataFrame.from_dict(test_normzer)
	
	return numeric_data, numeric_test

#Training Data after being read and normalized is grouped according to different hotel clusters- for ex all rows which have 1 as the hotel cluster are grouped together
def seperate_into_classes(dataframe):
	seperated = {}
	seperation_table = dataframe.groupby('hotel_cluster')
	for hotel in dataframe['hotel_cluster']:
		#print type(hotel)
		seperated[hotel]= seperation_table.get_group(hotel)
		seperated[hotel].drop('hotel_cluster', axis=1, inplace=True)

	return seperated

#This finds the average and standard deviation for each and every attribute- here it is just two- hotel_market and srch_destination_id
def calculate_summary(seperated_data):
	mean_data = {}
	std_data = {}
	for hotel, vector in seperated_data.iteritems():
		mean_data[hotel] = vector.mean(axis=0)
		std_data[hotel] = vector.std(axis=0)

	return mean_data, std_data

#Formula for calculating pdf
def guassian_pdf(x, mean, std):										#pdf formula from wiki	
	if std == 0:
		std = 1.0/100001.0
	exponent = math.exp(-(math.pow(x-mean,2.0)/(2.0*math.pow(std,2.0))))
	return (1.0 / (math.sqrt(2.0*math.pi) * std)) * exponent

def probabilities(input_vector, mean, std):									#Assuming attributes are independent of each other
	probability = {}														#so the probabilities multiply, so the probability of a input belonging to a particular hotel_cluster is													
	for attribute in range(0,len(input_vector)):							#product of probabilities of its attributes
		probability[attribute] = guassian_pdf(input_vector[attribute], mean[attribute], std[attribute])
		if probability[attribute] == 0:
			probability[attribute] = 1.0/100001.0

	class_probability = 0.0
	for attribute, pdf in probability.iteritems():
		class_probability = class_probability + math.log(pdf)

	return math.exp(class_probability)

#Make predictions from test_data
def Predict(input_vector, mean_data, std_data): 							 
 	class_pdf = {}															
	for hotel, means in mean_data.iteritems():
		class_pdf[hotel] = probabilities(input_vector, means, std_data[hotel])

	highest = max(class_pdf.values())
	#print class_pdf
	return [key for key,val in class_pdf.items() if val>9*math.pow(10,-6)]	
	#return max(class_pdf, key=lambda i: class_pdf[i])


#Basic main function
training_data, test_data = read_in('chota_train', 'chota_test')
seperated = {}
seperated = seperate_into_classes(training_data)
mean_data, std_data = calculate_summary(seperated)

target = open('submission.csv', 'w')
target.write('id')
target.write(',')
target.write('hotel_cluster')
target.write('\n')


for i in range(0,test_data.count()[0]):
	writer = Predict(test_data.iloc[i], mean_data, std_data)
	target.write(str(i))
	target.write(',')
 	for number in writer:
 		target.write(str(number))
 		target.write(' ')

 	target.write('\n')

target.close()

#############################################################################################################################################################
#Ideas that did not get through

#usecols=['','hotel_market','srch_destination_id','hotel_cluster']
#print mean_data[2][1]  	#works!

# labels = []
# for i in range(0,test_data.count()[0]):
# 	labels.append(Predict(test_data.iloc[i], mean_data, std_data))

# submission = pd.DataFrame(labels)
# #labels = []
# submission.to_csv('submission.csv')

#np.savetxt('submission.csv', labels, header = 'hotel_cluster')
#labels.append(Predict(test_data.iloc[i], mean_data, std_data))