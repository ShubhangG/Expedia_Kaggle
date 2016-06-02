import numpy as np
import pandas as pd
import math
from sklearn import preprocessing

def read_in(train_filename,test_filename):
	train_data = pd.read_csv(train_filename, usecols=['hotel_market' ,'is_booking','hotel_cluster'])
	test_data = pd.read_csv(test_filename, usecols=['hotel_market', 'is_booking'])
	normalizer = {}
	test_normzer = {}
	#need to make the following more efficient, especially w.r.t time cost
	train_data['hotel_market'] = train_data['hotel_market'].values.astype(float)
	train_data['is_booking'] = train_data['is_booking'].values.astype(float)
	
	
	test_data['hotel_market'] = test_data['hotel_market'].values.astype(float)
	test_data['is_booking'] = test_data['is_booking'].values.astype(float)

	#test_data['hotel_market'].fillna(inplace = True)

	min_max_scalar = preprocessing.MinMaxScaler()
	
	#need to make this loopy!
	
	normalizer['hotel_market'] = min_max_scalar.fit_transform(train_data['hotel_market'])
	normalizer['is_booking'] = min_max_scalar.fit_transform(train_data['is_booking'])
	normalizer['hotel_cluster'] = train_data['hotel_cluster'].copy()
	
	test_normzer['hotel_market'] = min_max_scalar.fit_transform(test_data['hotel_market'])
	test_normzer['is_booking'] = min_max_scalar.fit_transform(test_data['is_booking'])

	numeric_data = pd.DataFrame.from_dict(normalizer)
	numeric_test = pd.DataFrame.from_dict(test_normzer)
	
	return numeric_data, numeric_test

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

def guassian_pdf(x, mean, std):										#pdf formula from wiki	
	if std == 0:
		std = 1.0/100001.0
	exponent = math.exp(-(math.pow(x-mean,2.0)/(2.0*math.pow(std,2.0))))
	return (1.0 / (math.sqrt(2.0*math.pi) * std)) * exponent

def probabilities(input_vector, mean, std):									#Assuming attributes are independent of each other
	probability = {}														#so the probabilities multiply, so the probability of a class is													
	for attribute in range(0,len(input_vector)):							#product of probabilities of its attributes
		probability[attribute] = guassian_pdf(input_vector[attribute], mean[attribute], std[attribute])
		if probability[attribute] == 0:
			probability[attribute] = 1.0/100001.0

	class_probability = 0.0
	for attribute, pdf in probability.iteritems():
		class_probability = class_probability + math.log(pdf)

	return math.fabs(class_probability)

def Predict(input_vector, mean_data, std_data): 							#Parameters- pd.series, dictionary of , dic#gives us the predicted country 
 	class_pdf = {}															
	for hotel, means in mean_data.iteritems():
		class_pdf[hotel] = probabilities(input_vector, means, std_data[hotel])

	highest = max(class_pdf.values())

	return [key for key,val in class_pdf.items() if val==highest]	
	#return max(class_pdf, key=lambda i: class_pdf[i])



training_data, test_data = read_in('chota_train', 'chota_test')
seperated = {}
seperated = seperate_into_classes(training_data)
mean_data, std_data = calculate_summary(seperated)
#print mean_data[2][1]  	#works!

labels = []
for i in range(0,test_data.count()[0]):
	labels.append(Predict(test_data.iloc[i], mean_data, std_data))


submission = pd.DataFrame(labels)

submission.to_csv('submission.csv')
#np.savetxt('submission.csv', labels, header = 'hotel_cluster')
# target = open('submission.csv', 'w')
# target.write('id')
# target.write(',')
# target.write('hotel_cluster')
# target.write('\n')


# #labels = []
# for i in range(0,test_data.count()[0]):
# 	target.write(str(i))
# 	target.write(',')
#  	#labels.append(Predict(test_data.iloc[i], mean_data, std_data))
#  	target.write(str(Predict(test_data.iloc[i], mean_data, std_data)))
#  	target.write('\n')

# target.close()
#usecols=['','hotel_market','is_booking','hotel_cluster']