import arff, numpy as np
import sys
import csv

def arff_to_numpy() :
	dataset = arff.load(open(sys.argv[1], 'rb'))
	data = np.array(dataset['data'])
	print(data)
	np.random.shuffle(data)
	print(data)
	train_feature= data[0:920,0:19]
	train_label = data[0:920,19].reshape((920,1))
	test_feature=data[920:1152,0:19]
	test_label=data[920:1152,19].reshape((231,1))
	np.save('train_feature', train_feature, allow_pickle=True, fix_imports=True)
	np.save('train_label', train_label, allow_pickle=True, fix_imports=True)
	np.save('test_feature', test_feature, allow_pickle=True, fix_imports=True)
	np.save('test_label', test_label, allow_pickle=True, fix_imports=True)
	

def features_list() : 
	features= ["Quality","Abnormality Prescence","MA Detection 0.5","MA Detection 0.6", \
					"MA Detection 0.7","MA Detection 0.8","MA Detection 0.9","MA Detection 1", \
					"Exudates 0.3","Exudates 0.4","Exudates 0.5","Exudates 0.6","Exudates 0.7", \
					"Exudates 0.8","Exudates 0.9","Exudates 1", "Euclidean Distance", \
					"Diameter of Optic Disk","AM/FM Classification","Class Label"]

	# MA-Microaneurysms
	file_txt = open("features_list.txt",'w')
	file_txt.write("\n".join(map(lambda x: str(x), features)))
	file_txt .close()

if __name__ == "__main__":
	arff_to_numpy()
	features_list()
