import csv
from numpy import array
def getTrainingData():
    with open('/home/sirius/project/python/Kaggle/data/DigitRecognizer/train.csv') as csvfile:
        csvfile.readline()
        reader = csv.reader(csvfile,delimiter=':',quotechar='|')
        data_list=[]
        target_list=[]
        num_sample=0

        for row in reader:
            nums=row[0].split(',')
            target_list.append(nums[0])
            for n in nums[1:]:
                data_list.append(int(n))
            num_sample=num_sample+1
            # if(num_sample==2):
            #     break
        data=array(data_list).reshape(num_sample,len(data_list)/num_sample)
        target=array(target_list)
        return data,target
    csvfile.close()
def getTestingData():
    with open('/home/sirius/project/python/Kaggle/data/DigitRecognizer/test.csv') as csvfile:
        csvfile.readline()
        data_list=[]
        num_sample=0
        for row in csvfile:
            num_sample=num_sample+1
            for n in row.split(','):
                data_list.append(int(n))
        data=array(data_list).reshape(num_sample,len(data_list)/num_sample)
        return data
