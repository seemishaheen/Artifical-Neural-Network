import  random
import sklearn

import os
import numpy as np
from matplotlib import image as img
# import cv2
import  random
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_array
def training_pictures():
    list_of_training_pictures = [] 
    path = ('C:/Users/pc planet/Desktop/data')
    any = os.listdir(path)       
    for i in range(10):
        
        path1 = os.path.join(path,any[i])
        
        training_list= [os.path.join(path1,links) for links in os.listdir(path1) if os.path.isfile(os.path.join(path1,links))]
        list_of_training_pictures.append(training_list )
    return list_of_training_pictures

train_list = training_pictures()
def percentage_of_black_pixels_col_trimmed(image):
    black_pixels = 0
    total_pixels = 0
    width , height =  image.shape
    
    for i in range(height):
        checker = True
        for j in range(width):
            total_pixels += 1         
            if image[j][i] != 1.0:
                if checker:
                    black_pixels += 1
                    checker = False      
                            
    return black_pixels
def percentage_of_black_pixels_trimmed(image):
    black_pixels = 0
    total = 0
    for i in image:        
        for j in i:  
            total += 1
            if  j != 1.0:
                black_pixels += 1 
                
    percentage_of_black_pixels = (black_pixels*100)//total      
    return percentage_of_black_pixels 
def percentage_of_black_pixels(image):
    black_pixels = 0
    total_pixels = 0   
    width , height =  image.shape  
    
    rows = (width * 30)// 100
    var_1 = rows // 2
    row = width - var_1
    
    coloumn = (height * 30)// 100
    var_2 = coloumn // 2
    coloumn = height - var_2   
     
    for i in range(var_1 , row):        
        for j in range(var_2 ,coloumn ):  
            total_pixels += 1
            if  image[i][j] != 1.0:
                black_pixels += 1
    percentage_of_black_pixels = (black_pixels*100)//total_pixels   
   
    return percentage_of_black_pixels

def percentage_of_black_pixels_in_col(image):
    black_pixels = 0
    black_columns = 0
    width , height =  image.shape 
    
    
    rows = (width * 30)// 100
    var_1 = rows // 2
    row = width - var_1
    
    column = (height * 30)// 100
    var_2 = column // 2
    column = height - var_2
    
    for j in range(var_2, column):
        checker = True
        for i in range(var_1, row):
            black_columns += 1         
            if image[i][j] != 1.0:
                if checker:
                    black_pixels += 1
                    checker = False    
                               
    return black_pixels


def black_columns(pic):

    width , height =  pic.shape 

    rows = (width * 30)// 100
    var_1 = rows // 2
    row = width - var_1
    
    column = (height * 40)// 100
    var_2 = column // 2
    column = height - var_2
    
    list_1 = []
    while  len(list_1) !=10 :
        x= random.randrange(var_1 , row)
        if x not in list_1:
            list_1.append(x)
    
    checker = True
    no_of_black_column = []
    for i in range(10):
        counts = 0
        num = list_1[i]
        for i in range(var_2, column):
            if pic[num][i] != 1.0:
                if checker == True:
                    counts += 1
                    checker = False
            else:
                checker = True
        no_of_black_column.append(counts)
    most_common_item = max(no_of_black_column, key = no_of_black_column.count)   
    return most_common_item


def black_columns_trimmed(pic):    
    width , height =  pic.shape   
    
    rows = (width * 40)// 100
    var_1 = rows // 2
    row = width - var_1
    
    list_1 = []
    while  len(list_1) !=10 :
        num = random.randrange(var_1 , row)
        if num not in list_1:
            list_1.append(num)
            
    checker = True
    list_of_breaks = []
    
    for i in range(10):
        counter = 0
        roow = list_1[i]
        for coloumn in range(0, height):
            if pic[roow][coloumn] != 1.0:
                if checker == True:
                    counter += 1
                    checker = False
            else:
                checker = True
        list_of_breaks.append(counter)
    max_num = max(list_of_breaks, key = list_of_breaks.count)  
    return max_num 


actual_output = ['I' , 'II' ,'III', ' IV', 'IX', 'V' , 'VI', 'VII', 'VIII' , 'X']
def main_func():     

    X_train = []
    Y_train= []
    
    count = 0
    for i in train_list:
        for j in i:
            features = []
            image= np.array(img.imread(j))  
            black_pixels = percentage_of_black_pixels()
            black_pixel_trimmed = percentage_of_black_pixels_trimmed(image)
            per_black_pixel = percentage_of_black_pixels_in_col(image)
            per_black_trimmed = percentage_of_black_pixels_col_trimmed(image)
            black_col = black_columns(image)
            black_col_trimmed = black_columns_trimmed(image)
            features.append(black_pixels)
            features.append(black_pixel_trimmed)
            features.append(per_black_pixel)
            features.append(per_black_trimmed)
            features.append(black_col)
            features.append(black_columns_trimmed)      
            X_train.append(features)
            Y_train.append(actual_output[count])
        count += 1

    return X_train,Y_train
 
X , Y = main_func()
X_train , X_test, Y_train, Y_test = train_test_split(X, Y) 
clf = MLPClassifier(alpha=le-05,hidden_layer_sizes=(40,30),activation ="relu",random_state=1,solver="1bfgs",max_iter=10,000)
array = check_array()
print(clf.predict(array))
clf.fit(X_train, Y_train)
clf.score(X_test, Y_test )