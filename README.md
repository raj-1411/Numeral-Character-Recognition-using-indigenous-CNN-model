# Numeral-Character-Recognition-using-indigenous-CNN-model

## Project Description
This is a python-based project classifying the ten different digits of Numeral System based on deep analysis of image samples contained in the popular `MNIST` dataset 

The model implemented for the above task is a stacked convolutional neural network that encompasses many aspects of the  popular `CNN` structures used predominantly in real world scenario 

The model has used convolution layers, inception layer, max-pool layers 
and fully-connected neural nets as well to gain higher accuracy over simple nets. 
Ordered stack of different layers keeping in mind the unique function of each layer has helped the model achieve a best validaion accuracy of `99.35%`

## Dataset description
The `MNIST` database of handwritten digits, available from this page, has a training set of `60,000` examples, and a test set of `10,000` examples. 
The digits have been size-normalized and centered in a fixed-size image.
It is a good database for people who want to try learning techniques and pattern recognition methods on real-world data while spending minimal efforts on preprocessing and formatting. The dataset is available at:    
http://yann.lecun.com/exdb/mnist/

## Convolution Neural Network detailed arangement:
Five layers have been applied on the dataset, namely:  
-	`ConvLayer-1`  
-	`ConvLayer-2`  

-	`Inception Layer`  
-	`ConvLayer-3`
-	`Max-pool layers`
-	`Fully Connected Neural Nets`
 
-Pictorial Representation of the model: 

![Untitled1](https://user-images.githubusercontent.com/89198752/153747890-12326eb3-7f2d-4598-a2d6-9d9ffce30b3a.png)
![Untitled 2](https://user-images.githubusercontent.com/89198752/153747896-75683db8-a1b5-4da8-a850-a2756378f265.png)
![Untitled 3](https://user-images.githubusercontent.com/89198752/153747903-aa12e842-1402-4816-8aab-4cb81f1c817f.png)
![Untitled 4](https://user-images.githubusercontent.com/89198752/153747912-4236f5e0-0ed6-4a85-ba19-a89731107f32.png)
![Untitled 5](https://user-images.githubusercontent.com/89198752/153747915-33bc114a-a986-4eab-9561-f869c805d6e4.png)



## Classes of Division
In this project, we have used the predefined classes ranging from `0-9`


## Train-Validation Learning Curve
Train-Validation Curve is a popular method to helps us confirm normal behavioural characteristics of model over increasing number of epochs 
 
The model has been trained over `20` epochs with batch_size of `100`
-     Model
     ![image](https://user-images.githubusercontent.com/89198752/153714067-0e58018d-b6ea-4a58-a245-304ad5395d92.png)

## Dependencies
Since the entire project is based on `Python` programming language, it is necessary to have Python installed in the system. It is recommended to use Python with version `>=3.6`.
The Python packages which are in use in this project are  `matplotlib`, `numpy`, `torch` and `torchvision`. All these dependencies can be installed just by the following command line argument
- pip install `requirements.txt`
        
## Run the following for training and validation :
  
   `python main.py`
      
