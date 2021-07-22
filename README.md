# ImageMining
Image Mining Food recognition repository. This repository contains the code (in Jupyter notebooks) of the Image Mining course at UPM. The objective was to correctly classify different types of food from an input image and provide nutritional information about it. This was done in an Android application, which was connected to an API. For more detailed information, please check out the [paper](https://raw.githubusercontent.com/ESQ0001/ESQ0001.github.io/master/IM.pdf). \
Authors:
- Alejandro Esquivias (principal contributor and in charge of the API and classification code)
- Ákos Kuti (Initial data manipulation)
- Przemyslaw Lewandoski (Android application)
- Jiachun Chen (Conceptual framing)
- David Engelstein (Error Analysis)
- Ismael Sánchez (API and Android integration)

### Data

The image data was extracted from [ImageNet](https://www.image-net.org/) and  the nutritional information was extracted from [Nutritionix](https://www.nutritionix.com/). There was some data manipulation to incorporate the nutritional values in an Excel file.

### Results

After trying different transfer learning algorithms and pixel classification, we obtained the best result using the ResNet50 algorithm with an SGD optimizer with an accuracy of 71.5% in the test set. We also analyzed the errors behind the misclassifications. The main reasons were: \
- Non-food images
- Too much information in the image
- Similar format, which leads to misclassification, due to the pixels' resemblance
- More than one food item in the image
- Backgrounds too big compared to the food

### Improvements


For future work we suggest 
segmenting the images. In that way we would obtain only 
the desired section of the image, since one of our problem 
was our noisy images.  
 
Other transfer learning algorithms could also be  tried, and 
transfer  learning  in itself  could  be  used to  extract  features 
from  the  images  and  then  training  a  classifier  (SVM, 
decision trees, k-nearest-neighbor...etc).  
 
A  comparison  between  different  types  of  classifiers  is 
interesting for future iterations. Also, once a better 
accuracy  is  reached  it  could  be  useful  to  train  a  classifier 
on food vs non-food objects to prevent it from classifying 
pictures  that  are  not  food.  This  is  especially  useful,  if  we 
want  to  deploy  this  application  to  reduce  the  amount  of 
computational  power  needed,  when  calling  the  API  on  a 
remote server.


