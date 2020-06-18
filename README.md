# Pneumonia_Detection
According to https://www.healthline.com/health/pneumonia#is-it-contagious?
Pneumonia is an infection in one or both lungs. Bacteria, viruses, and fungi cause it.
This infection causes inflammation in the air sacs in your lungs, which are called alveoli. The alveoli fill with fluid or pus, making it difficult to breathe.
The germs that cause pneumonia are contagious. This means they can spread from person to person. Pneumonia can be Bacterial, Viral, Fungal.
Anyone can get pneumonia from 2 years old to people ages 65 years and older
people who smoke, use certain types of drugs, or drink excessive amounts of alcohol and many others.
So basically this disease is related to the lungs and the people who have mild systems must gone through the check-up. AI provide us that way to detect the people having pneumonia.
The model is trained on lots of lung X-ray images of several people, some of them have pneumonia and some are normal(Download dataset from https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).
So when a person walks for the check up the X-ray image of his lungs must be passed through the classifier and it will detect weather the person is having pneumonia or not?
This classifier is built on CNN(optimizer = 'adam') with EarlyStopping and have good accuracy in detecting the disease.
Go check it out by yourself too.
![alt text](https://github.com/shalom217/Pneumonia_Detection/blob/master/Normal.png)
![alt text](https://github.com/shalom217/Pneumonia_Detection/blob/master/pneumonia1.png)
