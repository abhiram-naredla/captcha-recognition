from django.shortcuts import render, HttpResponse
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings
from django.core.files.storage import FileSystemStorage

# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})

def DatasetView(request):
    return render(request, 'users/viewdataset.html', {})

def training(request):
    #importing libraries
    import numpy as np 

    #%matplotlib inline 
    #to use as command line calls #using inline graphs will come next to code

    import matplotlib.pyplot as plt #for graphs
    import os #for operating system dependent fucntionality
    from keras import layers #for building layers of neural net
    from keras.models import Model
    from keras.models import load_model
    from keras import callbacks #for training logs, saving to disk periodically
    import cv2 #OpenCV(Open Source computer vision lib), containg CV algos
    import string

    os.listdir(r"media\samples")

    n=len(os.listdir(r"media\samples"))
    n

    imgshape=(50,200,1) #50-height, 200-width, 1-no of channels

    character= string.ascii_lowercase + "0123456789" # All symbols captcha can contain
    nchar = len(character) #total number of char possible
    nchar

    #preprocesss image
    def preprocess():
        X = np.zeros((n,50,200,1)) #1070*50*200 array with all entries 0
        y = np.zeros((5,n,nchar)) #5*1070*36(5 letters in captcha) with all entries 0
        path=os.path.join(settings.MEDIA_ROOT,'samples')
        for i, pic in enumerate(os.listdir(path)):
        #i represents index no. of image in directory 
        #pic contains the file name of the particular image to be preprocessed at a time
            
            img = cv2.imread(os.path.join(path, pic), cv2.IMREAD_GRAYSCALE) #Read image in grayscale format
            pic_target = pic[:-4]#this drops the .png extension from file name and contains only the captcha for training
            
            if len(pic_target) < 6: #captcha is not more than 5 letters
                img = img / 255.0 #scales the image between 0 and 1
                img = np.reshape(img, (50, 200, 1)) #reshapes image to width 200 , height 50 ,channel 1 

                target=np.zeros((5,nchar)) #creates an array of size 5*36 with all entries 0

                for j, k in enumerate(pic_target):
                #j iterates from 0 to 4(5 letters in captcha)
                #k denotes the letter in captcha which is to be scanned
                    index = character.find(k) #index stores the position of letter k of captcha in the character string
                    target[j, index] = 1 #replaces 0 with 1 in the target array at the position of the letter in captcha

                X[i] = img #stores all the images
                y[:,i] = target #stores all the info about the letters in captcha of all images

        return X,y

    def createmodel():
        img = layers.Input(shape=imgshape) # Get image as an input of size 50,200,1
        conv1 = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(img) #50*200
        mp1 = layers.MaxPooling2D(padding='same')(conv1)  # 25*100
        conv2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(mp1)
        mp2 = layers.MaxPooling2D(padding='same')(conv2)  # 13*50
        conv3 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(mp2)
        bn = layers.BatchNormalization()(conv3) #to improve the stability of model
        mp3 = layers.MaxPooling2D(padding='same')(bn)  # 7*25
        
        flat = layers.Flatten()(mp3) #convert the layer into 1-D

        outs = []
        for _ in range(5): #for 5 letters of captcha
            dens1 = layers.Dense(64, activation='relu')(flat)
            drop = layers.Dropout(0.5)(dens1) #drops 0.5 fraction of nodes
            res = layers.Dense(nchar, activation='sigmoid')(drop)

            outs.append(res) #result of layers
        
        # Compile model and return it
        model = Model(img, outs) #create model
        model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=["accuracy"]*5)
        return model

    #Create model
    model=createmodel();
    model.summary();
    X,y=preprocess()
    #split the 1070 samples where 970 samples will be used for training purpose
    X_train, y_train = X[:970], y[:, :970]
    X_test, y_test = X[970:], y[:, 970:]

    #Applying the model
    hist = model.fit(X_train, [y_train[0], y_train[1], y_train[2], y_train[3], y_train[4]], batch_size=32, epochs=60, validation_split=0.2)
    #batch size- 32 defines no. of samples per gradient update
    #Validation split=0.2 splits the training set in 80-20% for training nd testing

    model.save("model.h5")

    #graph of loss vs epochs
    for label in ["loss"]:
        plt.plot(hist.history[label],label=label)
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    #graph of accuracy of dense_2 vs epochs
    for label in ["dense_1_accuracy"]:
        plt.plot(hist.history[label],label=label)
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy of Dense 2 layer")
    plt.show()

    #graph of accuracy of dense_4 vs epochs
    for label in ["dense_3_accuracy"]:
        plt.plot(hist.history[label],label=label)
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy of Dense 4 layer")
    plt.show()

    #graph of accuracy of dense_6 vs epochs
    for label in ["dense_5_accuracy"]:
        plt.plot(hist.history[label],label=label)
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy of Dense 6 layer")
    plt.show()

    #graph of accuracy of dense_8 vs epochs
    for label in ["dense_7_accuracy"]:
        plt.plot(hist.history[label],label=label)
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy of Dense 8 layer")
    plt.show()

    #graph of accuracy of dense_10 vs epochs
    for label in ["dense_9_accuracy"]:
        plt.plot(hist.history[label],label=label)
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy of Dense 10 layer")
    plt.show()

    #Loss on training set
    #Finding Loss on training set
    # preds = model.evaluate(X_train, [y_train[0], y_train[1], y_train[2], y_train[3], y_train[4]])
    # print ("Loss on training set= " + str(preds[0]))

    # #Finding loss on test set
    # preds = model.evaluate(X_test, [y_test[0], y_test[1], y_test[2], y_test[3], y_test[4]])
    # print ("Loss on testing set= " + str(preds[0]))

    final_accuracy = hist.history['dense_9_accuracy'][-1]
    final_loss = hist.history['loss'][-1]

    return render(request, "users/training.html", {"final_accuracy": final_accuracy, "final_loss": final_loss})

#to predict captcha
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import cv2
import numpy as np
from keras.models import load_model
import string
import os

# Assuming character is defined somewhere globally
character = string.ascii_lowercase + "0123456789"

def predict(request):
    if request.method == 'POST':
        # Get the uploaded file
        image_file = request.FILES['file']

        # Save the file to the server
        fs = FileSystemStorage(location="media/test_data")
        filename = fs.save(image_file.name, image_file)
        uploaded_file_url = fs.url(filename)
        filepath = fs.path(filename)

        # Load the model
        model_path = os.path.join(settings.MEDIA_ROOT, 'model.h5')
        model = load_model(model_path)

        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

        if img is not None:
            img = img / 255.0
            img = np.reshape(img, (1, 50, 200, 1))
        else:
            print("Image not detected")
            return render(request, 'users/UploadForm.html', {'error': 'Image not detected'})

        # Make predictions
        result = model.predict(img)
        k_ind = [np.argmax(i) for i in result]

        # Decode predictions into captcha text
        capt = ''.join([character[k] for k in k_ind])

        return render(request, 'users/UploadForm.html', {"capt": capt, 'path': uploaded_file_url})

    return render(request, 'users/UploadForm.html')


