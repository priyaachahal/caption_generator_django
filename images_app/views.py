from django.shortcuts import render

# we always send the response on http in Django, since it is a web application
from django.http import HttpResponse,JsonResponse
#JsonResponse sends the json response
# Create your views here.
from django.core.files.storage import FileSystemStorage #This helps in saving the input file into fileSystemStorage
#from keras.models import load_model
from tensorflow.keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow import Graph, Session
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model

model_graph = Graph()
with model_graph.as_default():
    tf_session = Session()
    with tf_session.as_default():
        model = load_model('./models/model_inception_ep007_acc0.325_loss3.153_val_acc0.315_val_loss3.704.h5')
        inception_model = load_model('./models/inception_model.h5')
        #features = np.load('./models/test_image_extracted_inception.npy',allow_pickle='True').item()
        tokenizer = np.load('./models/tokenizer.npy',allow_pickle='True').item()
        index_word = dict([(index,word) for word, index in tokenizer.word_index.items()])
        maximum_length = 34

# example of an HttpResponse to a request comning from urls.py
def index(requests):
    context = {'a':'HelloWorld'}
    return render(requests,'index.html',context)

def predictCaption(requests):
    fileObj = requests.FILES['filePath']
    fs = FileSystemStorage()
    filePathName = fs.save(fileObj.name, fileObj)
    filePathName = fs.url(filePathName)
    filename = fileObj.name
    # function to extract image features

    def extract_features(model,image_name):
        # Load the image
        target_size = (299,299,3)
        image = load_img(image_name, target_size=target_size)
        # Convert to an array using Keras
        image_array = img_to_array(image)
        features_array = preprocess_input(image_array)
        features_array = np.expand_dims(features_array, axis=0)
        with model_graph.as_default():
            with tf_session.as_default():
                feature_vector = model.predict(features_array)
        return (np.reshape(feature_vector, feature_vector.shape[1]))

    # Generate the description for test data based on best-score model
    def beam_search_predict(model, tokenizer, features, maximum_length,beam_index):

        in_text = [[tokenizer.texts_to_sequences(['startseq'])[0], 0.0]]

        while len(in_text[0][0]) < maximum_length:
            templist=[]
            # The description shouldn't exceed the maximum length
            for seq in in_text:
                # pad input
                padded_seq = pad_sequences([seq[0]], maxlen=maximum_length)

                # predict next word
                features1 = features.reshape(1,len(features))
                #print(features.shape)
                with model_graph.as_default():
                    with tf_session.as_default():
                        preds = model.predict([features1, padded_seq], verbose=0)

                # Top predictions
                top_preds =  np.argsort(preds[0])[-beam_index:]

                for word in top_preds:
                    next_seq, prob = seq[0][:], seq[1]
                    next_seq.append(word)
                    #update probability
                    prob += preds[0][word]
                    # append as input for generating the next word
                    templist.append([next_seq, prob])

            in_text = templist
            # Sorting according to the probabilities
            in_text = sorted(in_text, reverse=False, key=lambda l: l[1])
            # Take the top words
            in_text = in_text[-beam_index:]
        in_text = in_text[-1][0]
        final_caption_raw = [index_word[i] for i in in_text]
        final_caption = []
        for word in final_caption_raw:
            if word=='endseq':
                break
            else:
                final_caption.append(word)
        final_caption.append('endseq')
        return ' '.join(final_caption)

    # Generate caption based on greedy search
    def generate_description(model, tokenizer, features, maximum_length):
        input_word = 'startseq'
        for i in range(maximum_length):
             # Get the integer code of input_word
            sequence = tokenizer.texts_to_sequences([input_word])[0]
            sequence = pad_sequences([sequence], maxlen=maximum_length)
            features1 = features.reshape(1,len(features))
            with model_graph.as_default():
                with tf_session.as_default():
                    predicted_word = model.predict([features1, sequence])

            predicted_word =  np.argmax(predicted_word)

            word = index_word[predicted_word]
            #Append as input for generating the next word
            input_word += ' ' + word
            # stop if we predict the end of sequence
            if word == 'endseq':
                break

        return input_word

    features = extract_features(inception_model,fileObj)

    final_caption = beam_search_predict(model, tokenizer, features, maximum_length,3)
    final_caption = final_caption.replace("startseq ","").replace(" endseq","")

    input_word = generate_description(model, tokenizer, features, maximum_length)
    input_word = input_word.replace("startseq ","").replace(" endseq","")

    context = {'filePathName':filePathName,'predictedCaption':input_word,'beamSearch':final_caption,'imageName':filename}
    return render(requests,'index.html',context)

def viewDataBase(requests):
    import os
    listOfImages=[f for f in os.listdir('./media') if ".jpg" in f]
    listOfImagesPath= ['./media/'+i for i in listOfImages]
    context = {'listOfImagesPath':listOfImagesPath}
    return render(requests,'viewDB.html',context)
