from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten


class CNN:
    def __init__(self):
        self.numclasses = 1
        self.inputShape = (100, 100 ,3)

def BuildCNN():
    cnn = CNN()
    model = Sequential()
    model.add(Conv2D(32,(3,3),activation='relu',input_shape=cnn.inputShape))
    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))

    model.add(Dense(cnn.numclasses,activation='sigmoid'))

    return model

def main():
    bulidcnn = CNN()

if __name__ == '__main__':
    main()