from tensorflow.keras.preprocessing.image import ImageDataGenerator
from BuildCNN.BuildCNN import BuildCNN


class Train:
    def __init__(self):
        self.BatchSize = 32
        self.inputShape = (100, 100)
        self.classMode = 'binary'
        self.path = '../Images'
        self.epochs = 10
        self.optimizer = 'adam'
        self.loss = 'binary_crossentropy'

    def loadimages(self):
        datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
        train_data = datagen.flow_from_directory(self.path, \
                                                      batch_size=self.BatchSize, \
                                                      target_size=self.inputShape, \
                                                      class_mode=self.classMode, \
                                                      subset='training')

        valid_data = datagen.flow_from_directory(self.path, \
                                                      batch_size=self.BatchSize, \
                                                      target_size=self.inputShape, \
                                                      class_mode=self.classMode, \
                                                      subset='validation')

        return train_data, valid_data

    def getmodel(self):
        model = BuildCNN()
        return model

    def trainCNN(self):

        train_data, valid_data = self.loadimages()
        model = self.getmodel()
        model.compile(optimizer=self.optimizer,loss=self.loss,metrics=['accuracy'])
        print(model.summary())
        model.fit(train_data,\
                  validation_data = valid_data,\
                  epochs = self.epochs)

        # ans = model.evaulate(valid_data)
        # print("validation accuracy = {}%".format(ans[1]*100))
        model.save('model.h5')


def main():
    train = Train()
    train.trainCNN()

if __name__ == "__main__":
    main()