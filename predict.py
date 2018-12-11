from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import numpy
import pandas as pd
# fix random seed for reproducibility
numpy.random.seed(7)

# load dataset
df = pd.read_csv("4_seasons_combined.csv")
train = df[df['tourny'] == 0]
test = df[df['tourny'] == 1]
train.to_csv("train_1_season.csv", header=False)
test.to_csv("test_1_season.csv", header=False)
train_dataset = numpy.loadtxt("train_1_season.csv", delimiter=",")

X_train = train_dataset[:,6:84]#5:83]
Y_train = train_dataset[:,84]#83]

test_dataset = numpy.loadtxt("test_1_season.csv", delimiter=",")

X_test = test_dataset[:,6:84]
Y_test = test_dataset[:,84]

model = Sequential()
model.add(Dense(128, input_dim=78, activation='tanh'))
model.add(Dense(60, activation='relu'))
model.add(Dense(30, activation='elu'))
model.add(Dense(15, activation='elu'))
model.add(Dense(1, activation='tanh'))

model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=150, batch_size=10)

scores = model.evaluate(X_test, Y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

scores = model.evaluate(X_train, Y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
