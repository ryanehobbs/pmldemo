"""Main"""

import data.loader as data
import preprocessing
import models.linear as linear


def demo_preprocessing():

    X, y = data.load_data(source='samples/sample_data4.txt', columns=3, features=[1, 2])
    preprocessor = preprocessing.PolyFeatures(degrees=2)
    preprocessor.mapfeatures(X)
    print("done")

def demo_newmultilinear():

    X, y = data.load_data(source='samples/sample_data2.txt', columns=3, features=[1,2])
    linear_model = linear.Linear(solver='linear', normalize=True, iterations=1500, alpha=0.01)
    linear_model.fit(X, y)
    price1 = linear_model.predict([1650, 3])
    print("A 1650 sq. foot home with 3 bedrooms, we predict its price  to be ${}".format(price1))
    price2 = linear_model.predict([4000, 4])
    print("A 4000 sq. foot home with 4 bedrooms, we predict its price  to be ${}".format(price2))


def demo_newlinear():

    X, y = data.load_data(source='samples/sample_data.txt', columns=2, features=1)
    linear_model = linear.Linear(solver='linear', normalize=False, iterations=1500, alpha=0.01)
    linear_model.fit(X, y)
    price1 = linear_model.predict([3.5])*10000
    print("For a population of 35,000 people, we predict a profit of ${}".format(price1))
    price2 = linear_model.predict([7])*10000
    print("For a population of 70,000 people, we predict a profit of ${}".format(price2))

def demo_newlogistic():

    X, y = data.load_data(source='samples/sample_data3.txt', columns=3, features=[1,2])
    logistic_model = linear.Logistic(solver='logistic', normalize=False, iterations=400, alpha=0.01)
    logistic_model.fit(X, y)
    print("done")

def demo_logisticreg():

    X, y = data.load_data(source='samples/sample_data4.txt', columns=3, features=[1,2])
    logistic_model = linear.Logistic(solver='logistic', normalize=False, iterations=400, alpha=0.01, lambda_r=1)
    # map polynomial features
    preprocessor = preprocessing.PolyFeatures(degrees=6)
    X = preprocessor.mapfeatures(X)
    logistic_model.fit(X, y)
    quality = logistic_model.predict([2, 3])
    print("done")



if __name__ == '__main__':
    """Main function for console application"""

    # plot classification data
    demo_preprocessing()
    demo_newmultilinear()
    demo_newlinear()
    demo_newlogistic()
    demo_logisticreg()

