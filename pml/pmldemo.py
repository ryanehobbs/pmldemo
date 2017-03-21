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
    linear_model = linear.Linear(solver='linear', normalize=True, alpha=0.01, max_iter=1500)
    linear_model.fit(X, y)
    price1 = linear_model.predict([1650, 3])
    print("A 1650 sq. foot home with 3 bedrooms, we predict its price  to be ${}".format(price1))
    price2 = linear_model.predict([4000, 4])
    print("A 4000 sq. foot home with 4 bedrooms, we predict its price  to be ${}".format(price2))


def demo_newlinear():

    X, y = data.load_data(source='samples/sample_data.txt', columns=2, features=1)
    linear_model = linear.Linear(solver='linear', normalize=False, alpha=0.01, max_iter=1500)
    linear_model.fit(X, y)
    price1 = linear_model.predict([3.5])*10000
    print("For a population of 35,000 people, we predict a profit of ${}".format(price1))
    price2 = linear_model.predict([7])*10000
    print("For a population of 70,000 people, we predict a profit of ${}".format(price2))

def demo_newlogistic():

    X, y = data.load_data(source='samples/sample_data3.txt', columns=3, features=[1,2])
    logistic_model = linear.Logistic(solver='logistic', normalize=False, max_iter=400)
    logistic_model.fit(X, y)
    result = logistic_model.predict([45, 85])
    print("For a student with scores 45 and 85, we predict an admission probability of {0:.2f}%".format(result*100))

def demo_logisticreg():

    X, y = data.load_data(source='samples/sample_data4.txt', columns=3, features=[1,2])
    logistic_model = linear.Logistic(solver='logistic', normalize=False, max_iter=1500, lambda_r=1)
    # map polynomial features
    preprocessor = preprocessing.PolyFeatures(degrees=6)
    X = preprocessor.mapfeatures(X)
    logistic_model.fit(X, y)
    quality = logistic_model.predict([2, 3])
    print("done")

def demo_multiclass():

    import numpy as np
    X, y = data.load_matdata(file_name='samples/ex3data1.mat')
    logistic_model = linear.Logistic(solver='logistic', normalize=False, num_of_labels=10, max_iter=50, lambda_r=3)

    theta_t = [-2, -1, 1, 2]
    X_t = np.arange(1,16).reshape(5,-1, order='F')/10
    y_t = np.array([[1],[0],[1],[0],[1]])
    #logistic_model.fit(X_t, y_t, theta_t)
    logistic_model.fit(X, y, lambda_r=0.1)
    cost = logistic_model.predict(theta_t)

    #fprintf('\nCost: %f\n', J);
    #fprintf('Expected cost: 2.534819\n');
    #fprintf('Gradients:\n');
    #fprintf(' %f \n', grad);
    #fprintf('Expected gradients:\n');
    #fprintf(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n');

if __name__ == '__main__':
    """Main function for console application"""

    # plot classification data
    demo_preprocessing()
    demo_newmultilinear()
    demo_newlinear()
    demo_newlogistic()
    demo_logisticreg()
    demo_multiclass()


