from matplotlib import pyplot as plt
import numpy as np


# Generate synthetic data
def generate_data(num_samples=30):
    X = np.array(range(num_samples)) # create the number of points
    random_noise = np.random.uniform(10, 40, size=num_samples) # generate noise from a uniform distribution with mean=10 and std=40
    y = 3.5*X + random_noise # y will be 3.5 * X + the noise 
    return X, y



"""
features, target=generate_data(30)

plt.scatter(features, target)
plt.show()

print(features.shape)

"""

def mse(y_pred, y_gr):
    return np.mean((y_pred-y_gr)**2)

if __name__ =="__main__":
    features, target=generate_data(30)

    #Learning rate
    lr=0.003

    #Number of epochs
    epochs=500

    #Initializam parametrii ht(x)
    t0, t1=1.2,3.4

    dataset=[(x,y) for x,y in zip(features, target)]

    errors = []

    errors_per_epoch=[]
   # print(dataset)
    for epoch in range(epochs):
       errors=[]
       for x, y_gr in dataset:
           #Predictia
           y_pred=t1*x+t0 
           
           #Masuram eroarea
           error=mse(y_pred, y_gr)
           errors.append(error)

           #Gradient descend

           #1. Calcul derivate
           t1_grad=(y_pred-y_gr)*x
           t0_grad=y_pred-y_gr

           #2.Update parametri
           t0=t0-lr*t0_grad
           t1=t1-lr*t1_grad

       errors_per_epoch.append(np.mean(errors))

plt.plot(errors_per_epoch)
plt.show()


