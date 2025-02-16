##output of Regression dataset has to be input to below code to generate the 2d

import matplotlib.pyplot as plt

# Plot the first feature vs target
plt.scatter(X[:, 0].numpy(), y.numpy(), alpha=0.5)
plt.xlabel("Feature 1")
plt.ylabel("Target")
plt.title("Synthetic Regression Data")
plt.show()


##output of classification dataset has to be input to below code to generate the 2d



# Plot the 2D features with class labels
plt.scatter(X[:, 0].numpy(), X[:, 1].numpy(), c=y.numpy(), cmap=plt.cm.RdYlBu, alpha=0.7)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Synthetic Classification Data")
plt.colorbar()
plt.show()