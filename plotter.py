# Plot the results from .csv files
####################################
import matplotlib.pyplot as plt
import numpy as np

csv1 = np.genfromtxt('mlp1dp_lr01_bs100_std377.csv', delimiter=",")
# csv1 = csv1[1:]

epochs1 = csv1[:,0]
training_loss1 = csv1[:,1]
validation_loss1 = csv1[:,2]



# csv2 = np.genfromtxt('mlp1ed_bs200.csv', delimiter=",")
# # csv2 = csv2[1:]
# epochs2 = csv2[:,0]
# training_loss2 = csv2[:,1]
# validation_loss2 = csv2[:,2]


x1 = epochs1
tl1 = training_loss1
vl1 = validation_loss1

# x2 = epochs2
#
# tl2 = training_loss2
# vl2 = validation_loss2


# fig, ax = plt.subplots()
# line_1, = ax.plot(-t, label='Inline label')
# line_2, = ax.plot(-s, 'r-', label='Inline label')
# # Overwrite the label by calling the method.
# line_1.set_label('validation set')
# line_2.set_label('training set')
# ax.legend()
# ax.set_ylabel('normalized negative log-likelihood')
# ax.set_title('')
# ax.set_xlabel('Iterations')


fig, ax = plt.subplots()
line_1, = ax.plot(tl1, label='Inline label')
line_2, = ax.plot(vl1, label='Inline label')
# line_3, = ax.plot(tl1*100*0.8, label='Inline label')
# line_4, = ax.plot(vl2, label='Inline label')
# Overwrite the label by calling the method.
line_1.set_label('training-loss')
line_2.set_label('validation-loss')
# line_3.set_label('Augmented-train-loss')
# line_4.set_label('Augmented-valid-loss ')
ax.legend()
ax.set_ylabel('0-1 loss %')
ax.set_title('0/1 loss, noise level High, epsilon = 1')
ax.set_xlabel('epochs')
plt.axis([0, 100, 0, 20])
plt.show()