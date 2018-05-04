# source: http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html
# edited for ML4_programming by Yuri Wu, 2017-04-23
# for SVM visualization

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from sklearn import svm

# output as 'figure.pdf'
output = PdfPages('figure.pdf')

# import some data to play with
X=np.genfromtxt('demo_data.csv',delimiter=',')
y=np.genfromtxt('demo_targets.csv')

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
demo_list=[svm.SVC(kernel='linear', C=1).fit(X, y),
           svm.SVC(kernel='poly', degree=3, C=1).fit(X, y),
           svm.SVC(kernel='rbf', gamma=10, C=1).fit(X, y),
           svm.SVC(kernel='rbf', gamma=10, C=10).fit(X, y),
           svm.SVC(kernel='rbf', gamma=100, C=1).fit(X, y),
           svm.SVC(kernel='rbf', gamma=100, C=10).fit(X, y)]

# title for the plots
titles = ['linear kernel',
          'polynomial (degree 3) kernel',
          'RBF kernel, gamma=10, C=1',
          'RBF kernel, gamma=10, C=10',
          'RBF kernel, gamma=100, C=1',
          'RBF kernel, gamma=100, C=10']

# create a mesh to plot in
h = .005  # step size in the mesh
x_min, x_max = 0,1
y_min, y_max = 0,1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


for i, clf in enumerate(demo_list):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
	plt.subplot(3, 2, i + 1)
	plt.subplots_adjust(wspace=0.4, hspace=0.4)

	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) 

    # Put the result into a color plot
	Z = Z.reshape(xx.shape) 

	plt.contourf(xx, yy, Z, linestyles=['--', '-', '--'],
					cmap=plt.cm.coolwarm,alpha=0.5) 

	# Get mark of support vectors
	mark_sv = []
	for index in clf.support_:
		mark_sv.append(y[index]) 

	# Get non_support vectors
	nonSVX = []
	nonSVY = []
	nonSVMark = []
	pos = 0
	for p1, p2 in X:  
		if not p1 in clf.support_vectors_[:, 0] and not p2 in clf.support_vectors_[:, 1]:
			nonSVX.append(p1)
			nonSVY.append(p2)
			nonSVMark.append(y[pos]) 
		pos += 1

    # Plot also the training points
	svp = plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], c=mark_sv, marker='*', cmap=plt.cm.coolwarm)
	nonsvp = plt.scatter(nonSVX, nonSVY, c=nonSVMark, cmap=plt.cm.coolwarm)
	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())
	plt.xticks(())
	plt.yticks(())
	plt.title(titles[i])
	print(len(clf.support_)) 

output.savefig()
plt.show()
output.close()
