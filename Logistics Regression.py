import numpy as np 
#features
a=[1,2,3,4,5,6,7]
b=[4,6,7,4,3,1,8]
#labels
y1=[0,1,0,1,0,1,1]

#converting it into numpy arrays
x1=np.array(a).reshape(1,len(a))
x2=np.array(b).reshape(1,len(b))
y=np.array(y1).reshape(1,len(y1))

#initializing weights
w=np.random.randn(3).reshape(1,3)
def sigmoid(x):
	return 1/(1+np.exp(-x))

def train(x,y,alpha,m):
	global w
	prediction=np.matmul(w,x)
	final=sigmoid(prediction)
	error=final-y

	gradient=(alpha/m)*np.matmul(error,x.T) 
	w=w-gradient

def CrossEntropy(x, y,m):
	prediction=np.matmul(w,x)
	final=sigmoid(prediction)
	return np.sum(y*np.log(final)+(1-y)*np.log(1-final))*(-1/m)
m=y.size
X=np.vstack((np.ones(m).reshape(1,m),x1))
X1=np.vstack((X,x2))

for i in range(1000):
	train(X1,y,0.001,m)
	if(i%100==0):
		l=CrossEntropy(X1,y,m)
		print("Loss:",i,l)
print(w)

def relu(x):
	x[x>=0.5]=1
	x[x<0.5]=0
	return x
total=m
correct=0

#testing accuracy on same dataset
def test(x,y):
	global correct
	prediction=np.matmul(w,x)
	final=sigmoid(prediction)
	te=relu(final)
	for i in range(7):
		if(te[0][i]==y[0][i]):
			correct+=1
test(X1,y)
print("Accuracy=",(correct/total)*100,"%")