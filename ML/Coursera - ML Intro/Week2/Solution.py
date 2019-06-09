
import torch

###
# Methods
###
def computeCost(x, y, theta):
    m = len(y)
    J = 0

    h = torch.mm(x, theta) - y
    J = (1 / (2 * m)) * torch.sum(h * h)
    return J

def gradDescent(x, y, theta, alpha, num_iters):
    m = len(y)

    for i in range(num_iters):
        sum1 = torch.sum(torch.mm(x, theta) - y)
        sum2 = torch.sum((torch.mm(x, theta) - y) * torch.index_select(x, 1, torch.tensor([1])))
        
        theta[0][0] = theta[0][0] - alpha * (1 / m) * sum1
        theta[1][0] = theta[1][0] - alpha * (1 / m) * sum2

    return theta


#############
# Read data #
#############
a = []
b = []
with open("ex1/ex1data1.txt", "r") as ins:
    for line in ins:
        a.append(line.split(',')[0])
        b.append(line.split(',')[1])

# add a column of ones to x
X = torch.ones(len(a), 2)
y = torch.zeros(len(a), 1)

for i in range(len(a)):
    X[i][1] = float(a[i])
    y[i] = float(b[i])


################
# Compute cost #
################
theta = torch.zeros(2, 1)

J = computeCost(X, y, theta)
print("Expected: 32.07\nActual: ", J) 


############
# Gradient #
############

iter = 1500
alpha = 0.01
theta = gradDescent(X, y, theta, alpha, iter)
print("Expected theta: [[-3.6303], [1.1664]]\nActual theta: ", theta)

#########
# test case for compute cost 
# expected J: 11.9450
t1 = torch.tensor([[1, 2], [1, 3], [1,4], [1, 5]], dtype=torch.float)
t2 = torch.tensor([[7], [6], [5], [4]], dtype=torch.float)
t3 = torch.tensor([[0.1], [0.2]])
#J = computeCost(t1, t2, t3)

# test case for gradient
# expected theta: [[5.2148] , [-0.5733]]
t1 = torch.tensor([[1, 5], [1, 2], [1, 4], [1, 5]], dtype=torch.float)
t2 = torch.tensor([[1], [6], [4], [2]], dtype=torch.float)
t3 = torch.tensor([[0], [0]], dtype=torch.float)
t4 = torch.tensor(0.01)
#gradDescent(t1, t2, t3, t4, 1000)