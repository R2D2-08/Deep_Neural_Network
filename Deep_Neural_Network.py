#3 hidden layers with 16 neurons each, 1 output layer with 1 neuron, 10 input neurons, use the actual backpropagation algorithm
import torch
import numpy as np
def dataset(val,data1,data2):
    if val<100 : l1=torch.tensor([i for i in data1[val]])
    elif val>=100 : l1=torch.tensor([i for i in data2[val-100]])
    return l1
def describe_neural_network(l1,l2,l3,l4,l5,w1,w2,w3,w4,b1,b2,b3,b4):
    print(f"Layer 1 : {l1.numpy()}\nLayer 2 : {l2.numpy()}\n Layer 3 : {l3.numpy()}\nLayer 4 : {l4.numpy()}\nLayer 5 : {l5.numpy()}\nW1 : {w1.numpy()}\nW2 : {w2.numpy()}\nW3 : {w3.numpy()}\nW4 : {w4.numpy()}\nB1 : {b1.numpy()}\nB2 : {b2.numpy()}\nB3 : {b3.numpy()}\nB4 : {b4.numpy()}")
def sigmoid(x):
    return 1/(1+np.exp(-1*x))
def initialize_network():
    l1=torch.ones(size=(1,10),dtype=float)
    w1=torch.tensor(np.random.uniform(-1,1.,size=(10,16)))
    b1=torch.tensor(np.random.uniform(-1,1.,size=(1,16)))
    l2=torch.tensor(np.random.uniform(-1,1.,size=(1,16)))
    w2=torch.tensor(np.random.uniform(-1,1.,size=(16,16)))
    b2=torch.tensor(np.random.uniform(-1,1.,size=(1,16)))
    l3=torch.tensor(np.random.uniform(-1,1.,size=(1,16)))
    w3=torch.tensor(np.random.uniform(-1,1.,size=(16,16)))
    b3=torch.tensor(np.random.uniform(-1,1.,size=(1,16)))
    l4=torch.tensor(np.random.uniform(-1,1.,size=(1,16)))
    w4=torch.tensor(np.random.uniform(-1,1.,size=(16,1)))
    b4=torch.tensor(np.random.uniform(-1,1.,size=(1,1)))
    l5=torch.zeros(size=(1,1),dtype=float)
    return l1,l2,l3,l4,l5,w1,w2,w3,w4,b1,b2,b3,b4
def forward_pass(l1,l2,l3,l4,l5,w1,w2,w3,w4,b1,b2,b3,b4):
    l2=sigmoid((b1+torch.matmul(l1,w1)))
    l3=sigmoid((b2+torch.matmul(l2,w2)))
    l4=sigmoid((b3+torch.matmul(l3,w3)))
    l5=sigmoid((b4+torch.matmul(l4,w4)))
    return l2,l3,l4,l5
def backpropagation_and_update_params(l1,l2,l3,l4,l5,w1,w2,w3,w4,b1,b2,b3,b4,epoch):
    if epoch>=100 : y=1
    else : y=0
    lr=0.0001
    C_dash=sigmoid((torch.matmul(torch.matmul((torch.matmul(torch.matmul(l1,w1)+b1,w2)+b2),w3)+b3,w4)+b4)-y)
    w4=w4-lr*sigmoid(torch.matmul(torch.matmul((torch.matmul(b1+torch.matmul(l1,w1),w2)+b2),w3)+b3,torch.ones(np.shape(w4),dtype=float)))*(2*C_dash)
    b4=b4-lr*(2*C_dash)
    w3=w3-lr*sigmoid(torch.matmul(torch.matmul(torch.matmul(b1+torch.matmul(l1,w1),w2)+b2,torch.ones(np.shape(w3),dtype=float)),w4))*(2*C_dash)
    b3=b3-lr*sigmoid(torch.matmul(torch.ones(np.shape(b3),dtype=float),w4))*(2*C_dash)
    w2=w2-lr*sigmoid(torch.matmul(torch.matmul(torch.matmul(b1+torch.matmul(l1,w1),torch.ones(np.shape(w2),dtype=float)),w3),w4))*(2*C_dash)
    b2=b2-lr*sigmoid(torch.matmul(torch.matmul(torch.ones(np.shape(b2),dtype=float),w3),w4))*(2*C_dash)
    w1=w1-lr*sigmoid(torch.matmul(torch.matmul(torch.matmul(torch.matmul(l1,torch.ones(np.shape(w1),dtype=float)),w2),w3),w4))*(2*C_dash)
    b1=b1-lr*sigmoid(torch.matmul(torch.matmul(torch.matmul(torch.ones(np.shape(b1),dtype=float),w2),w3),w4))*(2*C_dash)
    return l2,l3,l4,l5,w1,w2,w3,w4,b1,b2,b3,b4
def learn():
    data1=np.random.uniform(0,.5,size=(100,10))
    data2=np.random.uniform(.5,1,size=(100,10))
    l1,l2,l3,l4,l5,w1,w2,w3,w4,b1,b2,b3,b4=initialize_network()
    epochs=200
    for epoch in range(epochs):
        l1=dataset(epoch,data1,data2)
        l2,l3,l4,l5=forward_pass(l1,l2,l3,l4,l5,w1,w2,w3,w4,b1,b2,b3,b4)
        l2,l3,l4,l5,w1,w2,w3,w4,b1,b2,b3,b4=backpropagation_and_update_params(l1,l2,l3,l4,l5,w1,w2,w3,w4,b1,b2,b3,b4,epoch)
    return l1,l2,l3,l4,l5,w1,w2,w3,w4,b1,b2,b3,b4
torch.set_printoptions(precision=10)
l1,l2,l3,l4,l5,w1,w2,w3,w4,b1,b2,b3,b4=learn()
describe_neural_network(l1,l2,l3,l4,l5,w1,w2,w3,w4,b1,b2,b3,b4)