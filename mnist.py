import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(20,40, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))

        self.linear1 = nn.Linear(4*4*40, 100)
        self.linear2 = nn.Linear(100,10)
        self.act=nn.Sigmoid()
    def forward(self,x):
        x=self.conv2(self.conv1(x))

        x=x.view(-1,4*4*40)
        return self.act(self.linear2(self.act(self.linear1(x))))



model=Net()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs=10000
criterion = nn.BCELoss()
for epoch in range(epochs):
    idx = np.random.choice(mnist.train.images.shape[0], 100)
    img=mnist.train.images[idx]
    lbl=mnist.train.labels[idx]
    img=torch.FloatTensor(img)
    img=img.view(-1,1,28,28)
    pred=model(img)

    lbl=torch.FloatTensor(lbl)
    loss = criterion(pred,lbl)
    optimizer.zero_grad()
    print(loss)
    loss.backward()
    optimizer.step()
    total = lbl.size(0)
    _, predicted = torch.max(pred.data, 1)
    _, true = torch.max(lbl.data,1)
    correct = (predicted == true).sum().item()

    if(epoch + 1) % 100 == 0:
        print('Epoch [{}/{}],  Loss: {:.4f}, Accuracy: {:.2f}%'
            .format(epoch + 1, epochs,loss.item(),
                    (correct / total) * 100))

# model.eval()
print(len(mnist.test.labels))
with torch.no_grad():
    correct = 0
    
    for images, labels in zip(torch.FloatTensor(mnist.test.images),torch.FloatTensor(mnist.test.labels)):
        images = images.view(-1, 1, 28, 28)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)


        _,lbl=torch.max(labels.data,0)


        correct += (predicted == lbl).sum().item()
    print((correct)/len(mnist.test.labels))

# Save the model and plot
# torch.save(model.state_dict(), MODEL_STORE_PATH + 'conv_net_model.ckpt')