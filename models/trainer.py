### Cast the data to torch format
### First import torch and utils
import torch
import torch.nn as nn
from torchvision.utils import save_image,make_grid
from torch.utils.data import Dataset, DataLoader

class CXRDataset(Dataset):
    def __init__(self,data,target,normalize=True):
        super().__init__()
        self.data = data
        ### Normalize intensities to be between 0-1
        if normalize:
            self.data = self.data/ self.data.max() ##########

        ## Make a tensor target
        self.target = torch.Tensor(target) ########

    def __len__(self):
        ### Method to return number of data points
        return len(self.target)

    def __getitem__(self,index):
        ### Method to fetch indexed element
        return self.data[index], self.target[index].type(torch.FloatTensor)


#### Training and validation loop!
from IPython.display import clear_output

def train_model(model, train_loader, valid_loader, max_epochs=10):
  trLoss = []
  trAcc = []
  vlLoss = []
  vlAcc = []

  # tracker = CarbonTracker(epochs=max_epochs,log_dir='./',log_file_prefix='SAI')
  # tracker.epoch_start()
  for epoch in range(max_epochs): ### Run the model for a certain number of iterations/epochs

      epLoss = 0
      epAcc = 0

      for x, y in train_loader: ### Fetch a batch of training inputs

          yPred = model(x) ####### Obtain a prediction from the network
          loss = criterion(yPred,y) ######### Compute the loss between prediction and ground truth

          ### Backpropagation steps
          ### Clear old gradients
          optimizer.zero_grad()
          ### Compute the gradients wrt model weights
          loss.backward()
          ### Update the model parameters
          optimizer.step()

          epLoss += loss.item()
          acc = accuracy(y,yPred)
          epAcc += acc
      trLoss.append(epLoss/len(train_loader))
      trAcc.append(epAcc/len(train_loader))

      epLoss = 0
      epAcc = 0

      for x, y in valid_loader: #### Fetch validation samples
          yPred = model(x) ##########
          loss = criterion(yPred,y)#######

          epLoss += loss.item()
          acc = accuracy(y,yPred)
          epAcc += acc
      vlLoss.append(epLoss/len(valid_loader))
      vlAcc.append(epAcc/len(valid_loader))



      if (epoch+1) % 2 == 0:
          clear_output(wait=True)
          plt.subplot(131)
          plt.imshow(x[0][0],cmap='gray')
          plt.subplot(132)
          plt.imshow(y[0][0])
          plt.subplot(133)
          # plt.imshow(yPred[0].data.reshape(128,128) > 0.5)
          plt.imshow(yPred[0][0].data)
          plt.show()
          print('Epoch: %03d, Tr. Acc: %.4f, Tr. Loss: %.4f, Vl.Acc: %.4f, Vl.Loss: %.4f'
                %(epoch,trAcc[-1],trLoss[-1],vlAcc[-1],vlLoss[-1]))
  # tracker.epoch_end()
  # tracker.stop()

  # Test set performance
def test_model(model, test_loader):
  # tracker = CarbonTracker(epochs=1,log_dir='./',log_file_prefix='SAI')
  # tracker.epoch_start()
  epAcc = 0
  epLoss = 0
  for x, y in test_loader: #### Fetch validation samples
      yPred = model(x) ##########
      loss = criterion(yPred,y)#######

      epLoss += loss.item()
      acc = accuracy(y,yPred)
      epAcc += acc
  testAcc = epAcc/len(test_loader)
  testLoss = epLoss/len(test_loader)
  # tracker.epoch_end()
  # tracker.stop()
  print("\n \n Performance on test set: test loss = %.4f, test accuracy = %.4f"%(testLoss,testAcc))
  return testAcc


  ### Function to compute binary accuracy
def accuracy(y,yPred):
    yPred = yPred > 0.5
    correct = yPred.eq(y).double()
    correct = correct.mean()
    return correct

