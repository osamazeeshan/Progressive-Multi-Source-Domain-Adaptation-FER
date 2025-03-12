from torchvision import models
import torch
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm, trange

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the architecture by modifying resnet.
# Original code is here http://tiny.cc/8zpmmz 
class ResNet101(models.ResNet):
    def __init__(self, num_classes=1000, pretrained=True, **kwargs):
        # Start with the standard resnet101
        super().__init__(
            block=models.resnet.Bottleneck,
            layers=[3, 4, 23, 3],
            num_classes=num_classes,
            **kwargs
        )
 
    # Reimplementing forward pass.
    # Replacing the forward inference defined here 
    # http://tiny.cc/23pmmz
    def _forward_impl(self, x):
        # Standard forward for resnet
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
 
        # Notice there is no forward pass through the original classifier.
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
 
        return x


def var_plot_tsne_graph(dataloader):
    seed = 10
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = ResNet101(pretrained=True).cuda()
    model.eval()
    features = None
    labels = []
    image_paths = []

    # for batch in tqdm(dataloader, desc='Running the model inference'):
    for data, target in dataloader:
        data, target = data.cuda(), target.cuda()
        labels += target
        image_paths += data
    
        output = model.forward(data)
    
        current_outputs = output.cpu().detach().numpy()
        if (features is None):
                features = current_outputs
        else:
            features = np.concatenate((features, current_outputs))

    tsne = TSNE(n_components=2).fit_transform(features)
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # initialize a matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors_per_class = [0,1,2,3,4,5,6]
    # for every class, we'll add a scatter plot separately
    for label in colors_per_class:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == label]
    
        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
    
        current_tx = current_tx[:100] if len(current_tx) > 100 else current_tx
        current_ty = current_ty[:100] if len(current_ty) > 100 else current_ty

        # convert the class color to matplotlib format
        # color = np.array(colors_per_class[label], dtype=np.float) / 255
    
        # add a scatter plot with the corresponding color and label
        ax.scatter(current_tx, current_ty, label=label)
    
    # build a legend using the labels we set previously
    ax.legend(loc='best')
    
    # finally, show the plot
    plt.savefig('1_tsne.png')
    plt.show()

    # scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))
 
    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)
 
    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range
 
