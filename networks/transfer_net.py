from models.resnet_model_modified import ResNet50Fc, ResNet18Fc
from torch import nn

from losses.coral_loss import CORAL_loss
from losses.mmd_loss import MMD_loss


class TransferNet(nn.Module):
    def __init__(self, num_class, base_net, transfer_loss='coral', use_bottleneck=False, bottleneck_width=256, width=1024): #1024
        super(TransferNet, self).__init__()
        if base_net == 'resnet50':
            self.base_network = ResNet50Fc()
        elif base_net == 'resnet18':
            self.base_network = ResNet18Fc()
        else:
            # Your own basenet
            return
        self.use_bottleneck = use_bottleneck
        self.transfer_loss = transfer_loss
        bottleneck_list = [nn.Linear(self.base_network.output_num()
        , bottleneck_width), nn.BatchNorm1d(bottleneck_width), nn.ReLU(), nn.Dropout(0.5)]
        self.bottleneck_layer = nn.Sequential(*bottleneck_list)

        classifier_layer_list = [nn.Linear(self.base_network.output_num(), width), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(width, num_class)]  

        # classifier_layer_list = [nn.Linear(4608, width), nn.ReLU(), nn.Dropout(0.5),
        #                          nn.Linear(width, num_class)]
        self.classifier_layer = nn.Sequential(*classifier_layer_list)

        self.bottleneck_layer[0].weight.data.normal_(0, 0.00005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)
        for i in range(2):
            self.classifier_layer[i * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer[i * 3].bias.data.fill_(0.0)

    def forward(self, source, target):
        source = self.base_network(source)
        target = self.base_network(target) if target is not None else None
        source_clf = self.classifier_layer(source)
        if self.use_bottleneck:
            source = self.bottleneck_layer(source)
            target = self.bottleneck_layer(target) if target is not None else None
        transfer_loss = self.adapt_loss(source, target, self.transfer_loss) if target is not None else None
        # transfer_loss = 0
        return source_clf, transfer_loss, source
    
    def forward_features(self, domain):
        domain_fea = self.base_network(domain)
        if self.use_bottleneck:
            domain_fea = self.bottleneck_layer(domain_fea)
        return domain_fea

    def forward_tsne(self, dataset):
        dataset = self.base_network(dataset)
        if self.use_bottleneck:
            dataset = self.bottleneck_layer(dataset)
        return dataset

    def predict(self, x):
        features = self.base_network(x)
        clf = self.classifier_layer(features)
        return clf

    def adapt_loss(self, X, Y, adapt_loss):
        """Compute adaptation loss, currently we support mmd and coral

        Arguments:
            X {tensor} -- source matrix
            Y {tensor} -- target matrix
            adapt_loss {string} -- loss type, 'mmd' or 'coral'. You can add your own loss

        Returns:
            [tensor] -- adaptation loss tensor
        """
        if adapt_loss == 'mmd':
            mmd_loss = MMD_loss()
            loss = mmd_loss(X, Y)
        elif adapt_loss == 'coral':
            coral_loss = CORAL_loss.coral_loss(X, Y)
            loss = coral_loss(X, Y)
        else:
            # Your own loss
            loss = 0
        return loss
