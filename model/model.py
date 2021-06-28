import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        source_model = models.vgg19(pretrained=True).features

        # replace in-place relus
        for name, layer in source_model.named_children():
            if isinstance(layer, nn.ReLU):
                setattr(source_model, name, nn.ReLU(inplace=False))
            # if isinstance(layer, nn.MaxPool2d): # did not give good results
            #     setattr(source_model, name, nn.AvgPool2d(
            #                                     kernel_size=2,
            #                                     stride=2,
            #                                     padding=0)
            #                                 )

        # get the feature layers
        features = list(source_model)
        # set to eval mode
        self.features = nn.ModuleList(features)
        # freeze layers
        for parameter in self.features.parameters():
            parameter.requires_grad = False

    def forward(self, x):
        results = []
        needed_layers = {1, 6, 11, 20, 29, 31}
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in needed_layers:
                results.append(x)
        # (style_layers, content_layers)
        return results[:-1], list(results[-1])

# content loss
def get_content_loss(base_content, target):
    return F.mse_loss(base_content, target)

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

def get_style_loss(base_style, gram_target):
    G = gram_matrix(base_style)
    loss = F.mse_loss(G, gram_target)
    return loss

if __name__ == "__main__":
    vgg_model = VGG19()
