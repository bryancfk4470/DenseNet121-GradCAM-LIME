import os
import cv2
import torchvision.transforms as transforms
import PIL
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import make_grid, save_image


## GradCAM
class GradCAM(object):
    def __init__(self, model, layers, img_path):
        self.model = model
        self.gradients = -1
        self.activations = -1
        self.layers = layers
        self.img_path = img_path
        self.img_list = list() 

        ##Define hook for forward propagation
        def forward_hook(module, input, output):
            self.activations = output
            return None

        ##Defining hooks for back propagation
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
            return None 

        ##Instantiation of hook
        self.layers.register_forward_hook(forward_hook)
        self.layers.register_backward_hook(backward_hook)

    def ImagePreprocess(self):
        # Check if input is a PIL image or a file path
        if isinstance(self.img_path, str):
            # If it's a file path, use OpenCV to read the image
            img = cv2.imread(self.img_path)
        elif isinstance(self.img_path, Image.Image):
            # If it's already a PIL image, convert it to numpy array
            img = np.array(self.img_path)
        else:
            raise ValueError("Input should be either a file path or a PIL image.")
        
        img = cv2.resize(img, (224, 224))
        # img = img[:, :, ::-1]   # BGR --> RGB
        # If the image is grayscale (2D), convert it to 3 channels (RGB)
        if len(img.shape) == 2:  # Grayscale image (height, width)
            img = np.expand_dims(img, axis=-1)  # Add channel dimension (height, width, 1)
            img = np.repeat(img, 3, axis=-1)    # Convert to 3 channels (height, width, 3)
        
        img = img[:, :, ::-1]  # BGR --> RGB

        transform = transforms.Compose([   
        transforms.ToTensor(), 
        ])

        img = Image.fromarray(np.uint8(img))
        img_bf = transform(img)
        
        ##Keep the IMG before standardization
        nor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        img_af = nor(img_bf)

        # img_bf = img_bf.unsqueeze(0)
        #Unsqueeze is to add a dimension, and squeeze is to remove a dimension
        img_af = img_af.unsqueeze(0)    # C*H*W --> B*C*H*W

        ##Return img before standardization and img after standardization
        return img_bf, img_af


    def forward(self, input):
        b, c, h, w = input.size()
        
        self.model.eval()

        prot = self.model(input)

        ##Backward propagation
        self.model.zero_grad()

        ## prot[:, prot.max (1) [- 1]] is to get an index of the matrix
        
        ##After getting the index, it is two-dimensional, and then. Squeeze () is used to reduce the dimension
        # score = prot[:, prot.max(1)[-1]].squeeze()
        index = torch.argmax(prot)
        score = prot[:, index]

        score.backward(retain_graph=False)
        ##After the activation and gradient are obtained, they can be used to obtain cam drawings
        b, k, h, w = self.gradients.size()
        

        ##The gradient matrix is changed into a row by row, and then the average is obtained in the row direction, which is equivalent to gap, [1,512]
        alpha = self.gradients.view(b, k, -1).mean(2)

        ##It is transformed into the form of [1, 512, 1, 1]
        weights = alpha.view(b, k, 1, 1)

        ##Add the multiplied [1, 512, 14, 14] on the first dimension (512) and keep the first dimension
        ## [1, 1, 14, 14]
        saliency_map = (weights*self.activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)

        ##Get the same size as the input
        _, _, h, w = input.size()

        ##Sample the feature map to the same size
        saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)

        ##Normalization
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data



        mask = saliency_map.cpu().data.numpy()

        return mask, index

    def HeatMap(self, mask, img):
        
        ##When making Heatmap, the number of channels obtained is in the back
        heatmap = cv2.applyColorMap(np.uint8(255*mask.squeeze()), cv2.COLORMAP_JET)

        ##Turn to torch's numpy and adjust the number of channels to the front
        heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)

        ##In the channel dimension, the three separated RGB matrixes are BGR to RGB
        b, g, r = heatmap.split(1)

        heatmap = torch.cat([r, g, b])

        result = heatmap+img.cpu()

        result = result.div(result.max()).squeeze()

        return img, heatmap, result

    def __call__(self):
        ##Image preprocessing
        img_bf, img_af = self.ImagePreprocess()

        ##Forward propagation and backward propagation
        mask, index = self.forward(img_af)
        ##Generate heat map
        img, heatmap, result = self.HeatMap(mask, img_bf)
        return img,heatmap,result,index
