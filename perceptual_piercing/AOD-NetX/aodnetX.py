import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttentionLayer(nn.Module):
    """Generates an attention map given the bounding box coordinates."""
    def __init__(self, kernel_size=7):
        super(SpatialAttentionLayer, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd for 'same' padding"
        #a conv layer
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=(kernel_size // 2), bias=False)
        #sigmoid, not softmax because bounding boxes has independent importance and softmax would make the sum of probabilities to 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, boxes):
        mask = torch.zeros_like(x[:, 0:1, :, :])  # Create a single-channel mask
        batch_size, _, height, width = x.shape

        #build masks using bounding box annotations
        for i in range(len(boxes)):
            for box in boxes[i]:
                if len(box) == 6:
                    _, x_center, y_center, w, h, _ = box
                    x1 = int((x_center - w / 2) * width)
                    y1 = int((y_center - h / 2) * height)
                    x2 = int((x_center + w / 2) * width)
                    y2 = int((y_center + h / 2) * height)
                    mask[i, :, y1:y2, x1:x2] = 1

        #apply sigmoid to attention layer to map into a probability space from 0 to 1 for each bounding box
        attention = self.sigmoid(self.conv1(mask))
        return attention



class DehazeNetAttention(nn.Module):
    def __init__(self):
        super(DehazeNetAttention, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        # nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.e_conv1 = nn.Conv2d(3, 3, 1, 1, 0, bias=True)
        self.e_conv2 = nn.Conv2d(3, 3, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(6, 3, 5, 1, 2, bias=True)
        self.e_conv4 = nn.Conv2d(6, 3, 7, 1, 3, bias=True)
        self.e_conv5 = nn.Conv2d(12, 3, 3, 1, 1, bias=True)

        # Spatial attention layer
        self.attention = SpatialAttentionLayer()

    def forward(self, x, boxes):
        # Apply attention based on bounding boxes
        attention_map = self.attention(x, boxes)
        x = x * attention_map  # Modulate the input image directly
    
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
    
        concat1 = torch.cat((x1, x2), 1)
        x3 = self.relu(self.e_conv3(concat1))
    
        concat2 = torch.cat((x2, x3), 1)
        x4 = self.relu(self.e_conv4(concat2))
    
        concat3 = torch.cat((x1, x2, x3, x4), 1)
        x5 = self.relu(self.e_conv5(concat3))
    
        # Final clean image synthesis
        clean_image = self.relu((x5 * x) - x5 + 1)
    
        return clean_image
