## Data Exploration
To download, visualize and preprocess the dataset.

## models
The three models used are:

### U-Net
The standard U-Net is a convolutional neural network designed for biomedical image segmentation. It consists of an encoder (contracting path) to capture context and a decoder (expanding path) for precise localization, with skip connections between corresponding layers to retain spatial information.

### Res+U-Net (Residual U-Net)
Res+U-Net extends the standard U-Net by adding residual connections within the network blocks. These connections help the model learn identity mappings, improve gradient flow, and can lead to better performance and faster convergence, especially on deeper networks.

### M-Net (Multi-scale U-Net)
M-Net is a multi-scale variant of U-Net. It processes the input image at multiple scales using downsampling and fuses features from different resolutions. This allows the model to capture both global context and fine details, which can improve segmentation accuracy, especially for objects of varying sizes.

## evaluation
Testing the models on the test set

## predictions
Making prediction on unseen data

Use the three notebooks to see all the above combined.

## Comparative Analysis
| Model  | epochs | Loss_function | dice_coef |
| ------------- | ------------- | ------------- | -------------|
| U-Net  | 50  | 0.0577790886 | 0.672770553 |
| Residual U-Net  | 50	  | 0.0536935590  | 0.6101779341  |
| M-Net | 50 | 0.0764972120 | 0.6455184817 |

