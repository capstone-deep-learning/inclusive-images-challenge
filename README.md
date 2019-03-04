# inclusive-images-challenge
## Inspired from https://www.kaggle.com/c/inclusive-images-challenge ##

## Data https://www.kaggle.com/c/inclusive-images-challenge/data ##
## Models (under./models) ##
 * CropnResize Used for preprocssing and did Image crop, resize (256*256) for data preparation

 * 7Layer5Conv3DenseBn custom model developed by our team

 * VGG16TransferLearning as name indicates leveraged the transfer learning with VGG16 (https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5)

## Architecture Diagram ##
![picture alt](./InclusiveImageChallengeArchitecture.png)

## Metrics ##
* All models are trained for 100 epochs

| Model                | Training Acc  | Validation Acc | Time per Epoch |
| -------------------- | ------------- | -------------  | -------------  | 
| VGG16                | TBF           | TBF            | TBF            |
| 7Layer5Conv3DenseBn  | 0.99          | 0.79           | 170 seconds    |
