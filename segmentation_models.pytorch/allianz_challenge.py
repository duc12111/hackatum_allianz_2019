import os
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import numpy as np
import sys

# noinspection PyUnresolvedReferences
from allianz_dataset import Dataset
import albumentations as albu

sys.path.insert(0, '/home/ahmad/hackthon/hackatum/segmentation_models.pytorch')
import segmentation_models_pytorch as smp


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


DATA_DIR = '/home/ahmad/hackthon/hackatum/data'


x_train_dir = os.path.join(DATA_DIR, 'data_scaled')
y_train_dir = os.path.join(DATA_DIR, 'seg_masks')

x_valid_dir = os.path.join(DATA_DIR, 'data_scaled_valid')
y_valid_dir = os.path.join(DATA_DIR, 'seg_masks_valid')

x_test_dir = os.path.join(DATA_DIR, 'data_scaled_test')
y_test_dir = os.path.join(DATA_DIR, 'seg_masks_test')


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

def get_training_augmentation_padding():
    test_transform = [
        albu.Resize(320, 512, cv2.BORDER_CONSTANT, cv2.INTER_NEAREST)
    ]
    return albu.Compose(test_transform)


def get_validation_augmentation_padding():
    test_transform = [
        albu.Resize(320, 512, cv2.BORDER_CONSTANT, cv2.INTER_NEAREST)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


# Visualize resulted augmented images and masks
augmented_dataset = Dataset(
    x_train_dir, 
    y_train_dir,
    augmentation=get_training_augmentation_padding(),
    classes=['crack'],
)


# Create model and train
ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['crack']
ACTIVATION = 'sigmoid'
DEVICE = 'cuda'

# create segmentation model with pretrained encoder
model = smp.FPN(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES), 
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

train_dataset = Dataset(
    x_train_dir, 
    y_train_dir,
    augmentation=get_training_augmentation_padding(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

valid_dataset = Dataset(
    x_valid_dir, 
    y_valid_dir,
    augmentation=get_validation_augmentation_padding(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)

loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0001),
])

# create epoch runners 
# it is a simple loop of iterating over dataloader`s samples
train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)

# train model for 40 epochs
max_score = 0

for i in range(0, 5):

    break

    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)
    
    # do something (save model, change lr, etc.)
    max_score = valid_logs['iou_score']
    torch.save(model, './best_model.pth')
    print('Model saved!')
        
    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')


# load best saved checkpoint
best_model = torch.load('./best_model.pth')

# create test dataset
test_dataset = Dataset(
    x_test_dir,
    y_test_dir,
    augmentation=get_validation_augmentation_padding(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

test_dataloader = DataLoader(test_dataset)


# evaluate model on test set
test_epoch = smp.utils.train.ValidEpoch(
    model=best_model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
)

logs = test_epoch.run(test_dataloader)


# test dataset without transformations for image visualization
test_dataset_vis = Dataset(
    x_test_dir, y_test_dir,
    classes=CLASSES,
)


for i in range(18):

    image_vis = test_dataset_vis[i][0].astype('uint8')
    image, gt_mask = test_dataset[i]
    
    gt_mask = gt_mask.squeeze()
    
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())

    visualize(
        image=image_vis, 
        ground_truth_mask=gt_mask,
        predicted_mask=pr_mask
    )

