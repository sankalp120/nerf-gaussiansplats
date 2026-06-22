from dataset import NeRFDataset
dataset = NeRFDataset(
    "data/lego/transforms_train.json"
)

img, pose = dataset[0]

print(img.shape)
print(img.min(), img.max())
print(pose.shape)   
from dataset import NeRFDataset

dataset = NeRFDataset(
    "data/lego/transforms_train.json"
)

print("Dataset size:", len(dataset))

image, pose = dataset[0]

print("Image shape:", image.shape)
print("Pose shape:", pose.shape)

print("Image dtype:", image.dtype)
print("Pose dtype:", pose.dtype)

print("Image min:", image.min())
print("Image max:", image.max())