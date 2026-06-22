from dataset import NeRFDataset

dataset = NeRFDataset(
    "transforms_train.json"
)

print(len(dataset))

image, pose = dataset[0]

print(image.shape)
print(pose.shape)