import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
from torchvision.transforms import transforms
import torch.optim as optim


device = torch.device("cuda" if torch.cuda_is_available() else "cpu")

cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(cifar100_mean, cifar100_std)
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar100_mean, cifar100_std)
])

train_dataset = torchvision.datasets.CIFAR100(root = "./cifar100", train = True, Download = True, transform = train_transform)
val_dataset = torchvision.datasets.CIFAR100(root = "./cifar100", train = False, Download = True, transform = val_transform)

subset_target_classes = [
    # Flowers
    'orchid', 'poppy', 'sunflower',
    # Mammals
    'fox', 'raccoon', 'skunk',
    # Insects
    'butterfly', 'caterpillar', 'cockroach'
]

class_to_ids = {class:i for i, class in enumerate(train_dataset.classes)}
target_indices = {class_to_ids[class] for class in subset_target_classes}

def filter_dataset(dataset, target_indices):
	indices =[]
	for i,(_, label) in enumerate(dataset):
		if label in target_indices:
			indices.append(i)
	return Subset(dataset, indices)


train = filter_dataset(train_dataset, target_indices)
val = filter_dataset(val_dataset, target_indices)

new_label_map = {old:new for new, old in enumerate(target_indices)}

RemappedDataset(Dataset):
	def __init__(self, subset,label_map):
		self.subset = subset 
		self.label_map = label_map
	
	def __len__(self):
		return len(self.subset)
	
	def __getitem__(self, idx):
		image, label = subset[idx]
		new_label = label_map[label]
		return image, new_label
		
	
train_dataset_proto = RemappedDataset(train, new_label_map)
val_dataset_proto = RemappedDataset(val, new_label_map)

batch_size = 64

train_loader_proto = DataLoader(
    train_dataset_proto,
    batch_size=batch_size,
    shuffle=True
)

val_loader_proto = DataLoader(
    val_dataset_proto,
    batch_size=batch_size,
    shuffle=False
)

class NatureCNN(nn.Module):
	def __init__(self, num_classes = 9):
		super.__init__()
		self.num_classes = num_classes

		self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernal_size = 3, padding = 1)
		self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernal_size = 3, padding = 1)
		self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernal_size = 3, padding = 1)

		self.pool = MaxPool2d(2,2)

		self.fc1 = nn.Linear(128*4*4, 256)
		self.fc2(256, self.num_classes)

		self.dropout = nn.Dropout(0.5)

	def forward(self):
		
		x = self.pool(torch.relu(self.conv1(x)))
		x = self.pool(torch.relu(self.conv2(x)))
		x = self.pool(torch.relu(self.conv3(x)))

		x = torch.flatten(x,1)
		x = torch.relu(self.fc1(x))
		x = self.dropout(x)
		x = self.fc2(x)

		return x


model = NatureCNN(num_classes=9).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


		
		
model.train()
history = []
for epoch in range(num_epochs):
	
	running_loss = 0.0
	
	for images, labels in train_loader_proto:
		images = images.to(device)
		labels = labels.to(device)
		optimizer.zero_grad()
		output = model(images)
		loss = criterion(output, labels)
		running_loss=running_loss + loss.item()
		
		loss.backward()
		optimizer.step()
	
	epoch_loss = running_loss/len(train_loader_proto)
	history.append(epoch_loss)
	print(epoch_loss)


model.eval()

with torch.no_grad():
	correct = 0.0
	total = 0.0
	for images, labels in val_loader_proto:
		images = images.to(device)
		labels = labels.to(device)
		
		output = model(images)
		
		_,prediction = torch.max(output,1)
		correct += (prediction == labels).sum().item()

		total += labels.size(0)
		#loss.backward()
		#optimizer.step()
	print("Accuracy: ", (correct/total)*100)
		
	
		
		
		
		
		



