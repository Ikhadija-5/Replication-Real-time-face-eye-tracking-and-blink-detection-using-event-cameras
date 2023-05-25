# main.py
from torch.utils.data import DataLoader, random_split
import model
import data
import train

# Load and preprocess data
# raw_data = data.load_data()
# preprocessed_data = data.preprocess_data(raw_data)

# Create the model
my_model = model.GR_YOLO()

# Define batch size
batch_size = 32

# Create the data loader
csv_file_path = "/home/Khadija/Eye_Blink_Detection/state-farm-distracted-driver-detection/driver_imgs_list.csv"
images_folder_path = "/home/Khadija/Eye_Blink_Detection/state-farm-distracted-driver-detection/imgs/train"
dataset = data.ObjectDetectionDataset(csv_file_path,images_folder_path)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Train the model
train.train_model(my_model, train_loader, num_epochs=10, learning_rate=0.001)

# Evaluate the model
test_data =  " "
test_labels = " "

test_loader = DataLoader(test_data, test_labels, batch_size=batch_size)
train.evaluate_model(my_model, test_loader)


