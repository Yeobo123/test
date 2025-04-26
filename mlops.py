import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import SGD
import mlflow
import mlflow.pytorch
from PIL import Image

# Thiết bị
device = "cuda" if torch.cuda.is_available() else "cpu"

# ===== Load dữ liệu =====
data_folder = './data/FMNIST'
fmnist = datasets.FashionMNIST(data_folder, download=True, train=True)
tr_images = fmnist.data
tr_targets = fmnist.targets
class_names = fmnist.classes

# ===== Dataset tùy chỉnh =====
class FMNISTDataset(Dataset):
    def __init__(self, x, y):
        x = x.float().view(-1, 28 * 28)  # flatten
        self.x, self.y = x, y

    def __getitem__(self, ix):
        return self.x[ix].to(device), self.y[ix].to(device)

    def __len__(self):
        return len(self.x)

def get_data():
    train = FMNISTDataset(tr_images, tr_targets)
    return DataLoader(train, batch_size=32, shuffle=True)

# ===== Model =====
def get_model():
    model = nn.Sequential(
        nn.Linear(28 * 28, 1000),
        nn.ReLU(),
        nn.Linear(1000, 10)
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=1e-2)
    return model, loss_fn, optimizer

def train_batch(x, y, model, opt, loss_fn):
    model.train()
    prediction = model(x)
    batch_loss = loss_fn(prediction, y)
    batch_loss.backward()
    opt.step()
    opt.zero_grad()
    return batch_loss.item()

@torch.no_grad()
def accuracy(x, y, model):
    model.eval()
    prediction = model(x)
    _, argmaxes = prediction.max(-1)
    return (argmaxes == y).cpu().numpy().tolist()

# ===== Dự đoán ảnh từ thư mục =====
def process_images_in_folder(folder_path, model, class_names):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten
    ])

    model.eval()

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                img = Image.open(img_path).convert('L')
                input_tensor = transform(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(input_tensor)
                    _, pred_class = output.max(1)

                label = class_names[pred_class.item()]
                print(f"🖼️ {img_name} → Predict: {label}")
                plt.imshow(img, cmap='gray')
                plt.title(f'Predicted: {label}')
                plt.axis('off')
                plt.show()
            except Exception as e:
                print(f"Lỗi xử lý ảnh {img_name}: {e}")
                
# Lưu mô hình vào thư mục sau khi huấn luyện
def save_model(model, path):
    torch.save(model.state_dict(), path)  # Lưu trạng thái của mô hình vào file


# ===== Huấn luyện + MLflow =====
def train_with_mlflow():
    trn_dl = get_data()
    model, loss_fn, optimizer = get_model()

    losses, accuracies = [], []
    epochs = 5
    lr = 1e-2
    


    with mlflow.start_run():
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("lr", lr)
        mlflow.log_param("optimizer", "SGD")
        mlflow.log_param("model_architecture", "Linear(784 → 1000 → 10)")

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}")
            epoch_losses, epoch_accuracies = [], []

            for x, y in trn_dl:
                batch_loss = train_batch(x, y, model, optimizer, loss_fn)
                epoch_losses.append(batch_loss)

            epoch_loss = np.mean(epoch_losses)

            # Accuracy
            epoch_accuracies = []
            for x, y in trn_dl:
                is_correct = accuracy(x, y, model)
                epoch_accuracies.extend(is_correct)
            epoch_accuracy = np.mean(epoch_accuracies)

            mlflow.log_metric("loss", epoch_loss, step=epoch)
            mlflow.log_metric("accuracy", epoch_accuracy, step=epoch)
            losses.append(epoch_loss)
            accuracies.append(epoch_accuracy)
        


        # Log model + input_example
        example_input = torch.randn(1, 28*28).numpy()  # Chuyển sang numpy
        mlflow.pytorch.log_model(model, "fmnist_model", input_example=example_input)
        # Lưu mô hình sau khi huấn luyện
        model_path = "fmnist_model.pth"  # Đặt tên cho mô hình
        save_model(model, model_path)
        print(f"✅ Đã lưu mô hình tại {model_path}")

        # Plot loss & accuracy
        plt.figure(figsize=(20, 5))
        plt.subplot(121)
        plt.title("Loss over epochs")
        plt.plot(np.arange(epochs) + 1, losses, label="Loss")
        plt.legend()

        plt.subplot(122)
        plt.title("Accuracy over epochs")
        plt.plot(np.arange(epochs) + 1, accuracies, label="Accuracy")
        plt.gca().set_yticklabels(['{:.0f}%'.format(x * 100) for x in plt.gca().get_yticks()])
        plt.legend()
        plt.show()

        return model

# ===== Main =====
if __name__ == "__main__":
    model = train_with_mlflow()

    

    test_folder = 'test_images'
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
        print(f"\n📁 Đã tạo thư mục '{test_folder}'. Hãy thêm ảnh PNG/JPG để test.")
    else:
        process_images_in_folder(test_folder, model, class_names)
