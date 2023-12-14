import os

import matplotlib.pyplot as plt
import torch


def plot_images(dataloader):
    images, labels = next(iter(dataloader))

    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(20, 4))

    for i in range(5):
        # Convert the tensor to a NumPy array and transpose it
        img = images[i].numpy().transpose(1, 2, 0)
        label = "CelebA" if labels[i] == 0 else "ImageNet"

        # Display the image
        ax[i].imshow(img)
        ax[i].set_title(label)
        ax[i].axis("off")  # Turn off axis numbers and labels

    plt.show()


def make_save_dir(base_dir):
    # Ensure the base directory exists
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Find the next version number
    existing_versions = [
        d
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("version_")
    ]
    if existing_versions:
        latest_version = max([int(v.split("_")[1]) for v in existing_versions])
        next_version = latest_version + 1
    else:
        next_version = 1

    # Create new version directory
    version_dir = os.path.join(base_dir, f"version_{next_version}")
    os.makedirs(version_dir)

    return version_dir


def log_message(message, save_dir):
    file_path = os.path.join(save_dir, "training_log.txt")
    with open(file_path, "a") as file:
        file.write(message + "\n")


def save_model(model, save_dir, epoch="last"):
    # Save the model checkpoint
    checkpoint_path = os.path.join(save_dir, f"model_checkpoint-{epoch}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

    return checkpoint_path


def load_model(model, load_dir):
    checkpoint_path = os.path.join(load_dir, "model_checkpoint-last.pth")
    model.load_state_dict(torch.load(checkpoint_path))
    return model
