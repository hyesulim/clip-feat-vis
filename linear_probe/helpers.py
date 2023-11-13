import matplotlib.pyplot as plt


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
