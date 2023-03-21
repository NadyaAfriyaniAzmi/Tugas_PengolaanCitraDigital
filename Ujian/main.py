import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk


class ImageEditor(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Aplikasi Perbaikan Citra")
        self.geometry("500x500")
        # create labels for displaying the images
        self.original_label = tk.Label(self, text="Original Image")
        self.original_label.pack(side=tk.LEFT, padx=10, pady=10)

        self.processed_label = tk.Label(self, text="Processed Image")
        self.processed_label.pack(side=tk.RIGHT, padx=10, pady=10)

        # create buttons for selecting image and applying image processing methods
        self.select_image_button = tk.Button(self, text="Select Image", command=self.select_image)
        self.select_image_button.pack(pady=10)

        self.gray_button = tk.Button(self, text="Convert to Grayscale", state=tk.DISABLED, command=self.convert_to_gray)
        self.gray_button.pack(pady=10)

        self.flip_button = tk.Button(self, text="Flip Image", state=tk.DISABLED, command=self.flip_image)
        self.flip_button.pack(pady=10)

        self.image_path = None
        self.image = None

    def select_image(self):
        # show file dialog to select the image
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if file_path:
            self.image_path = file_path
            self.image = Image.open(file_path)

            # display the original image
            self.show_image(self.image, self.original_label)

            # enable the image processing buttons
            self.gray_button.config(state=tk.NORMAL)
            self.flip_button.config(state=tk.NORMAL)

    def convert_to_gray(self):
        if self.image:
            # apply grayscale to the image
            gray_image = self.image.convert('L')

            # display the processed image
            self.show_image(gray_image, self.processed_label)

    def flip_image(self):
        if self.image:
            # flip the image horizontally
            flipped_image = self.image.transpose(Image.FLIP_LEFT_RIGHT)

            # display the processed image
            self.show_image(flipped_image, self.processed_label)

    def show_image(self, image, label):
        # resize the image to fit in the label
        width, height = image.size
        ratio = min(1.0, 300.0 / max(width, height))
        new_width = int(ratio * width)
        new_height = int(ratio * height)
        image = image.resize((new_width, new_height), Image.ANTIALIAS)

        # convert the image to PhotoImage and display it in the label
        photo_image = ImageTk.PhotoImage(image)
        label.config(image=photo_image)
        label.image = photo_image


if __name__ == '__main__':
    app = ImageEditor()
    app.mainloop()
