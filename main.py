import torch
import torch.nn as nn
import pygame
import numpy as np
from PIL import Image

# defining our model
def create_CNN():
    model = nn.Sequential(
        nn.Conv2d(1, 6, 5, padding = 2),
        nn.ReLU(),
        nn.AvgPool2d(2, stride = 2),
        nn.Conv2d(6, 16, 5, padding = 0),
        nn.ReLU(),
        nn.AvgPool2d(2, stride = 2),
        nn.Flatten(),
        nn.Linear(400, 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        nn.Linear(84, 10)
    )

    return model

# loading our model based on a file path and setting to evaluation mode
def load_model(filepath):
    model = create_CNN()
    model.load_state_dict(torch.load(filepath, map_location = torch.device("cpu")))
    model.eval()
    return model

# load image
def load_img(filepath):
    image = Image.open(filepath).convert("L") # loading image and converting it to grayscale
    image = image.resize((28, 28)) # reshape the image to 28 by 28 
    image_arr = np.array(image) # convert image to numpy array
    image_tensor = torch.tensor(image_arr, dtype = torch.float32) # convert image to tensor

    # the shape (1, 1, 28, 28) to indicate that the image has one channel (black and white) and there is only image in the tensor
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0) # convert image shape from (28, 28) to (1, 1, 28, 28)
    
    return image_tensor

# create drawing gui and save image
def create_canvas(filepath):
    pygame.init()
    
    screen = pygame.display.set_mode((280, 280))
    pygame.display.set_caption("Digit Recognition")

    drawing = False
    last_pos = None

    running = True
    while (running):
        for event in pygame.event.get():
            if (event.type == pygame.QUIT):
                running = False
            elif (event.type == pygame.MOUSEBUTTONDOWN):
                drawing = True
            elif (event.type == pygame.MOUSEMOTION and drawing):
                mouse_pos = pygame.mouse.get_pos()
                if (last_pos):
                    pygame.draw.line(screen, (255, 255, 255), last_pos, mouse_pos, 20)
                last_pos = mouse_pos
            elif (event.type == pygame.MOUSEBUTTONUP):
                drawing = False
                last_pos = None
            elif (event.type == pygame.KEYDOWN):
                if (event.key == pygame.K_RETURN):
                    pygame.image.save(screen, filepath)
                    running = False

        pygame.display.flip()
    
    pygame.quit()

if (__name__ == "__main__"):
    # loaded the model
    model = load_model("basicCNN\weights.pth")
    
    # init gui
    create_canvas("digit_img.png")

    # load our saved image from gui
    loaded_img = load_img("digit_img.png")

    # get raw output from model
    outputs = model(loaded_img)

    # apply softmax to raw outputs
    probs = torch.softmax(outputs, dim = 1)
    # convert probabilities to numpy array and detach it from softmax to give raw probabilities
    probs = probs.detach().cpu().numpy()
    print("Class Probabilities: ")
    for i, prob in enumerate(probs[0]):
        print(f'Class {i}: {prob * 100}%')