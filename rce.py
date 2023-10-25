import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2
WINDOWSIZEX = 700
WINDOWSIZEY = 690
BOUNDRINC = 15
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
IMAGESAVE = False
MODEL = load_model("cse.h5")
LABELS = {0:"ZERO", 1: "ONE",
          2:"TWO", 3: "THREE",
          4:"FOUR", 5: "FIVE",
          6:"SIX", 7: "SEVEN",
          8:"EIGHT", 9:"NINE"}

pygame.init()
FONT = pygame.font.Font("freesansbold.ttf", 18)

DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
pygame.display.set_caption("SARIM Board")

iswriting = False
number_xcord = []
number_ycord = []
image_cnt = 1
img_arr=None
PREDICT = True
rect_min_x = WINDOWSIZEX
rect_min_y = WINDOWSIZEY
rect_max_x = 0
rect_max_y = 0

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)

            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            if len(number_xcord) > 0 and len(number_ycord) > 0:
                 rect_min_x, rect_max_x = min(number_xcord), max(number_xcord)
                 rect_min_y, rect_max_y = min(number_ycord), max(number_ycord)

                 img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)

                 number_xcord = []
                 number_ycord = []

        if img_arr is not None:
            if IMAGESAVE:
                cv2.imwrite(f"image{image_cnt}.png", img_arr)
                image_cnt += 1

            if PREDICT:
                image = cv2.resize(img_arr, (28, 28), interpolation=cv2.INTER_AREA)
                image = np.pad(image, (10, 10), 'constant', constant_values=0)
                image = cv2.resize(image, (28, 28)) / 255

                label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1, 28, 28, 1)))])
                textSurface = FONT.render(label, True, RED, WHITE)
                textRecObj = textSurface.get_rect()
                textRecObj.left, textRecObj.top = rect_min_x, rect_min_y - 20
                DISPLAYSURF.blit(textSurface, textRecObj)

        pygame.draw.rect(DISPLAYSURF, RED, (rect_min_x, rect_min_y, rect_max_x - rect_min_x, rect_max_y - rect_min_y), 2)


        if event.type == KEYDOWN and event.unicode == "n":
            DISPLAYSURF.fill(BLACK)

        pygame.display.update()
