import cv2
import numpy as np
import os
from utils import remove_files_in_dir, pre_process_original_image, extract_possible_cards, rotate_possible_cards, CARD_WIDTH, zoom_image, MASK_HEIGHT, MASK_WIDTH

CARDS_DIRECTORY = './cards'
DESTINATION_DIRECTORY = './masks'

BORDER_SIZE = 10

remove_files_in_dir(DESTINATION_DIRECTORY)
#crop_all(CARDS_DIRECTORY, DESTINATION_DIRECTORY)

# Adjust this values if needed
# You'll need to find an optimal crop (keep the whitespace to minimum)
# Coordinates always start from top left
START_CROPPING_Y = 18 + BORDER_SIZE
START_CROPPING_X = 10 + BORDER_SIZE
def run():
    for image_name in os.listdir(CARDS_DIRECTORY):
        if image_name != ".DS_Store":
            print(f'{image_name}')
            image = cv2.imread(f'{CARDS_DIRECTORY}/{image_name}')
            processed_image = pre_process_original_image(image)
            possible_cards, possible_cards_not_processed = extract_possible_cards(processed_image, image)

            possible_cards_with_borders = []
            for possible_card in possible_cards:
                image_with_border = cv2.copyMakeBorder(possible_card, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, cv2.BORDER_CONSTANT, value=[0,0,0])
                possible_cards_with_borders.append(image_with_border)
           
            rotated_possible_cards, _ = rotate_possible_cards(possible_cards_with_borders, possible_cards_not_processed)

            for index, rotated_possible_card in enumerate(rotated_possible_cards):
                # each card is rotated 4 times (see the rotation function)
                # but we know that for masks we can use the one from index 1
                # because we are assuming that the uploaded card is almost perpendicular with the X axis
                if(index != 1):
                    continue
                h, w = rotated_possible_card.shape[:2]
                # zoom in or out; depends on the CARD_WIDTH / w; if greater than 1, it will zoom out (make it bigger)
                zoom_factor = CARD_WIDTH / (w - BORDER_SIZE * 2)
                print('CARD_WIDTH', CARD_WIDTH)
                print('w', w)
                print('zoom_factor', zoom_factor)
                zoomed_possible_card = zoom_image(rotated_possible_card,  zoom_factor )
                
                # 10 should have the 0 class
                # 1 should also work
                # you only have to maintain the pattern: 1st char identifies the number, the 2nd char identifies the card type
                if image_name[0] == "1":
                    file_name = f'{image_name[1]}{image_name[2]}'
                else:
                    file_name = f'{image_name[0]}{image_name[1]}'

                cv2.imwrite(f'{DESTINATION_DIRECTORY}/{file_name}.jpg', zoomed_possible_card[START_CROPPING_Y:START_CROPPING_Y+MASK_HEIGHT, START_CROPPING_X:START_CROPPING_X+MASK_WIDTH])

run()