import cv2
import numpy as np
import time
from utils import show_with_wait, CARD_WIDTH, zoom_image, find_card_dimensions, remove_files_in_dir, extract_possible_cards, match_against_all_masks, mark_overlaps, remove_negatives, rotate_possible_cards, pre_process_original_image, extract_cards_headers

# Renders images on the screen and save data under ./generated directory
DEBUG = False

# Just for debugging;
# Important: cv2.waitKey(0) and cv2.destroyAllWindows() are not called here, but at the end of the program
# The above 2 are required in order to exit the program
def draw_result(max_values_dict, max_image, max_lefts_dict, max_rights_dict):
    if DEBUG is False:
        return
    for file_prefix, value in max_values_dict.items():
        image = np.copy(max_image[file_prefix])
        
        background_img = image

        y_offset = max_lefts_dict[file_prefix][1]
        if(y_offset < 0):
            continue
        cv2.rectangle(background_img, max_lefts_dict[file_prefix], max_rights_dict[file_prefix], 100, 2)
        cv2.imshow(f'{file_prefix}_{value}', background_img)

MASKS_DIRECTORY = './masks';

def run(image_path, image_name):
    remove_files_in_dir('./generated')
    image = cv2.imread(f'{image_path}/{image_name}')

    processed_image = pre_process_original_image(image, True)
    possible_cards, possible_cards_not_processed = extract_possible_cards(processed_image, image)
    print('len(possible_cards)', len(possible_cards))
    
    card_width, card_height = find_card_dimensions(possible_cards[0], possible_cards_not_processed[0])
    zoom_factor = CARD_WIDTH / card_width

    print("CARD_WIDTH", CARD_WIDTH)
    print("card_width", card_width)

    print("zoom_factor", zoom_factor)
    processed_image = zoom_image(processed_image, zoom_factor)
    image = zoom_image(image, CARD_WIDTH / card_width)

    possible_cards, possible_cards_not_processed = extract_possible_cards(processed_image, image)
    print('Resized len(possible_cards)', len(possible_cards))

    rotated_possible_cards, rotated_possible_cards_not_processed = rotate_possible_cards(possible_cards, possible_cards_not_processed)
    
    print('Resized len(rotated_possible_cards)', len(rotated_possible_cards))
    print('Resized len(rotated_possible_cards_not_processed)', len(rotated_possible_cards_not_processed))

    rotated_possible_cards_headers = extract_cards_headers(rotated_possible_cards, rotated_possible_cards_not_processed, image_name)
    print('Resized len(rotated_possible_cards_headers)', len(rotated_possible_cards_headers))


    all_predictions = []
    final_predictions = {}
    all_zooms = []
    for index, rotated_possible_card_header in enumerate(rotated_possible_cards_headers):
        max_values_dict, max_name, max_lefts_dict, max_rights_dict, max_template_dict, max_image, max_zoom_dict= match_against_all_masks(rotated_possible_card_header, MASKS_DIRECTORY)
        max_values_dict = mark_overlaps(max_values_dict, max_name, max_lefts_dict)
        max_values_dict = remove_negatives(max_values_dict, max_name)
        all_predictions.append(max_values_dict)
        all_zooms.append(max_zoom_dict)

        draw_result(max_values_dict, max_image, max_lefts_dict, max_rights_dict)
    
    for index, predictions in enumerate(all_predictions):
        for file_prefix, value in predictions.items():
            if file_prefix in final_predictions:
                if final_predictions[file_prefix] < value:
                    final_predictions[file_prefix] = value
            else:
                final_predictions[file_prefix] = value
    
    if not final_predictions:
        print("No cards detected")
    else:
        print("Cards detected")
        if DEBUG:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    print(f'final_predictions={final_predictions}')

IMAGE_PATH = './test'
IMAGE_NAME = 'J_5.jpg'
start_time = time.time()

run(IMAGE_PATH, IMAGE_NAME)

end_time = time.time()
elapsed_time = end_time - start_time
print("Time elapsed:", elapsed_time, "seconds")
print("Time elapsed:", round(elapsed_time, 2), "seconds")