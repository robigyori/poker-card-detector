import cv2
import numpy as np
import os
import math 
# The images must be greater than the masked templates
# if you change this values, make sure that you don't have masks with greater size
# otherwise you'll get an error at matchTemplate
CARD_WIDTH = 254
CARD_HEIGHT = 382

# if the overlap/match is less than this value, it will be ignored
# everything below 0.8 is BS
MIN_OVERLAP = 0.8

MASK_WIDTH = 21
MASK_HEIGHT = 75

#[88, 92]
PERPENDICULARITY_ACCEPTED_THRESHOLD_IN_DEGREES = 2;

def show_with_wait(image, name=""):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def remove_files_in_dir(directory):
    files = os.listdir(directory)
    
    for file in files:
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Removed {file_path}")

# Standard image manipulation
# Adjust the values (180, 255) if you need to;
# 180, 255 worked the best on my test data
def pre_process_original_image(image, debug = False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded_image = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    return thresholded_image

# We need to be able to ignore_card_dimensions, because for the first card we don't know the dimensions
def find_contours_of_possible_cards(processed_image, ignore_card_dimensions = False):
    contours, hierarchy = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    possible_cards_contours = []
    # sort from left to right (X axis)
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])
    
    for index, contour in enumerate(contours):
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
        # We want to return only "possible" rectangles
        # Notice that I'm using >= 2 instead of >= 4, because I'm expecting the cards to overlap, but usually the top is visible
        if len(approx) >= 2:
            x, y, w, h = cv2.boundingRect(contour)

            if ignore_card_dimensions:
                possible_cards_contours.append(contour)
            elif(w >= CARD_WIDTH /2 and h >= CARD_HEIGHT / 4):
                possible_cards_contours.append(contour)

    return possible_cards_contours

# Important: 90 angle means that the contour is perpendicular with X axis
# Yes, I found out the hard way
def find_rotation_angles(contours):
    rotation_angles = []

    for index, contour in enumerate(contours):
        rotation_angles.append(find_rotation_angle(contour))
    
    return rotation_angles

def find_coordinates(contours):
    coordinates = []

    for index, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        coordinates.append((x, y, w, h))
    
    return coordinates

def find_rotation_angle(contour):
    rect = cv2.minAreaRect(contour)
    angle = rect[2]
    return angle

# Check the docs if you want to test with a different interpolation
# Usually it's a tradeoff between performance and quality
def zoom_image(processed_image, zoom_factor):
    zoomed_image = cv2.resize(processed_image, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)
    return zoomed_image

# The information we are looking for is in the card header
# => we can ignore the rest of the card
# Important: card = possible card; it can be anything with a contour
def extract_possible_cards_headers(processed_image, cards_coordinates):
    possible_card_header = []
    for card in cards_coordinates:
        x, y, w, h = card
        # y:y+h, x:x+w was the initial implementation, but cropping the whole card will only slow down the matchTemplate
        # make sure that you crop at least the size of the mask (you cannot match a mask against an image that is smaller than the mask)
        possible_card_header.append(processed_image[y:y+max(MASK_HEIGHT * 4, int(h/2)), x:x+max(MASK_WIDTH * 2, int(w/2))])

    return possible_card_header;

# Rotates the image at a given angle
# The extra math (cos, sin etc) tries to rotate the image without changing it's center/pivot
# Important: I'm expecting to gaps to be filled with black pixels (e.g. if you rotate a rectangle at 45degrees, the content will rotate but in the corners you'll have "gaps")
def rotate(image, angle):
    if(angle == 0):
        return image
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width/ 2, height / 2), angle, 1)
    
    # Calculate the dimensions of the rotated image
    cos_theta = np.abs(rotation_matrix[0, 0])
    sin_theta = np.abs(rotation_matrix[0, 1])
    new_width = int((height * sin_theta) + (width * cos_theta))
    new_height = int((height * cos_theta) + (width * sin_theta))

    rotation_matrix[0, 2] += (new_width / 2) - (width / 2)
    rotation_matrix[1, 2] += (new_height / 2) - (height / 2)

    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    return rotated_image

# Returns the rotation needed in order to make the line perpendicular on the X axis
def calculate_angle(line):
    x1, y1, x2, y2 = line
    delta_y = y2 - y1
    delta_x = x2 - x1
    return np.arctan2(delta_y, delta_x) * 180 / np.pi

# Chatgpt
# TODO: check if it's ok
# Function to calculate the distance between two points
def distance(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

# Function to check if a line is close to any other line
def is_close_to_any(line, lines, threshold):
    x1, y1, x2, y2 = line
    for other_line in lines:
        x1_other, y1_other, x2_other, y2_other = other_line
        if (distance((x1, y1), (x1_other, y1_other)) < threshold and
                distance((x2, y2), (x2_other, y2_other)) < threshold):
            return True
    return False

def find_lines(processed_image, possible_card_not_processed, index):
    edges = cv2.Canny(processed_image, 50, 150)
    lines_with_duplicates = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=20)
    
    # each line is wrapped with an extra array that I don't need => spreading the values to get the following form [[x1, y1, x2, y2], [x1, y1, x2, y2], etc]
    lines_with_duplicates_spread = []
    for line in lines_with_duplicates:
        lines_with_duplicates_spread.append(line[0])
    
    lines = []
    for line in lines_with_duplicates_spread:
        # TODO: threshold should be calculated based on the original image's size
        if not is_close_to_any(line, lines, threshold=50):
            lines.append(line)

    return lines

# an alternative of find contours and extract rect
# currently not used
def crop_between_points(image, points):
    # Convert to numpy arrays
    points = np.array(points)

    # Calculate the width and height of the rectangle
    width_top = np.sqrt(((points[1][0] - points[0][0]) ** 2) + ((points[1][1] - points[0][1]) ** 2))
    width_bottom = np.sqrt(((points[2][0] - points[3][0]) ** 2) + ((points[2][1] - points[3][1]) ** 2))
    width = int((width_top + width_bottom) / 2)

    height_left = np.sqrt(((points[0][0] - points[3][0]) ** 2) + ((points[0][1] - points[3][1]) ** 2))
    height_right = np.sqrt(((points[1][0] - points[2][0]) ** 2) + ((points[1][1] - points[2][1]) ** 2))
    height = int((height_left + height_right) / 2)

    # Define the perspective transformation matrix
    source_pts = np.float32(points)
    dest_pts = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    matrix = cv2.getPerspectiveTransform(source_pts, dest_pts)

    # Apply the perspective transformation to crop the region
    cropped_image = cv2.warpPerspective(image, matrix, (width, height))

    return cropped_image

def find_lines_that_intersect_at_90(lines):
    all_angles = []
    for line in lines:
        all_angles.append(calculate_angle(line))

    #[(lineA, lineB), (lineC, lineD), etc] e.g. [(x1, y1, x2, y3), (x1, y1, x2, y2), etc]
    tuple_of_lines_that_intersect = []

    #[(angleLineA, angleLineB), (angleLineC, angleLineD), etc]
    tuple_angles_of_lines_that_intersect = []

    print('all_angles', all_angles)
    angles_used = []
    for index1, angle1 in enumerate(all_angles):
        if angle1 in angles_used:
            continue
        for index2, angle2 in enumerate (all_angles):
            if angle2 in angles_used:
               continue
            # skip if already processed
            if(index2 <= index1):
                continue
            angle1_abs = abs(angle1)
            angle2_abs = abs(angle2)
            angles_sum = angle1_abs + angle2_abs;
            # check if the sum is between [89, 91]
            if(
                angles_sum <= 90 + PERPENDICULARITY_ACCEPTED_THRESHOLD_IN_DEGREES
                and
                angles_sum >= 90 - PERPENDICULARITY_ACCEPTED_THRESHOLD_IN_DEGREES
            ):
                # store a tuple containing the 2 lines which intersect
                tuple_of_lines_that_intersect.append((lines[index1], lines[index2]))
                tuple_angles_of_lines_that_intersect.append((angle1, angle2))
                
                angles_used.append(angle1)
                angles_used.append(angle2)
    
    return tuple_of_lines_that_intersect, tuple_angles_of_lines_that_intersect

# Step 1: Find contours (ideally there should be only 1 contour, but if there are multiple, the one with the greater width will be used)
# Step 2: Find the rotation angle
# Step 3: Rotate the card to be perpendicular with X axis
# Step 4: Find the contours of the rotated card
# Step 5: Extract the width (with find_coordinates)
def find_card_dimensions(possible_card, possible_card_not_processed):
    contours = find_contours_of_possible_cards(possible_card)
    
    max_width = -1
    max_height = -1
    for contour in contours:
        rotation_angle = find_rotation_angle(contour)

        possible_card_rotated = rotate(possible_card, rotation_angle)
        possible_card_not_processed_rotated = rotate(possible_card_not_processed, rotation_angle)

        contours_of_possible_card_rotated = find_contours_of_possible_cards(possible_card_rotated, True)

        cv2.drawContours(possible_card_not_processed_rotated, contours_of_possible_card_rotated, -1, (0, 255, 0), 3)

        width = -1
        height = -1
        coordinates = find_coordinates(contours_of_possible_card_rotated)
       
        for coordinate in coordinates:
             x, y, w, h = coordinate
             print('x', x)
             if(w > width):
                 x1 = x
                 y1 = y
                 width = w
                 height = h
        
        # width is our reference
        # max_height is the height corresponding to the max_width
        if(max_width < width):
            max_width = width
            max_height = height

    return max_width, max_height

# What we want to achieve here:
# - make the cards perpendicular with X axis (the smaller lateral(s) to be parallel with X and the longest one(s) to be perpendicular)
# Rotate the possible card n * 2 times, where n is the number of detected corners (each element of tuple_of_lines_that_intersect_at_90 is considered to be a corner)
# * 2 because we don't know which angle will rotate the card to the desired position (e.g. maybe -89.1 or it's match 0.9), so we have to rotate with both angles
# Later update:
# It's n * 2 * 2 to be more accurate (because we have to flip de possible cards by 180 too)
# in order to cover the case when the upper part of the card is not visible, but the lower part is
def rotate_possible_cards(possible_cards, possible_cards_not_processed):
    possible_cards_rotated = []
    possible_cards_not_processed_rotated = []
    
    for index, possible_card in enumerate(possible_cards):
        lines = find_lines(possible_card, possible_cards_not_processed[index], index)

        tuple_of_lines_that_intersect_at_90, tuple_of_angles_of_lines_that_intersect = find_lines_that_intersect_at_90(lines)
        
        for index2, tuple_of_angles in enumerate(tuple_of_angles_of_lines_that_intersect):
            angle1, angle2 = tuple_of_angles[:2];
            possible_card_rotated_angle1 = rotate(possible_card, angle1)
            possible_card_rotated_angle2 = rotate(possible_card, angle2)


            possible_card_rotated_angle1_180 = rotate(possible_card_rotated_angle1, 180)
            possible_card_rotated_angle2_180 = rotate(possible_card_rotated_angle2, 180)

            possible_cards_rotated.append(possible_card_rotated_angle1)
            possible_cards_rotated.append(possible_card_rotated_angle2)
            possible_cards_rotated.append(possible_card_rotated_angle1_180)
            possible_cards_rotated.append(possible_card_rotated_angle2_180)

            possible_card_not_processed = possible_cards_not_processed[index]
            possible_card_not_processed_rotated_angle1 = rotate(possible_card_not_processed, angle1)
            possible_card_not_processed_rotated_angle2 = rotate(possible_card_not_processed, angle2)


            possible_card_not_processed_rotated_angle1_180 = rotate(possible_card_not_processed_rotated_angle1, 180)
            possible_card_not_processed_angle2_180 = rotate(possible_card_not_processed_rotated_angle2, 180)

            possible_cards_not_processed_rotated.append(possible_card_not_processed_rotated_angle1)
            possible_cards_not_processed_rotated.append(possible_card_not_processed_rotated_angle2)
            possible_cards_not_processed_rotated.append(possible_card_not_processed_rotated_angle1_180)
            possible_cards_not_processed_rotated.append(possible_card_not_processed_angle2_180)

    return possible_cards_rotated, possible_cards_not_processed_rotated

def extract_possible_cards(processed_image, image):
    possible_cards = []
    possible_cards_not_processed = []
    contours = find_contours_of_possible_cards(processed_image)
    possible_card_coordinates = find_coordinates(contours)

    for coordinates in possible_card_coordinates:
        x, y, w, h = coordinates
        possible_card = processed_image[y:y+h, x:x+w]
        possible_cards.append(possible_card)
        possible_cards_not_processed.append(image[y:y+h, x:x+w])

    return possible_cards, possible_cards_not_processed;

def extract_cards_headers(possible_cards, possible_cards_not_processed, image_name):
    processed_possible_cards_headers = []

    for index, possible_card in enumerate(possible_cards):
       
        contours = find_contours_of_possible_cards(possible_card)

        cards_coordinates = find_coordinates(contours)
        possible_cards_headers = extract_possible_cards_headers(possible_card, cards_coordinates)
        for index2, possible_card_header in enumerate(possible_cards_headers):
            processed_possible_cards_headers.append(possible_card_header)

            cv2.imwrite(f'./generated/000_{image_name[0]}_{index}_{index2}_.jpg', possible_card_header)
        
    return processed_possible_cards_headers

# I believe this is a copy paste from opencv docs
# You can change the method if you want, but I've noticed the best accuracy with TM_CCOEFF_NORMED
def match_against_mask(image, mask):
    method = cv2.TM_CCOEFF_NORMED

    h, w = mask.shape[:2]

    img = np.copy(image)
    res = cv2.matchTemplate(img ,mask, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    bottom_right = (top_left[0] + w, top_left[1] + h)      
    return top_left, bottom_right, max_val


def match_against_all_masks(image, masks_directory):
    # Mapping the results:
    # key = card identifier (prefix of the mask) - e.g. 6s, 6h, 6c, 6d, etc

    # max_values_dict <- here we are storing the predictions
    max_values_dict= {}
    # some of the below are used only for debugging; I recommend to not erase anything from here
    max_lefts_dict = {}
    max_rights_dict = {}
    max_mask_dict = {}
    max_name = {}
    max_image = {}
    max_zoom_dict = {}
    processed_masks_dict = {}

    # Here we are applying transformations on the masks
    for mask_name in os.listdir(masks_directory):
        # because macOS;
        if mask_name != ".DS_Store":
            mask = cv2.imread(f'{masks_directory}/{mask_name}')
            processed_masks_dict[mask_name] = pre_process_original_image(mask)
    
    # Here we are running the detection
    for mask_name, processed_mask in processed_masks_dict.items():
            current_zoom_factor = 1
            file_prefix = f'{mask_name[0]}{mask_name[1]}'
            
            top_left, bottom_right, max_val = match_against_mask(image, processed_mask)

            if max_val < MIN_OVERLAP:
                continue
            if(file_prefix not in max_values_dict):
                max_values_dict[file_prefix] = max_val
                max_name[file_prefix] = mask_name
                max_lefts_dict[file_prefix] = top_left
                max_rights_dict[file_prefix] = bottom_right
                max_mask_dict[file_prefix] = processed_mask
                max_image[file_prefix] = image
                max_zoom_dict[file_prefix] = current_zoom_factor
            elif(max_val > max_values_dict[file_prefix]):
                max_values_dict[file_prefix] = max_val
                max_name[file_prefix] = mask_name
                max_lefts_dict[file_prefix] = top_left
                max_rights_dict[file_prefix] = bottom_right
                max_mask_dict[file_prefix] = processed_mask
                max_image[file_prefix] = image
                max_zoom_dict[file_prefix] = current_zoom_factor

    return max_values_dict, max_name, max_lefts_dict, max_rights_dict, max_mask_dict, max_image, max_zoom_dict

# This can be improved
# The scope of this function is to extract the best guess for a given zone
def mark_overlaps(max_values_dict, max_name, max_lefts_dict):
    DELTA = 10
    for file_prefix, _ in max_name.items():
        for file_prefix2, _ in max_name.items():

            # sometimes the rectangles won't overlap perfectly
            if(
                file_prefix != file_prefix2
                and
                max_values_dict[file_prefix] > 0
                and
                max_values_dict[file_prefix2] > 0
                and
                (max_lefts_dict[file_prefix][0] + DELTA >= max_lefts_dict[file_prefix2][0]) 
                and
                (max_lefts_dict[file_prefix][0] + DELTA >= max_lefts_dict[file_prefix2][0]) 
                and 
                (max_lefts_dict[file_prefix][0] - DELTA <= max_lefts_dict[file_prefix2][0])
                ):
                if(max_values_dict[file_prefix] >= max_values_dict[file_prefix2]):
                    max_values_dict[file_prefix2] = -2
                else:
                    max_values_dict[file_prefix] = -2
    return max_values_dict

def remove_negatives(max_values_dict, max_name):
    for file_prefix, _ in max_name.items():
        if max_values_dict[file_prefix] < 0:
            # TODO: do not modify max_values_dict
            del max_values_dict[file_prefix]
    
    return max_values_dict

