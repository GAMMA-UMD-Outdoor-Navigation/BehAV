import requests
import base64
import time
import numpy as np
import cv2
import io
import os 
import sys
import re 
from PIL import Image
from skimage.measure import regionprops
# from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
from requests.exceptions import RequestException
import torch
print(torch.cuda.is_available())
from FastSAM.fastsam.model import FastSAM
from FastSAM.fastsam.prompt import FastSAMPrompt

""" Please Update the folders for the following before running the code
    -save_plot
    - ground_truth image
    - test_image
    - model checkpoints
    """
class LandmarkDetector:
    def __init__(self, api_key, ground_truth_image_path, test_image_path):
        self.api_key = api_key
        self.ground_truth_image_path = ground_truth_image_path
        self.test_image_path = test_image_path
        # self.show_masked_image = True
        self.image_plot = True
        self.save_image_plot = "YOUR FILE PATH HERE"
        # self.save_path_mask = "/home/vignesh/Gamma/output_images/masks/"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = FastSAM('/home/vignesh/Gamma/Landmark_detector/FastSAM-x.pt')
        self.max_retries = 3
        self.delay = 5

    def get_next_file_number(self):
        files = os.listdir(self.save_image_plot)
        existing_numbers = []
        for f in files:
            if f.startswith('Image_plots_') and f.endswith('.jpg'):
                match = re.search(r'Image_plots_(\d+)\.jpg', f)
                if match:
                    existing_numbers.append(int(match.group(1)))
        return max(existing_numbers, default=0) + 1

    def save_images(self,original_img, masked_img, circled_img, mask_number, distance):
        os.makedirs(self.save_image_plot, exist_ok=True)
        # os.makedirs(self.base_path_mask, exist_ok=True)
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.imshow(original_img)
        plt.title('Original image')
        plt.axis('off')
        plt.subplot(132)
        plt.imshow(masked_img)
        plt.title('Masked image')
        plt.axis('off')
        plt.subplot(133)
        plt.imshow(circled_img)
        plt.title('Circled image')
        plt.axis('off')
        plt.tight_layout()
        next_number = self.get_next_file_number()
        img_plot_filename = f"Image_plots_{next_number}.jpg" 
        self.save_path_plot = os.path.join(self.save_image_plot, img_plot_filename)
        plt.figtext(0.06, 0.95, f"Mask Number: {mask_number}, \nDistance: {distance}", 
                    ha="center", va = "top", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
        plt.savefig(self.save_path_plot)
        plt.close()

    def make_api_request(self, headers, data):
        for attempt in range(self.max_retries):
            try:
                response = requests.post("https://api.openai.com/v1/chat/completions", headers= headers, json=data, timeout=30)
                response.raise_for_status()
                return response.json()
            except RequestException as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt + 1 < self.max_retries:
                    print(f"Retrying in {self.delay} seconds...")
                    time.sleep(self.delay)
                else:
                    print("Max retries reached. Unable to get a response.")
                    return None
                
    def load_image_path(self, file_path):
        with open(file_path, "rb") as image_file:
            return f"data:image/jpeg;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"
    
    def load_image(self, image):
        pil_image = Image.fromarray(image)
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        return f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"


    def get_mask_coordinates(self, masks, target_mask_number):
        if target_mask_number < 1 or target_mask_number > len(masks):
            raise ValueError(f"Invalid mask number: {target_mask_number}. Should be between 1 and {len(masks)}")
        else:
            # print(target_mask_number)
            mask = masks[target_mask_number - 1]
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        binary_mask = (mask > 0).astype(int)
        regions = regionprops(binary_mask)
        if len(regions) == 0:
            raise ValueError("No regions found in the binary mask. The mask might be empty.")
        for i, region in enumerate(regions):
            centroid = region.centroid  # (y, x) coordinates
            bbox = region.bbox  # (min_row, min_col, max_row, max_col)
            
            # Calculate the center of the bounding box
            center_y = (bbox[0] + bbox[2]) // 2
            center_x = (bbox[1] + bbox[3]) // 2
        region = regions[0]
        centroid = region.centroid
        bbox = region.bbox
        center_y = (bbox[0] + bbox[2]) // 2
        center_x = (bbox[1] + bbox[3]) // 2
        return (center_x, center_y)   

    def plot_dot_on_image(self, image, x, y, radius=40, color=(255, 0, 0), thickness=-1):
        # if isinstance(image, str):
        #     image = cv2.imread(image)
        if not isinstance(image, np.ndarray):
            raise ValueError("The 'image' argument must be a numpy array or a valid file path.")
        circled_img = image.copy()
        cv2.circle(circled_img, (int(x), int(y)), radius, color, thickness)
        return circled_img

    def run(self):
        start_time = time.time()
        image_feed = cv2.imread(self.test_image_path)
        image = image_feed
        # print(image.shape)
        image = cv2.cvtColor(image_feed, cv2.COLOR_BGR2RGB)
        
        results = self.model(image, device=self.device, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
        prompt_process = FastSAMPrompt(image, results, device=self.device) 
        ann = prompt_process.everything_prompt() #These annotations contain all the masks 
        # print(ann.shape)
        masked_image, merged_ann = prompt_process.plot(annotations=ann, distance_threshold=200) # These annotations contains only the masks after the threshold
        prompt_presence = """Compare two images:
            1. A ground truth image of a landmark building.
            2. A test image.

            Task 1: Is the landmark present in the test image
            Example Response format for Task 1: 
            Landmark - YES/NO

            If YES, proceed to Tasks 2 and 3:

            Task 2: Identify the mask number containing the landmark (even partially).
            Task 3: Estimate the camera distance from the landmark in the test image. 
            Based on on the Ground truth image of a landmark building which is taken from 21.38 meters away from the camera. Give me a number and no text

            Example response format for Task 2 and Task 3:
            Mask number: 3
            Distance: 120 meters """
        
    #iribe- 80 meters 
    #Douglass - 15 meters
    #M-circle-  21.38 meters
    #Idea - 29.1 meters
    #Testudo - 3 meters
    #chapel- 32 meters
        
        headers = {
        'Authorization': f'Bearer {self.api_key}',
        'Content-Type': 'application/json'
        }
        data_presence = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_presence},
                        {"type": "image_url", "image_url": {"url": self.load_image_path(self.ground_truth_image_path)}},
                        {"type": "image_url", "image_url": {"url": self.load_image(masked_image)}}
                    ]
                }
            ],
            "max_tokens": 300
        }
        response_json = self.make_api_request(headers, data_presence)
        # print("API Response:", response_json)
        if response_json:
            response_text = response_json['choices'][0]['message']['content'].strip()
            # present = response_text.split(' ')[-1]  
            landmark_status = response_text.split('-')[1].strip().split('\n')[0]
            # print(f'The landmark is {present} in the image')
            print(response_text)
            
            if landmark_status == 'YES':
                # print((response_text['choices'][0]['message']['content'][13].strip()))
                # mask_number_line = [line for line in response_text.split('\n') if 'Mask number:' in line][0].split(':')
                mask_number_line = [line for line in response_text.split('\n') if 'Mask number:' in line][0]
                mask_number = re.search(r'Mask number:\s*(\d+)', mask_number_line)
                distance_line = [line for line in response_text.split('\n') if 'Distance:' in line][0]
                distance_number = distance_line.split(':')[1].strip()
                # print(distance_number)
                if mask_number:
                    mask_number = int(mask_number.group(1))
                    # print(f"Extracted mask number: {mask_number}")
                else:
                    print("Mask number not found")
                # print(mask_number_line)
                target_mask_number = mask_number
                # print(ann)
                coordinates = self.get_mask_coordinates(merged_ann, target_mask_number)
                # print(coordinates)
                # target_mask_number = response_text.split('-')
                # print(target_mask_number)
                # pixel_location = self. get_mask_coordinates(ann, target_mask_number)
                X , Y = coordinates[0], coordinates[1] 
                # print(f"The pixel location of the target [X ={X}, Y = {Y}]")
                circled_img = self.plot_dot_on_image(image, X, Y )
                # print(circled_img.shape)
                if self.image_plot:
                    self.save_images(image,masked_image,circled_img, target_mask_number,distance_number )
        else:
            print("Unable to determine mask number or distance as the landmark building is not present in the given image.")

        end_time = time.time()
        print(f"Elapsed time: {end_time - start_time} seconds")


api_key = ""
ground_truth_image_path = "YOUR FILE PATH HERE"
test_image_path = "YOUR FILE PATH HERE"

detector = LandmarkDetector(api_key, ground_truth_image_path, test_image_path)
detector.run()

