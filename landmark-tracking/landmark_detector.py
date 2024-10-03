import requests
import base64
import time
import numpy as np
import cv2
import io
from PIL import Image
from skimage import measure
from skimage.measure import regionprops, find_contours
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
from requests.exceptions import RequestException
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

class LandmarkDetector:
    def __init__(self, api_key, ground_truth_image_path, test_image_path):
        self.api_key = api_key
        self.ground_truth_image_path = ground_truth_image_path
        self.test_image_path = test_image_path
        
        # Initialize headers for API requests
        # self.headers = {
        #     'Authorization': f'Bearer {self.api_key}',
        #     'Content-Type': 'application/json'
        # }
        
        print("Is CUDA Available: ",torch.cuda.is_available())
        # SAM Model for Segmentaion
        self.sam_checkpoint = "./sam_models/sam_vit_b_01ec64.pth"
        self.model_type = "vit_b"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        self.sam.to(device=self.device)
        
        # Other configuration parameters
        self.max_retries = 3
        self.delay = 5
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=8,
            points_per_batch=100000,
            pred_iou_thresh=0.80,
            stability_score_thresh=0.5,
            crop_n_layers=1,
            crop_overlap_ratio=0.2,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=10000,
            output_mode='binary_mask'
        )

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
    from skimage.measure import regionprops

    def get_mask_coordinates(self, masks, target_mask_number):
        if target_mask_number < 1 or target_mask_number > len(masks):
            raise ValueError(f"Invalid mask number: {target_mask_number}. Should be between 1 and {len(masks)}")
        mask = masks[target_mask_number - 1]['segmentation']
        region = regionprops(mask.astype(int))[0]
        centroid = region.centroid  # (y, x) coordinates
        bbox = region.bbox 
        return centroid, bbox

    
    def process_and_display_masks(self, masks, image, distance_threshold=100):
        merged_masks = []
        used_masks = set()

        for i, mask in enumerate(masks):
            if i in used_masks:
                continue
            merged_mask = mask['segmentation'].copy()
            for j, other_mask in enumerate(masks[i+1:], start=i+1):
                if j in used_masks:
                    continue
                if np.any(binary_dilation(merged_mask, iterations=distance_threshold) & other_mask['segmentation']):
                    merged_mask |= other_mask['segmentation']
                    used_masks.add(j)

            merged_masks.append({
                'segmentation': merged_mask,
                'area': np.sum(merged_mask),
                'bbox': regionprops(merged_mask.astype(int))[0].bbox
            })

        if len(merged_masks) == 0:
            return image

        sorted_anns = sorted(merged_masks, key=lambda x: x['area'], reverse=True)
        mask_properties = []

        for idx, ann in enumerate(sorted_anns):
            m = ann['segmentation']
            contours = measure.find_contours(m, 0.5)
            for contour in contours:
                contour = np.array(contour, dtype=np.int32)
                cv2.polylines(image, [contour[:, [1, 0]]], isClosed=True, color=(255, 255, 255), thickness=int(1.5))

            y_coords, x_coords = np.where(m)
            centroid_x = np.mean(x_coords)
            centroid_y = np.mean(y_coords)

            mask_properties.append({
                'centroid': (centroid_x, centroid_y),
                'area': ann['area']
            })

        mask_properties.sort(key=lambda x: (x['centroid'][1], x['centroid'][0]))

        # Numbering
        for idx, prop in enumerate(mask_properties):
            cv2.putText(image, str(idx + 1), (int(prop['centroid'][0]), int(prop['centroid'][1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)

        return image

    def run(self):
        start_time = time.time()

        prompt_presence = """
        I have two images:
        1. A ground truth image of a landmark building.
        2. A test image.

        Tasks:
        1. Identify the landmark building in the second image with the help of the first image. Tell that the building is present YES or NO in the second image.

        Please respond ONLY with one piece of information:
        1. The building is present in the image with YES or NO

        Example response format:
        Landmark - YES or NO
        """
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
                        {"type": "image_url", "image_url": {"url": self.load_image_path(self.test_image_path)}}
                    ]
                }
            ],
            "max_tokens": 300
        }
        # print(f"Headers: {headers}")
        # print(f"Data: {data_presence}")
        response_json = self.make_api_request(headers, data_presence)
        if response_json:
            response_text = response_json['choices'][0]['message']['content'].strip()
            present = response_text.split(' ')[-1]  

            if present.upper() == "YES":
                print("The landmark is present in the image.")
                
                # Use SAM to find mask number and distance
                image = cv2.imread(self.test_image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                masks = self.mask_generator.generate(image)
                
                masked_image = self.process_and_display_masks(masks, image)
                
                prompt_mask_distance = """
                I have two images: 
                1. A ground truth image of a landmark building taken from a known distance of 80 meters.
                2. A masked image with numbered masks, taken from an unknown distance.

                Tasks:
                1. Identify which numbered mask in the second image contains the landmark building from the first image. Even if only a small portion of the building is visible, return that mask number.
                2. Based on the relative size and appearance of the landmark in both images, estimate the approximate distance of the camera from the landmark in the masked image. Use the known 80-meter distance of the first image as a reference.

                Please respond ONLY with two pieces of information:
                1. The mask number where the landmark is located
                2. The estimated distance in meters

                Example response format:
                Mask number: 3
                Distance: 120 meters
                """

                data_mask_distance = {
                    "model": "gpt-4-vision-preview",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_mask_distance},
                                {"type": "image_url", "image_url": {"url": self.load_image_path(self.ground_truth_image_path)}},
                                {"type": "image_url", "image_url": {"url" : self.load_image(masked_image)}}
                            ]
                        }
                    ],
                    "max_tokens": 300
                }

                response_json_mask_distance = self.make_api_request(headers, data_mask_distance)
                if response_json_mask_distance:
                    response_text_mask_distance = response_json_mask_distance['choices'][0]['message']['content'].strip()
                    # print()
                    target_mask_number = int(response_json_mask_distance['choices'][0]['message']['content'][13].strip())
                    pixel_location = self. get_mask_coordinates(masks, target_mask_number)
                    print(f"The pixel location of the target [X ={pixel_location[0][1]}, Y = {pixel_location[0][0]}]")
                    print(response_text_mask_distance)
                else:
                    print("Unable to determine mask number or distance.")

            else:
                print("The landmark is NOT present in the image.")
        else:
            print("Unable to determine if the landmark is present.")

        end_time = time.time()
        print(f"Elapsed time: {end_time - start_time} seconds")


api_key = ""
ground_truth_image_path = "./images/gt.jpg"
test_image_path = "./images/landmark.jpg"

detector = LandmarkDetector(api_key, ground_truth_image_path, test_image_path)
detector.run()
