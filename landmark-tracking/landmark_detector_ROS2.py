import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import requests
import base64
import time
import numpy as np
import cv2
import io
import os 
import re
from PIL import Image as PILImage
from skimage import measure
from skimage.measure import regionprops
from scipy.ndimage import binary_dilation
import torch
print(torch.cuda.is_available())
import matplotlib.pyplot as plt
from requests.exceptions import RequestException
# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from FastSAM.fastsam.model import FastSAM
from FastSAM.fastsam.prompt import FastSAMPrompt

# self.show_image_popup(image_rgb, X, Y)
class LandmarkDetectorNode(Node):
    def __init__(self):
        super().__init__('landmark_detector_node')
        self.get_logger().info('Started the Node.')
        # Parameters
        self.declare_parameter('api_key', '') #ADD YOUR API KEY HERE
        self.declare_parameter('ground_truth_image_path', 'Images/Iribe/2.jpg')
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        
        self.api_key = self.get_parameter('api_key').get_parameter_value().string_value
        self.ground_truth_image_path = self.get_parameter('ground_truth_image_path').get_parameter_value().string_value
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.timer = self.create_timer(10.0, self.timer_callback)
        self.latest_image = None
        self.is_processing = False
        # Initialize subscriber
        self.publisher = self.create_publisher(Image, '/processed_image', 1
                                               )
        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            1
        )
        self.bridge = CvBridge()

        print("Is CUDA Available: ", torch.cuda.is_available())

        # FastSAM Model for Segmentation
        self.image_plot = True
        self.save_image_plot = "./Image_plots/"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = FastSAM('./Landmark_detector/FastSAM-x.pt')
        self.max_retries = 3
        self.delay = 5

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            read = image_file.read()
            encode = base64.b64encode(read)
            decoded = encode.decode('utf-8')
        return decoded

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if msg.encoding != 'bgr8':
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            self.latest_image = cv_image
        except CvBridgeError as e:
            self.get_logger().error(f'cv_bridge error: {str(e)}')
            # Try manual conversion
            try:
                np_arr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
                if msg.encoding == 'rgb8':
                    cv_image = cv2.cvtColor(np_arr, cv2.COLOR_RGB2BGR)
                elif msg.encoding == 'mono8':
                    cv_image = np_arr
                else:
                    cv_image = np_arr  # Use as-is for other encodings
                
                self.get_logger().info(f"Manual conversion successful. Shape: {cv_image.shape}")
                self.latest_image = cv_image
            except Exception as e2:
                self.get_logger().error(f'Error during manual conversion: {str(e2)}')
                return 

    def timer_callback(self):
        if self.latest_image is not None and not self.is_processing:
            self.is_processing = True
            self.get_logger().info('Image received.')
            self.get_logger().info('Processing image...')
            self.process_image(self.latest_image)
            # self.latest_image = None
            # self.is_processing = False
        else:
            self.get_logger().info('No new image to process or still processing previous image.')

    def publish_processed_image(self, cv_image):
        try:
            ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
            # Publish the image
            self.publisher.publish(ros_image)
            self.get_logger().info('Published processed image')
        except CvBridgeError as e:
            self.get_logger().error(f'Error converting image: {str(e)}')

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
            # plt.show()
            plt.savefig(self.save_path_plot)
            plt.close()

    def make_api_request(self, headers, data):
        for attempt in range(self.max_retries):
            try:
                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
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
        pil_image = PILImage.fromarray(image)
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        return f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"

    def get_mask_coordinates(self, masks, target_mask_number):
        target_mask_number = int(target_mask_number)
        print(f"Number of masks available: {len(masks)}")
        # print(masks)
        if target_mask_number < 1 or target_mask_number > len(masks):
            raise ValueError(f"Invalid mask number: {target_mask_number}. Should be between 1 and {len(masks)}")
        else:
            print(target_mask_number)
            mask = masks[target_mask_number - 1]
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        binary_mask = (mask > 0).astype(int)
        regions = regionprops(binary_mask)
        if len(regions) == 0:
            raise ValueError("No regions found in the binary mask. The mask might be empty.")
        
        # Loop through all regions and print their properties
        for i, region in enumerate(regions):
            # centroid = region.centroid  # (y, x) coordinates
            bbox = region.bbox  # (min_row, min_col, max_row, max_col)
            
            # Calculate the center of the bounding box
            center_y = (bbox[0] + bbox[2]) // 2
            center_x = (bbox[1] + bbox[3]) // 2
        region = regions[0]
        # centroid = region.centroid
        bbox = region.bbox
        center_y = (bbox[0] + bbox[2]) // 2
        center_x = (bbox[1] + bbox[3]) // 2
        return (center_x, center_y)

    def plot_dot_on_image(self, image, x, y, radius=50, color=(255, 0, 0), thickness=-1):
        # if isinstance(image, str):
        #     image = cv2.imread(image)
        if not isinstance(image, np.ndarray):
            raise ValueError("The 'image' argument must be a numpy array or a valid file path.")
        image_copy = image.copy()

        # Draw the circle
        cv2.circle(image_copy, (int(x), int(y)), radius, color, thickness)
        # cv2.imshow('Image with Dot', image_copy)
        # cv2.imwrite(self.save_path_HD, image_copy)
        return image_copy
    
    def process_image(self, image):
        start_time = time.time()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.model(image, device=self.device, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
        prompt_process = FastSAMPrompt(image, results, device=self.device) 
        ann = prompt_process.everything_prompt() #These annotations contain all the masks 
        # print(ann.shape)
        masked_image, merged_ann = prompt_process.plot(annotations=ann, distance_threshold=500) # These annotations contains only the masks after the threshold 
        prompt_presence = """Compare two images:
            1. A ground truth image of a landmark building.
            2. A test image.

            Task 1: Is the landmark present in the test image?
            Example Response format for Task 1: 
            Landmark - YES/NO

            If YES, proceed to Tasks 2 and 3:

            Task 2: Identify the mask number containing the landmark (even partially). Compare the masked image and the groung truth image and give me mask number which matches exactly with the ground truth image.
            Task 3: Estimate the camera distance from the landmark in the test image. Based on on this data -Ground truth image of a landmark building is taken from 80 meters from the camera.

            Example response format for Task 2 and Task 3:
            Mask number: 3
            Distance: 120 meters
            
            Output the result in the exact format as the example response format """
        
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
        # print(f"Headers: {headers}")
        # print(f"Data: {data_presence}")
        # cv2.imshow("ROS2 Image Viewer", masked_image)
        cv2.waitKey(1) 
        response_json = self.make_api_request(headers, data_presence)
        self.get_logger().info(f"API Response:{response_json}")
        if response_json:
            response_text = response_json['choices'][0]['message']['content'].strip()
            # present = response_text.split(' ')[-1]  
            landmark_status = response_text.split('-')[1].strip().split('\n')[0]

            # print(landmark_status)
            # print(f'The landmark is {present} in the image')
            print(response_text)
            
            if landmark_status == 'YES':
                # print((response_text['choices'][0]['message']['content'][13].strip()))
                # mask_number_line = [line for line in response_text.split('\n') if 'Mask number:' in line][0].split(':')
                mask_number_line = [line for line in response_text.split('\n') if 'Mask number:' in line][0]
                mask_number = re.search(r'Mask number:\s*(\d+)', mask_number_line)
                distance_line = [line for line in response_text.split('\n') if 'Distance:' in line][0]
                distance_number = distance_line.split(':')[1].strip()
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
                X = coordinates[0]
                Y = coordinates[1] 
                circled_img = self.plot_dot_on_image(image, X, Y )
                # print(f"The pixel location of the target [X ={X}, Y = {Y}]")
                if self.image_plot:
                    self.save_images(image,masked_image,circled_img, target_mask_number,distance_number )
                    self.publish_processed_image(circled_img)
                    # cv2.imshow('Processed Image', circled_img)
                    # cv2.waitKey(1000)
        else:
            print("Unable to determine mask number or distance as the landmark building is not present in the given image.")

        end_time = time.time()
        print(f"Elapsed time: {end_time - start_time} seconds")

def main(args=None):
    rclpy.init(args=args)
    node = LandmarkDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down node.')
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
