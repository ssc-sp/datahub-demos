# Databricks notebook source
!pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

# COMMAND ----------

!pip3 install opencv-contrib-python

# COMMAND ----------

import cv2
from dbruntime.patches import cv2_imshow # This is how you show images on Databricks
img = cv2.imread("/dbfs/FileStore/shared_uploads/sean.stilwell@ssc-spc.gc.ca/000447.jpg", cv2.IMREAD_ANYCOLOR) # Read the image
cv2_imshow(img) # Display the image

# COMMAND ----------

# MAGIC %md
# MAGIC # CSI 4533 - Projet
# MAGIC
# MAGIC * Student Name: Sean Stilwell
# MAGIC * Student Number: 300053246
# MAGIC * Date: March 8, 2022
# MAGIC
# MAGIC ## Part 3
# MAGIC
# MAGIC The third part of your project is to use an object detector based on convolutional networks to detect vehicles and pedestrians in the scene.
# MAGIC
# MAGIC This means that the Ground-truth file will not be used at all to track the actors in the scene. This file will only be used to evaluate the quality of the obtained results.
# MAGIC
# MAGIC So add the object detector designed in part 1. The output of your tracker should be the images of the sequence showing the detection boxes with the colors showing the temporal associations. In addition your tracker should produce a detection text file with a format similar to the ground-truth file, i.e. one detection per line with:
# MAGIC
# MAGIC Image_No. Object_ID X Y Width Height Class 
# MAGIC
# MAGIC The object_ID is simply the RGB value of the box color. The class number for pedestrians is 1 and for cars is 3.
# MAGIC
# MAGIC Next you need to calculate the MOTA of your solution. This is done by comparing your detection file with the ground-truth file as follows:
# MAGIC
# MAGIC For each detected object, check if this object has a corresponding object in the ground-truth file (IoU > 0.4). If yes, note the ID of this object in the GT (if there is more than one, consider the object with the highest IoU). Otherwise count this detection as a false positive (FP).
# MAGIC
# MAGIC For each object in the ground-truth, check if this object has a corresponding object in the detection file (IoU > 0.4). If not, count this object as a false negative.
# MAGIC
# MAGIC Finally, for each group of objects with the same label in your detections, count the number of different labels from the ground-truth. This number-1 gives the number of identity changes.
# MAGIC
# MAGIC From these values, it is possible to calculate the MOTA value.
# MAGIC
# MAGIC ## Code
# MAGIC
# MAGIC We start by loading the NN_detector code. This is used directly from lab 3, I declare the identical NN_detector class followed by an adapted run_detector script.

# COMMAND ----------

# SOURCE: LAB 3

from torchvision.models import detection # Used for detection of vehicles
import numpy as np
import torch
from timeit import default_timer as timer

class NN_detector:

	def __init__(self, GPU_detect:bool = False):
		# sélectionner le modèle
		self.raw_model = detection.retinanet_resnet50_fpn
		self.gpu_detect = GPU_detect

		# définir les classes, pour référence
		self.classes = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
				'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
				'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
				'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
				'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
				'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
				'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
				'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
				'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
				'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
				'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
				'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
				]

		# initialiser le périphérique de calcul par défaut
		self.device = torch.device("cpu")
		self.set_device()
		self.model = self.set_model()
		
	def set_device(self):
		# définissez le périphérique de calcul en fonction de la valeur transmise de GPU_detect si possible		
		if self.gpu_detect:
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
			print(self.device)
		device_label = 'GPU' if self.device == torch.device("cuda") else 'CPU'
		print("Using the %s to run the inference" %(device_label))

	def set_model(self):
		# définir les paramètres du modèle
		model = self.raw_model(pretrained=True, progress=True, num_classes=len(self.classes), pretrained_backbone=True).to(self.device)

		# définir le modèle comme gelé
		model.eval()
		return model

	def preprocess_image(self, img):
		# prétraiter l'image à transmettre au modèle
		image = cv2.imread(img)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = image.transpose((2, 0, 1))
		image = np.expand_dims(image, axis=0)
		image = image / 255.0
		image = torch.FloatTensor(image)
		image = image.to(self.device)
		return image

	def detect(self, image):
		image = self.preprocess_image(image)
		start = timer()

		# exécuter le modèle avec l'image sélectionnée
		inference = self.model(image)[0]
		end = timer()

		print('Inference complete in %.4f milisec' %((end - start)*1000))
		inference['boxes'] = inference["boxes"].detach().cpu().numpy()
		
		return inference

# COMMAND ----------

# MAGIC %md
# MAGIC Based directly off the run_detector script, we then perform our detections on the images. I choose to place the results in a list, that way I can make a dataframe identical to the one in the ground truth file, meaning it's easier for me to re-use my old code.

# COMMAND ----------

import cv2              # Image editing library
import pandas as pd     # To read the txt file and to hold the IOU data
import random           # To randomly select colours.
import warnings

warnings.filterwarnings("ignore")

# We constrain the images based on assignment requirements (images 86-467)
IMAGE_START = 1
IMAGE_END = 750
SCORE_THRESHOLD = 0.4
DET = NN_detector(GPU_detect = True)

data = []

# We iterate through the images and run run_detector to predict where each box is and what the value is.
for j in range(IMAGE_START, IMAGE_END):
    path_to_image = '/dbfs/FileStore/shared_uploads/sean.stilwell@ssc-spc.gc.ca/' + str(int(j + 1)).zfill(6) + '.jpg'
    detections = DET.detect(path_to_image)

    for i in range(0, len(detections["boxes"])):
        score = detections["scores"][i]
        id = int(detections["labels"][i])

        # ici, 3 correspond à une voiture, 1 à une personne et 6 à un autobus
        if score > SCORE_THRESHOLD and (id == 3 or id == 6): # Change to id==1 for peds file
            val = 3
            box = detections["boxes"][i]       
            (x1, y1, x2, y2) = box.astype("int")
            row = [j, -1, x1, y1, x2-x1, y2-y1, 0, val]
            data.append(row)


# COMMAND ----------

df = pd.DataFrame(data)
df.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1
# MAGIC
# MAGIC Changed only to use the dataframe built by run_detector rather than the given gt.txt file. Also removed the constraint that width >= 1.2 * height, as our measure is now from the neural network.

# COMMAND ----------

# We then reset the index to start at 0
df = df.reset_index(drop=True)
df.head(5) # We display the first 5 rows of the dataframe for confirmation

# COMMAND ----------

# MAGIC %md
# MAGIC ### Helper Functions
# MAGIC
# MAGIC These functions facilitate the matching between images, using the IOU and matrix.

# COMMAND ----------

# Helper for step 9, determines if we need to continue associating values or if all values are 0 or lower in the matrix.
def df_has_over_zero(df):
    for i in range(1, df.shape[0]):
        for j in range(1, df.shape[1]):
            if df.iloc[i,j] > 0:
                return True
    return False

# COMMAND ----------

# Helper for computing the intersection over union of two bounding boxes
def intersection_over_union(box_1, box_2):
    # We retrieve the row from the dataframe and extract the information
    current = df.iloc[box_1]
    x1 = current[2] # x coordinate of top left corner
    y1 = current[3] # y coordinate of top left corner
    width1 = current[4]
    height1 = current[5]

    # We repeat the above steps for the row of the next image.
    next = df.iloc[box_2]
    x2 = next[2] # x coordinate of top left corner
    y2 = next[3] # y coordinate of top left corner
    width2 = next[4]
    height2 = next[5]

    # This formula waas adapted from: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    xA = max(x1, x2)
    yA = max(y1, y2)
    xB = min(x1+width1, x2+width2)
    yB = min(y1+height1, y2+height2)
    intersect_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    box_1_area = width1 * height1
    box_2_area = width2 * height2

    iou = intersect_area / float(box_1_area + box_2_area - intersect_area)
    return iou

# COMMAND ----------

# MAGIC %md
# MAGIC ### Determining trajectories
# MAGIC
# MAGIC This code conducts the matching between images using the IOU function above. It starts by calculating the IOU over the matrix of current vehicles (columns) and next vehicles (rows). Any values below 0.4 are set to 0. It then iterates again through the matrix while there are values > 0 present, finding the max value at each iteration. Other values in that row/column are set to 0, while the matched ones are set to -1.
# MAGIC
# MAGIC In order to save the results, we use a dictionary called `trajectory_tracker`. For example, if image 1 from the original image is matched with image 32 in the next image, `trajectory_tracker[1] = 32`. 

# COMMAND ----------

trajectory_tracker = {} # Tracks the next image of each box. For example, 1 next appears in 20, 20 next appears in 43, etc.

# We iterate through the images and complete steps 1 - 9 in this loop.
for x in range(IMAGE_START, IMAGE_END):
    df_current_vehicles = df[df[0] == x + 1]    # Get the current image's vehicles boxes (Step 1)
    df_next_vehicles = df[df[0] == x + 2]       # Get the next image's vehicles boxes (Step 2)

    # Create a matrix with current vehicles as columns (i) and next vehicles as rows (j) (Step 3)
    matrix = pd.DataFrame(0, index=df_next_vehicles[0].index, columns=df_current_vehicles[0].index)
    
    # We then compute the Intersection over Union of the current and next vehicles (Step 4). If a value is under 0.4, we will set it to 0. (Step 5)
    for i in range(matrix.shape[1]):        # thru each column
        for j in range(matrix.shape[0]):    # thru each row
            # i, j are positions in matrix. We retrieve the corresponding box values to those positions.
            box_1 = matrix.columns.values[i]
            box_2 = matrix.index.values[j]
            iou = intersection_over_union(box_1, box_2) # We then provide those boxes to the IOU function.
            if iou < 0.4: # Values under 0.4 = discard
                matrix.iloc[j, i] = 0
            else:
                matrix.iloc[j, i] = iou

    # Repeat the following steps until there are no values > 0 in the matrix (Step 9)
    # We then find the maximum value (i,j) in the matrix (Step 6).
    while df_has_over_zero(matrix):
        max_row, max_cell, max_val = -1, -1, -1
        for i in range(matrix.shape[1]):        # thru each column
            for j in range(matrix.shape[0]):    # thru each row
                if matrix.iloc[j, i] > max_val: # If we find a value greater than the current max, that becomes the new max.
                    max_val = matrix.iloc[j, i]
                    max_column = i              # Save the column position
                    max_row = j                 # Save the row position

        #  We set all other values in col i and row j to 0 (Step 7) while i,j is set to -1
        for i in range(matrix.shape[1]): # thru each column
            matrix.iloc[max_row, i] = 0
        for j in range(matrix.shape[0]): # thru each row
            matrix.iloc[j, max_column] = 0
        matrix.iloc[max_row, max_column] = -1

        # We associate box i with box j, adding box j to box i's trajectory (Step 8)
        trajectory_tracker[matrix.columns.values[max_column]] = matrix.index.values[max_row]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Output creation
# MAGIC
# MAGIC With the trajectories of boxes determined, we can then create our output. We start by randomly setting the colours for each box. If that box has a matched box in `trajectory_tracker`, we set the colour of that box to the same colour. If a box already appears in the colours dictionary, we don't need to reset it and can simply set the next corresponding box to that colour.
# MAGIC
# MAGIC After that, we simply iterate through the images for the problem and add a coloured rectangle to each of the images. When finished with all the boxes, we save the new image to the output folder and append it to a video. After finishing, we simply release the video.

# COMMAND ----------

# Boxes i without an associated box are considered ended (Step 10)
# Boxes j without an associated box are considered new (Step 11). 
colors = {} # Shows what each box will be coloured

# Make a video of the results
videoPath = '/home/test_vehicles.mp4'
out = cv2.VideoWriter(videoPath, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (480, 640))

# We iterate through the rows of the dataframe that holds the images.
for x in range(df.shape[0]):
    # New trajectories get a random colour (Step 12)
    if x not in colors:
        colors[x] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    # Previous trajectories get the same colour (Step 13)
    if x in trajectory_tracker:
        colors[trajectory_tracker[x]] = colors[x]

# We then draw in the images to show the boxes (Step 14)
for x in range(IMAGE_START, IMAGE_END): # Iterate through each image
    df_restricted = df[df[0] == x + 1]  # Restrict the dataframe to the current image
    image_key = '/dbfs/FileStore/shared_uploads/sean.stilwell@ssc-spc.gc.ca/' + str(int(x + 1)).zfill(6) + '.jpg'
    output_key = '/home/' + str(int(x + 1)).zfill(6) + '.jpg'
    image = cv2.imread(image_key)       # Read the image from the folder, using the key made above
    
    for y in range(df_restricted.shape[0]): # We then iterate through the boxes contained in the given images
        current = df_restricted.iloc[y]     # Retrieves the current box to be added
        x1 = current[2]         # x coordinate of top left corner
        y1 = current[3]         # y coordinate of top left corner
        width1 = current[4]     # Width of the image
        height1 = current[5]    # Height of the image

        # We can then use our given values to create a start and end point for the box. We also retrieve the colour of the box.
        start_point = (int(x1), int(y1))
        end_point = (int(x1 + width1), int(y1 + height1))
        color = colors[df_restricted.index.values[y]] # Retrives the colour from the dictionary

        # We then draw the box on the image using the start/end points and the color.
        image = cv2.rectangle(image, start_point, end_point, color, 2)
        
        # We also write the image to the output folder.
        # cv2.imwrite(output_key, image)

    out.write(image) # Write image to the video

# Return to step 1, with F(t+1) becoming the new F(t) (Step 15)
out.release() # End video

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2
# MAGIC
# MAGIC Changed since we need to consider that boxes are going to vary now - i.e a predicted box isn't exactly the same as the corresponding accurate box.
# MAGIC
# MAGIC ### Code
# MAGIC
# MAGIC We start by adding an ID to the original dataframe. This allows us to track where a box first originated from and is assigned similarly to how colours are assigned.

# COMMAND ----------

id = [-1] * df.shape[0]
counter = 0
# We first add IDs to each box, similarly to how colours are assigned.
for x in range(df.shape[0]):
    # A newly appearing value gets a new ID
    if id[x] == -1:
        id[x] = counter
        counter += 1

    # Its successors get the same ID
    if x in trajectory_tracker:
        id[trajectory_tracker[x]] = id[x]

df['id'] = id

# COMMAND ----------

df.head()

# COMMAND ----------

# detections = df[df['1', 'id', '2', '3', '4', '5', '7']]
detections = df.filter(['1', 'id', '2', '3', '4', '5', '7'], axis=1)
detections.to_csv('/dbfs/FileStore/shared_uploads/sean.stilwell@ssc-spc.gc.ca/detections_vehicles.csv', index=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### FP, FN, IDS Calculation
# MAGIC
# MAGIC We read the new tracking file that contains information on objects over the series of images. We can then go through the images two at a time in both the old dataframe and the new one to make the following comparisons:
# MAGIC
# MAGIC * If a box appears in the NEW file, the same box (i.e same dimensions) should appear in the other. If not, we increment false negatives.
# MAGIC * If a box appears in the OLD file, the same box (i.e same dimensions) should appear in the other. If not, we increment false positives.
# MAGIC * If a box appears appropriately in BOTH files, we ensure that it has a matching picture in the next images (where needed). If not, we increment IDS.

# COMMAND ----------

# Helper for computing the intersection over union of two bounding boxes
def intersection_over_union_2(box_1, box_2):
    # We retrieve the row from the dataframe and extract the information
    x1 = box_1[2] # x coordinate of top left corner
    y1 = box_1[3] # y coordinate of top left corner
    width1 = box_1[4]
    height1 = box_1[5]

    # We repeat the above steps for the row of the next image.
    x2 = box_2[2] # x coordinate of top left corner
    y2 = box_2[3] # y coordinate of top left corner
    width2 = box_2[4]
    height2 = box_2[5]

    # This formula waas adapted from: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    xA = max(x1, x2)
    yA = max(y1, y2)
    xB = min(x1+width1, x2+width2)
    yB = min(y1+height1, y2+height2)
    intersect_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    box_1_area = width1 * height1
    box_2_area = width2 * height2

    iou = intersect_area / float(box_1_area + box_2_area - intersect_area)
    return iou

# COMMAND ----------

# MAGIC %md
# MAGIC We can then perform our calculations. We match boxes to their ground truth versions using the above IOU function, if a match is 0.4 or better, the two are matched. If no match can be for our predicted box, that means there's a false positive. If no match can be found for our ground truth box, that means there's a false negative. We use the above IDs to calculate ID swapping. This allows us to compute our MOTA.

# COMMAND ----------

# Read the new GT file. Uses the same process as above.
df_new = pd.read_csv('/dbfs/FileStore/shared_uploads/sean.stilwell@ssc-spc.gc.ca/Ground_truth_with_tracking_cleaned.txt', header=None)
df_new = df_new[(df_new[7] == 3) & (df_new[0] >= IMAGE_START) & (df_new[0] <= IMAGE_END)].copy() # Constrains to only vehicles and only the desired images.
df_new = df_new.reset_index(drop=True)

# For outputting into a CSV file (for viewing)
images = []
fp_list = []
fn_list = []
ids_list = []
mota_list = []

for x in range(IMAGE_START, IMAGE_END):
    # We then create variables to hold the values for MOTA calculation.
    FPs = 0
    FNs = 0
    IDs = 0

    # Retrieve the current image and the next image from the old dataframe.
    old_image_current = df[df[0] == x + 1]
    old_image_next = df[df[0] == x + 2]

    # Repeat for the new dataframe
    new_image_current = df_new[df_new[0] == x + 1]
    new_image_next = df_new[df_new[0] == x + 2]

    # We first iterate through the given trajectories to find any IDs or FNs.
    for y in range(new_image_current.shape[0]):
        old_match = pd.Series()

        # We need to match the box from the new dataframe to the NN prediction's box.
        for z in range(old_image_current.shape[0]):
            matching = intersection_over_union_2(old_image_current.iloc[z], new_image_current.iloc[y])
            if matching > 0.4:
                old_match = old_image_current.iloc[z]
                break
        
        if not old_match.empty:
            old_next = old_image_next[(old_match['id'].item() == old_image_next['id'])]
        
            # Find the image that should appear in the next image (if it exists)
            next_match = new_image_next[(new_image_next[1] == new_image_current.iloc[y][1])]
            if not next_match.empty and old_next.empty: # Shows that the next image should have a matching box, but our algorithm does not reflect it. IDS is incremented.
                IDs += 1
        else: # Indicates the image could not be matched to our algorithm.
            FNs += 1

    # We can then iterate through our old dataframe to find any FPs.
    for y in range(old_image_current.shape[0]):
        new_match = pd.Series()

        # We need to match the box from the new dataframe to the NN prediction's box.
        for z in range(new_image_current.shape[0]):
            matching = intersection_over_union_2(new_image_current.iloc[z], old_image_current.iloc[y])
            if matching > 0.4:
                new_match = new_image_current.iloc[z]
                break
        
        if new_match.empty: # Indicates the image could not be matched to our algorithm.
            FPs += 1

    # Compute the MOTA
    if new_image_current.shape[0] > 0:
        mota = (1 - (FPs + FNs + IDs) / new_image_current.shape[0])
    else:
        mota = 1

    # We then append the results to the lists.
    images.append(x+1)
    fp_list.append(FPs)
    fn_list.append(FNs)
    ids_list.append(IDs)
    mota_list.append(mota)

# We then output the results to a CSV file.
d = {'Image': images, 'FPs': fp_list, 'FNs': fn_list, 'IDs': ids_list, 'MOTA': mota_list}
df_mota = pd.DataFrame(data=d)
df_mota.to_csv('/dbfs/FileStore/shared_uploads/sean.stilwell@ssc-spc.gc.ca/vehicle_res.csv', index=False)

# COMMAND ----------

# MAGIC %md
# MAGIC We then output the averages of the evaluation for a quick overview. The complete information is output to a .csv file.

# COMMAND ----------

print("Average False Positives:", df_mota['FPs'].mean())
print("Average False Negatives:", df_mota['FNs'].mean())
print("Average False ID swaps", df_mota['IDs'].mean())
print("Average MOTA:", df_mota['MOTA'].mean())
df_mota.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC Lastly, we compute the overall MOTA for all images.

# COMMAND ----------

numerator = df_mota['FPs'].sum() + df_mota['FNs'].sum() + df_mota['IDs'].sum()
denominator = df_new.shape[0]
print("Total False Positives:", df_mota['FPs'].sum())
print("Total False Negatives:", df_mota['FNs'].sum())
print("Total False ID swaps", df_mota['IDs'].sum())
print("Total MOTA:", 1 - numerator/denominator)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Output

# COMMAND ----------

from dbruntime.patches import cv2_imshow
img = cv2.imread("/home/000447.jpg", cv2.IMREAD_ANYCOLOR)
cv2_imshow(img)

# COMMAND ----------

from IPython.display import Video
Video('/home/test_vehicles.mp4', embed=True)

from ipywidgets import Video
video = Video.from_file("/home/test_vehicles.mp4", width=320, height=320, play=True)
video
