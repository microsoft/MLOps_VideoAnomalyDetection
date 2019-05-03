import cv2 #.cv2 import VideoCapture, imwrite
import argparse
import os
import glob
from PIL import Image

# define input arguments that this script accepts
parser = argparse.ArgumentParser(description='Process input arguments')
parser.add_argument('--input_data', default='./data/video/UCSDped1', type=str, dest='input_data', help='data folder mounting point')
parser.add_argument('--output_data', default='./data/raw/UCSDped1', type=str, dest='output_data', help='data folder mounting point')

# process input arguments
args = parser.parse_args()
input_data = args.input_data
output_data = args.output_data

print("Input folder: %s" % input_data)
print("Output folder: %s" % output_data)

recordings = sorted(glob.glob(os.path.join(input_data, 'Train', 'Train*[0-9]')))
n_recordings = len(recordings)

print("Found %s images for training" % n_recordings)
print("Folders: "),
print(os.listdir(input_data))
print(os.listdir(os.path.join(input_data, 'Train')))

train_recordings = list(zip([input_data] * n_recordings, recordings))

# Recordings used for testing
# recordings_parent_folder = os.path.join('data', folders[0])
recordings = glob.glob(os.path.join(input_data, 'Test', 'Test*[0-9]'))
n_recordings = len(recordings)
test_recordings = sorted(list(zip([input_data] * n_recordings, recordings)))

for split in [train_recordings, test_recordings]:
    for _, folder in split:
        out_folder = os.path.join(output_data, folder[folder.find("Train"):])
        os.makedirs(out_folder, exist_ok=True)
        # print(out_folder)

        video_file = os.path.join(folder, 'video.mp4')
        print("Processing video: %s" % video_file)
        vidcap = cv2.VideoCapture(video_file)
        
        count = 0
        read_success, image = vidcap.read()
        # with open("filename.txt", 'w') as f:
        #     f.write("someting\n")
        while read_success:
            count += 1
            filename = os.path.join(out_folder, "%03d.tif" % count)
            # img = Image.fromarray(image)
            # img.save( filename )
            write_success = cv2.imwrite(filename, image)     # save frame as JPEG file   
            if not write_success:
                print('Error writting image ', filename)
            # print("imwrite returned: %s" % ret)   
            read_success, image = vidcap.read()
        else:
            print("Processed %d frames" % count)