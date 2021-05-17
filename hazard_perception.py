# Initial Config
from imageai.Detection import ObjectDetection
import os
import cv2
import numpy as np
import glob
from IPython.display import clear_output

# Object detection
def object_detection(model_path: str, input_image_path: str, output_image_path: str):
    """
    Detects objects within the image and makes a new image highlighting them
    with a probability attached.

    Parameters
    ----------
    model_path: str, the path of the model to be used
    input_image_path: str, the path of the image to be analysed
    output_image_path: str, the path of the new image to be created

    Returns
    -------
    A new image with object detection analysis completed.

    """
    # Perform the pre-requisite steps for the analysis
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()

    # Set the model path and load the model
    detector.setModelPath(model_path)
    detector.loadModel()

    # Perform the object detection analysis
    detections, objects_path = detector.detectObjectsFromImage(
        input_image=input_image_path,
        output_image_path=output_image_path,
        minimum_percentage_probability=30,
        extract_detected_objects=True,
    )

    for eachObject, eachObjectPath in zip(detections, objects_path):
        print(
            eachObject["name"],
            " : ",
            eachObject["percentage_probability"],
            " : ",
            eachObject["box_points"],
        )
        print("Object's image saved in " + eachObjectPath)
        print("--------------------------------")


# Split video into images
def split_video_into_images(video_file_location: str,
                            output_image_file_location: str,
                            skip_frames: int=None):
    """
    Takes in a video file and splits into images.

    Parameters
    ----------
    video_file_location: str, the location of the video file to be split
    output_image_file_location: str, the output location of the image files
    skip_frames: int, an additional parameter to skip the frames to make processing faster

    Returns
    -------
    A series of images (jpeg format) outputted into a data folder
    """
    # Load in the video file
    cam = cv2.VideoCapture(video_file_location)

    # Extract the total frames
    total_frames = cam.get(cv2.CAP_PROP_FRAME_COUNT)

    # frame
    currentframe = 0

    # Handle the skip_frames param
    frames = []
    if skip_frames:
        frames = list(range(0, int(total_frames), skip_frames))
    else:
        frames = frames

    while True:
        # reading from frame
        ret, frame = cam.read()

        if ret:
            if skip_frames:
                if currentframe in frames:
                    # If the skip_frames param is passed off, then only process frames within the frames list
                    # if video is still left continue creating images
                    name = output_image_file_location + str(currentframe) + ".jpg"
                    print("\r", "Creating..." + name, end="")

                    # writing the extracted images
                    cv2.imwrite(name, frame)
            else:
                # If the skip_frames param is not passed, then process all the frames
                name = output_image_file_location + str(currentframe) + ".jpg"
                print("\r", "Creating..." + name, end="")

                # writing the extracted images
                cv2.imwrite(name, frame)

            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1
        else:
            break

    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()


# Combine images back into video file
def combine_images_to_video(images_filepath: str, video_output_path: str, video_name: str):
    """
    Takes images from the filepath input and turns it into a video file (mp4 format).

    Parameters
    ----------
    images_filepath: str, the filepath of the images to be combined
    video_output_path: str, the output of the video file
    video_name: str, the name of the video to be named

    Returns
    -------
    A video file (mp4 format) of the combined images as a video file.

    """
    # Making an image empty array
    img_array = []

    # Look for all jpg files in the filepath param provided
    # Sort by the modification time
    for filename in sorted(glob.glob(f"{images_filepath}*.jpg"), key=os.path.getmtime):
        # Read in the image
        img = cv2.imread(filename)

        # Extract the height, width and layers
        height, width, layers = img.shape

        # Create the size
        size = (width, height)

        # Append to the img_array list
        img_array.append(img)

    # Create the video from the img_array
    out = cv2.VideoWriter(
        video_output_path + video_name, cv2.VideoWriter_fourcc(*"DIVX"), 15, size
    )

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


# Make new filepaths from the input images
def make_new_filepath(input_image_path: str) -> str:
    """
    Takes the input image filepaths and turns them into output filepaths.

    If the input file path is: '/images/0.jpg', then the output is 'images/0_new.jpg'.

    Parameters
    ----------
    input_image_path: str, the filepath of the input image

    Returns
    -------
    A str of the filepath of the output image

    """
    # Make a list
    input_image_list = input_image_path.split("/")

    # Extract the filename - it will always be the last element in the list
    old_name = input_image_list[-1]

    # Remove the old filename
    input_image_list = input_image_list[:-1]

    # Make a new filename
    new_name = 'analysed/' + old_name.split(".")[0] + "_new.jpg"

    # Create an output image list
    output_image_list = input_image_list + [new_name]

    # Return the new filepath as a string
    return "/".join(output_image_list)


def make_new_filepaths(input_images_paths: list) -> list:
    """
    Takes a list of the input images filepaths and returns their equivalent output filepaths.

    Parameters
    ----------
    input_images_paths: a list, of the input images filepaths

    Returns
    -------
    A list of the filepaths of the output images

    """
    output_filepaths = []

    for input_image_path in input_images_paths:
        output_filepath = make_new_filepath(input_image_path)
        output_filepaths.append(output_filepath)

    return output_filepaths

def main():
    """
    The execution of the main functions based on the execution of the hazard perception functions.

    Returns
    -------
    An analysed video clip with the hazards being identified accordingly.

    """
    # Split video into images
    video_file_location = '/Users/aniruddha.sengupta/Desktop/Hazard Perception/videos/video.mp4'
    output_image_file_location = '/Users/aniruddha.sengupta/Desktop/Hazard Perception/images/'

    split_video_into_images(video_file_location=video_file_location,
                            output_image_file_location=output_image_file_location)

    # Object detection
    model_path = '/Users/aniruddha.sengupta/Desktop/Hazard Perception/weights/yolo.h5'

    # Make a list of input image paths
    input_images_paths = []
    for filename in sorted(glob.glob(f'{output_image_file_location}*.jpg'), key=os.path.getmtime):
        input_images_paths.append(filename)

    # Create the output filepaths
    output_images_paths = make_new_filepaths(input_images_paths)

    for input_image_path, output_image_path in zip(input_images_paths, output_images_paths):
        object_detection(model_path=model_path,
                         input_image_path=input_image_path,
                         output_image_path=output_image_path)
        clear_output(wait=True)

    # Combine images back to video
    analysed_image_file_location = '/Users/aniruddha.sengupta/Desktop/Hazard Perception/images/analysed/'

    analysed_image_paths = []
    for filename in sorted(glob.glob(f'{analysed_image_file_location}*.jpg'), key=os.path.getmtime):
        analysed_image_paths.append(filename)

    video_output_path = '/Users/aniruddha.sengupta/Desktop/Hazard Perception/videos/'

    combine_images_to_video(images_filepath=analysed_image_file_location,
                            video_output_path=video_output_path,
                            video_name='project.mp4')


# Execution
if __name__ == "__main__":
    main()

