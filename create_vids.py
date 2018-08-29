import cv2
import os
import sys


import skvideo.io
import skvideo.measure
from skimage.io import imread

def create_video(start_frame, video_name):
    image_folder = 'mmnist'
    video_name = f'data/{video_name}.avi'

    included = [f"{str(i)}.jpg" for i in range(start_frame, start_frame+120)]
    all_imgs =  set(os.listdir(image_folder))
    img_paths = [p for p in os.listdir(image_folder) if p in included ]
    print(included)
    img_paths.sort(key = lambda x: int(x[:-4]))
    #images = [img for img in os.listdir(image_folder) if img.endswith(".png")]



    writer = skvideo.io.FFmpegWriter(video_name)

    try:
        for image in img_paths:
            im = imread(os.path.join(image_folder, image))
            writer.writeFrame(im)
    except:
        print("failed to write to file")
    else:
        print(f"Created video with name {video_name} starting on frame {start_frame}")
    finally:
        writer.close()

    return
    
    first_frame = cv2.imread(os.path.join(image_folder, img_paths[0]))
    height, width, layers = first_frame.shape

    video = cv2.VideoWriter(video_name, -1, 1, (width,height))

    for image in img_paths:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    try:
        cv2.destroyAllWindows()
        video.release()
    except:
        print("Something went wrong")
        raise
    else:
        print(f"Created video with name {video_name} starting on frame {start_frame}")


if __name__ == "__main__":
    start_img = sys.argv[1]
    video_name = sys.argv[2]
    create_video(int(start_img), video_name)