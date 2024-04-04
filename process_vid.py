from collections import OrderedDict
import json
import os
import cv2
import sys
import multiprocessing

def crop_faces_from_video(in_videofile, landmarks_path, crop_faces_out_dir, overwrite=False, frames_num=10,
                          buf=0.10, clean_up=True):
    id = os.path.splitext(os.path.basename(in_videofile))[0]
    json_file = os.path.join(landmarks_path, id + '.json')
    out_dir = os.path.join(crop_faces_out_dir, id)
    if not os.path.isfile(json_file):
        return
    if not overwrite and os.path.isdir(out_dir):
        return

    try:
        with open(json_file, 'r') as jf:
            face_box_dict = json.load(jf)
    except Exception as e:
        print(f'failed to parse {json_file}')
        print(e)
        raise e

    os.makedirs(out_dir, exist_ok=True)
    capture = cv2.VideoCapture(in_videofile)
    # frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(frames_num):
        capture.grab()
        # if i % frame_hops != 0:
        #     continue
        success, frame = capture.retrieve()
        if not success or str(i) not in face_box_dict:
            continue
        #

        crops = []
        bboxes = face_box_dict[str(i)][0]
        if bboxes is None:
            continue
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = [int(b) for b in bbox]
            w = xmax - xmin
            h = ymax - ymin
            p_h = int(h * buf)
            p_w = int(w * buf)
            crop = frame[max(ymin - p_h, 0):ymax + p_h, max(xmin - p_w, 0):xmax + p_w]
            crops.append(crop)

        for j, crop in enumerate(crops):
            cv2.imwrite(os.path.join(out_dir, "{}_{}.png".format(i, j)), crop)

def crop_faces_from_video_batch(input_filepath_list, crop_faces_out_dir):
    os.makedirs(crop_faces_out_dir, exist_ok=True)
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        jobs = []
        results = []
        for input_filepath in input_filepath_list:
            jobs.append(pool.apply_async(crop_faces_from_video,
                                         (input_filepath, crop_faces_out_dir,),
                                         )
                        )

        for job in tqdm(jobs, desc="Cropping faces"):
            results.append(job.get())


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Arguments missing.")
        # print("Usage: main.py [train_data_path] [test_data_path]")

    input_filepath_list = sys.argv[1]
    crops_path = sys.argv[2]

    crop_faces_from_video_batch(input_filepath_list, crops_path)