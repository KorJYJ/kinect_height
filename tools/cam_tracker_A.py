import tools.pyKinectAzure.pykinect_azure as pykinect
from tools.pyKinectAzure.pykinect_azure.k4a import _k4a

import argparse
import os
import os.path as osp
import time
import torch

from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import *
from yolox.tracking_utils.timer import Timer
import sys
import cv2
import math
import numpy as np

import copy

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

#cap = cv2.VideoCapture(url)
pykinect.initialize_libraries()
# Modify camera configuration
device_config = pykinect.default_configuration

## 카메라 화소 및 depth 마다 Camera Matrix가 다르므로 여건에 맞춰서 설정
# Modify camera configuration

device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_UNBINNED
device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_15
kinect = pykinect.start_device(config=device_config)
kinect_calibration = kinect.get_calibration(device_config.depth_mode, device_config.color_resolution)

    
# kinect = pykinect.start_playback('/home/kist/Desktop/221102_기관고유_Depth/set1_sub2.mkv')
# kinect_calibration = kinect.get_calibration()
# kinect.set_color_conversion(pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32)
# print(device_config)

def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        #"--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
        "--path", default="./videos/palace.mp4", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            #logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info


# def image_demo(predictor, vis_folder, current_time, args):
#     if osp.isdir(args.path):
#         files = get_image_list(args.path)
#     else:
#         files = [args.path]
#     files.sort()
#     tracker = BYTETracker(args, frame_rate=args.fps)
#     timer = Timer()
#     results = []

#     for frame_id, img_path in enumerate(files, 1):
#         outputs, img_info = predictor.inference(img_path, timer)
#         if outputs[0] is not None:
#             online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
#             online_tlwhs = []
#             online_ids = []
#             online_scores = []
#             for t in online_targets:
#                 tlwh = t.tlwh
#                 tid = t.track_id
#                 vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
#                 if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
#                     online_tlwhs.append(tlwh)
#                     online_ids.append(tid)
#                     online_scores.append(t.score)
#                     # save results
#                     results.append(
#                         f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
#                     )
#             timer.toc()
#             online_im = plot_tracking(
#                 img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
#             )
#         else:
#             timer.toc()
#             online_im = img_info['raw_img']

#         # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
#         if args.save_result:
#             timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
#             save_folder = osp.join(vis_folder, timestamp)
#             os.makedirs(save_folder, exist_ok=True)
#             cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)

#         if frame_id % 20 == 0:
#             logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

#         ch = cv2.waitKey(0)
#         if ch == 27 or ch == ord("q") or ch == ord("Q"):
#             break

#     if args.save_result:
#         res_file = osp.join(vis_folder, f"{timestamp}.txt")
#         with open(res_file, 'w') as f:  
#             f.writelines(results)
#         logger.info(f"save results to {res_file}")
# import matplotlib.pyplot as plt
# import pandas as pd



def measure_height_1(depth_image, tlwh):
    x1, y1, w, h = tlwh
    
    head_pixel = _k4a.k4a_float2_t()
    feet_pixel = _k4a.k4a_float2_t()
    mid_pixel = _k4a.k4a_float2_t()
    
    head_pixel.xy.x = x1 + w//2
    head_pixel.xy.y = y1+15

    feet_pixel.xy.x = x1 + w//2
    feet_pixel.xy.y = y1 + h


    Head_Depth = depth_image[int(y1)+20, int(x1+w//2)]
    if y1+h > depth_image.shape[0]:
        return 0
    Feet_Depth = depth_image[int(y1+h), int(x1 + w//2)]
    
    
    head = kinect_calibration.convert_2d_to_3d( head_pixel,  Head_Depth, pykinect.K4A_CALIBRATION_TYPE_DEPTH, pykinect.K4A_CALIBRATION_TYPE_DEPTH)
    feet = kinect_calibration.convert_2d_to_3d( feet_pixel,  Feet_Depth, pykinect.K4A_CALIBRATION_TYPE_DEPTH, pykinect.K4A_CALIBRATION_TYPE_DEPTH)
    

    height = math.sqrt(pow(head.v[0] - feet.v[0],2) + pow(head.v[1] - feet.v[1],2) + pow(head.v[2] - feet.v[2],2))
    
    
    return height
    
# def measure_height_2(depth_image, tlwh):
#     mean_depth = 0
#     x1, y1, w, h = tlwh

#     head_pixel = _k4a.k4a_float2_t()
#     feet_pixel = _k4a.k4a_float2_t()
#     mid_pixel = _k4a.k4a_float2_t()
    
#     head_pixel.xy.x = x1 + w/2
#     head_pixel.xy.y = y1

#     feet_pixel.xy.x = x1 + w/2
#     feet_pixel.xy.y = y1 + h
#     if y1+h > depth_image.shape[0]:
#         return 0

#     for i in range(-3, 3):
#         for j in range(-3, 3):
#             if 0 < int(x1 + w//2) +i < depth_image.shape[0] and 0 < int(y1+ h//2) + j < depth_image.shape[1]:
#                 mean_depth = mean_depth + depth_image[int(y1+ h//2) +i , int(x1 + w//2) + j].astype('float') / 36
        

    
    
#     head = calibration.convert_2d_to_3d( head_pixel,  mean_depth, pykinect.K4A_CALIBRATION_TYPE_DEPTH, pykinect.K4A_CALIBRATION_TYPE_DEPTH)
#     feet = calibration.convert_2d_to_3d( feet_pixel,  mean_depth, pykinect.K4A_CALIBRATION_TYPE_DEPTH, pykinect.K4A_CALIBRATION_TYPE_DEPTH)
    
#     height = abs(head.v[1] - feet.v[1])    
    
#     return height

# def measure_height_3(depth_image, tlwh):
#     x1 , y1, w, h = tlwh
#     bbox_human_depth = copy.deepcopy(depth_image[int(y1):int(y1+h), int(x1):int(x1+w)])

#     threshold1 = np.quantile(bbox_human_depth.flatten(), 0.15)
#     threshold2 = np.quantile(bbox_human_depth.flatten(), 0.80)
    
#     filter = bbox_human_depth<threshold2
#     filter2 = threshold1 < bbox_human_depth
    
#     bbox_human_result_depth = filter*filter2*bbox_human_depth
    
#     mean_y = np.mean(bbox_human_result_depth, axis=1)
#     # human height not zero
#     for yi in range(len(bbox_human_result_depth)):
#         if mean_y[yi] >0:
#             break
        
#     for xi in range(len(bbox_human_result_depth[yi])):
#         if 0 < bbox_human_result_depth[yi, xi] < 3000:
#             break
    
    
#     head_pixel = _k4a.k4a_float2_t()
#     feet_pixel = _k4a.k4a_float2_t()
    
    
#     head_pixel.xy.x = xi
#     head_pixel.xy.y = yi

#     feet_pixel.xy.x = x1 + w/2
#     feet_pixel.xy.y = y1 + h


#     Head_Depth = bbox_human_result_depth[yi, xi]
#     Feet_Depth = bbox_human_result_depth[int(x1 + w/2), int(y1+h)]
    
    
#     head = calibration.convert_2d_to_3d( head_pixel,  Head_Depth, pykinect.K4A_CALIBRATION_TYPE_DEPTH, pykinect.K4A_CALIBRATION_TYPE_DEPTH)
#     feet = calibration.convert_2d_to_3d( feet_pixel,  Feet_Depth, pykinect.K4A_CALIBRATION_TYPE_DEPTH, pykinect.K4A_CALIBRATION_TYPE_DEPTH)
    

#     height = math.sqrt(pow(head.v[0] - feet.v[0],2) + pow(head.v[1] - feet.v[1],2) + pow(head.v[2] - feet.v[2],2))

    
#     return height

def make_height_class():
    interval = np.arange(100, 201, 5)
    
    height_class = []
    
    for i in range(len(interval) - 1):
        height_class.append((interval[i],  interval[i+1]))
        
    return height_class


def human_segment(color_image, depth_image, tlwh):
    x1, y1, w, h = tlwh
    
    bbox_human_depth = copy.deepcopy(depth_image[int(y1):int(y1+h), int(x1):int(x1+w)])
    bbox_human_rgb = copy.deepcopy(color_image[int(y1):int(y1+h), int(x1):int(x1+w)])
    plt.hist(bbox_human_depth.flatten(), bins = 100)
    a = pd.DataFrame(bbox_human_depth.flatten())
    print(a.describe())
    plt.pause(0.01)
    plt.clf()

    threshold1 = np.quantile(bbox_human_depth.flatten(), 0.15)
    threshold2 = np.quantile(bbox_human_depth.flatten(), 0.80)
    
    filter = bbox_human_depth<threshold2
    filter2 = threshold1 < bbox_human_depth
    
    bbox_human_result  = filter.reshape(filter.shape[0], filter.shape[1], 1)*filter2.reshape(filter2.shape[0], filter2.shape[1], 1)*bbox_human_rgb
    bbox_human_result_depth = filter*filter2*bbox_human_depth
    cv2.imshow('a', bbox_human_result)
    
    print(bbox_human_result_depth.mean(axis=1))
    
    return bbox_human_result

    
def imageflow_demo(predictor, vis_folder, current_time, args):   
    width = 512
    height = 512
    fps = 30
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    
    # save
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = osp.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = osp.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    
    
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []

    # height class interval 5
    height_class = make_height_class()

    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        # cap = kinect.update()
        cap = kinect.update()
        ret, color_image = cap.get_transformed_color_image()
        color_image = color_image[:, :, 0:3]
        ret_depth, depth_image = cap.get_depth_image()   
        
        if ret and ret_depth:
            outputs, img_info = predictor.inference(color_image, timer)
            if outputs[0] is not None:
                
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                online_height = []
                
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id

                    t.update_height(measure_height_1(depth_image, tlwh))
                    # meausre human height
                    # measure_height_3(depth_image, tlwh)
                    # height = measure_height_2(depth_image, tlwh)
                    # online_height.append(str(round(height/10, 2)))
                    
                    height = str(round(t.get_height()/10, 2)+3)
                    online_height.append(height)
                    
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        
                        online_scores.append(t.score)

                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                        
                timer.toc()
                online_im = plot_tracking(
                    img_info['raw_img'], online_tlwhs, online_ids, online_height, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
                
                online_depth = plot_tracking(
                    depth_image.astype(np.uint8), online_tlwhs, online_ids, online_height, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
                cv2.imshow('Tracking', cv2.resize(online_im, (640, 640)))
                cv2.imshow('depth', online_depth)
            else:
                timer.toc()
                online_im = img_info['raw_img']
            if args.save_result:
                vid_writer.write(online_im)
            
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    current_time = time.localtime()
    # if args.demo == "image":
    #     image_demo(predictor, vis_folder, current_time, args)
    imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
