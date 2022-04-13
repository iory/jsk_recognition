#!/usr/bin/env python

import copy
import sys
from collections import OrderedDict
import os.path as osp

from torchvision.transforms import ToTensor, Normalize
import cv2
import numpy as np
import yaml

from image_geometry import PinholeCameraModel
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError
import message_filters
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from jsk_topic_tools import ConnectionBasedTransport

import torch

from tasks.segmentation_2d.models import build_model  # noqa


def normalize_depth(depth, min_val=None, max_val=None):
    """Normalize depth image
    Args:
        depth (np.ndarray, torch.Tensor)
        min_val (float, optional)
        max_val (float, optional)
    Returns:
        depth (np.ndarray, torch.Tensor)
    """
    if isinstance(depth, np.ndarray):
        min_val = np.nanmin(depth) if min_val is None else min_val
        max_val = np.nanmax(depth) if max_val is None else max_val
        depth = remove_nan(depth)
        depth = (depth - min_val) / (max_val - min_val)
        depth = np.clip(depth, 0, 1)
    else:
        min_val = depth.min().item() if min_val is None else min_val
        max_val = depth.max().item() if max_val is None else max_val
        depth = (depth - min_val) / (max_val - min_val)

    return depth


def remove_nan(arr):
    """Remove nan value
    Args:
        arr (np.ndarray)
    Returns:
        arr (np.ndarray)
    """
    nan_mask = np.isnan(arr)
    arr[nan_mask] = 0

    return arr

def load_yaml(filename, mode="r", strict=False, as_orderdict=False):
    """Load yaml file

    Args:
        filename (str)
        mode (str, optional):
        strict (bool, optional)
        as_orderdict (bool, optional)

    Returns:
        data (Box[str, any]): dict data allowed dot access
    """
    if not osp.exists(filename):
        if strict:
            raise FileNotFoundError("no such file `{}`".format(filename))
        else:
            print("no such file `{}`".format(filename))
            return None

    if as_orderdict:
        # load as OrderedDict
        class OrderedLoader(yaml.SafeLoader):
            pass
        OrderedLoader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            lambda loader, node: OrderedDict(loader.construct_pairs(node)),
        )
        Loader = OrderedLoader
    else:
        Loader = yaml.SafeLoader

    with open(filename, mode) as f:
        data = yaml.load(f, Loader=Loader)

    return data



class ImagePrep(object):
    """A class transform np.ndarry image to torch.Tensor and normalize"""

    def __init__(self, min_depth=920, max_depth=1200):
        self.min_depth = min_depth
        self.max_depth = max_depth

        self.to_tensor = ToTensor()
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])
        self.normalizer = Normalize(mean=self.mean, std=self.std)

    def __call__(self, rgb=None, depth=None):
        """
        Args:
            rgb (np.ndarray): (H, W, C)
            depth (np.ndarray): (H, W)
        Returns:
            img (torch.Tensor): (1, C, H, W)
        """
        if rgb is None and depth is None:
            raise ValueError("rgb or depth must be specified")

        if (rgb is not None) and (depth is None):
            img_in = self._normalize_rgb(rgb)
        elif rgb is None and depth is not None:
            img_in = self._normalize_depth(depth)
        else:
            rgb_in = self._normalize_rgb(rgb)
            depth_in = self._normalize_depth(depth)
            img_in = torch.cat((rgb_in, depth_in), dim=0)

        # (C, H, W) -> (1, C, H, W)
        img_in = img_in.unsqueeze(0)
        return img_in

    def _normalize_depth(self, depth):
        """Normalize depth
        Args:
            depth (np.ndarray): (H, W)
        Returns:
            normalized_depth (torch.Tensor): (1, H, W)
        """
        n_depth = depth.copy()
        max_val = min(self.max_depth, n_depth.max())
        min_val = max(self.min_depth, n_depth.min())
        n_depth = normalize_depth(
            n_depth, min_val=min_val, max_val=max_val)
        n_depth = self.to_tensor(n_depth)
        return n_depth

    def _normalize_rgb(self, rgb):
        """Normalize RGB value for tensor image
        Args:
            img (np.ndarray): (H, W, C)
        Returns:
            img (torch.Tensor): (C, H, W)
        """
        n_rgb = rgb.copy()
        n_rgb = self.to_tensor(n_rgb)
        n_rgb = self.normalizer(n_rgb)

        return n_rgb

    def unnormalize_rgb(self, img):
        """Revesing normalize RGB value for tensor image
        Args:
            img (torch.Tensor): (1, C, H, W)
        """
        device = img.device
        mean = self.mean.to(device)
        std = self.std.to(device)
        img[:, :3, :, :] = 255.0 * \
            (img[:, :3, :, :] * std[:, None, None] + mean[:, None, None])
        return img


COLORMAP = (
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (255, 255, 0),
    (0, 255, 255),
    (0, 165, 255),
    (255, 0, 255),
    (255, 255, 255),
    (0, 0, 0),
    )


def _transpose(img, dim_ord, copy=True):
    """Transpose image
    Args:
        img (np.ndarray): input image
        dim_ord (str): dimension order of input image
        copy (bool, optional): indicates whether return copied object
    """
    if dim_ord == "chw":
        img = img.transpose(1, 2, 0)
    elif dim_ord == "hwc":
        pass
    else:
        raise ValueError(
            "expected dim_ord=[`chw`, `hwc`], but got {}".format(dim_ord))

    if copy:
        return img.copy()
    return img


def imread(img, dim_ord="chw"):
    """Read image

    Args:
         img (str, np.ndarray, torch.Tensor): image path or tensor
         dim_ord (str): order of dimension ["chw", "hwc"]
         is_normalized (bool, optional): indicates whether input image is normalized

    Returns:
         img (np.ndarray)
    """
    if isinstance(img, str):
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(img, torch.Tensor):
        assert len(img.shape) == 3, "expected CxHxW or HxWxC shape"
        img = img.cpu().detach().numpy()
    elif isinstance(img, np.ndarray):
        assert len(img.shape) == 3, "expected CxHxW or HxWxC shape"
        pass
    else:
        raise TypeError("unexpected type of img: {}".format(type(img)))

    img = _transpose(img, dim_ord=dim_ord)
    return img


def imshow_segmentation(
    img,
    segmaps,
    img_ord="chw",
    score_thre=0.5,
    target_idx=None,
    alpha=0.3,
    ignore_idx=-1
):
    """Draw segmentaion mask

    Args:
        img (str, np.ndarray, torch.Tensor): image path or tensor, in shape CxHxW or HxWxC
        segmaps (np.ndarray, torch.Tensor): predicted or GT segmentaion maps, in shape NxHxW
        img_ord (str, optional): order of image shape(default: 'chw')
        score_thre (float, optional): segmentation score threshold(default: 0.5)
        target_idx (int, optional): if specified, only visualize output specified index channels(default: None)
        alpha (float, optional): alpha weight value(default: 0.2)
        ignore_idx (int, optional): index to ignore

    Returns:
        img (np.ndarray)
    """
    img = imread(img, img_ord)

    assert len(segmaps.shape) == 3, "expected `segmaps` is in shape NxHxW"
    if isinstance(segmaps, torch.Tensor):
        segmaps = segmaps.cpu().detach().numpy()

    N, H, W = segmaps.shape
    mask = np.zeros_like(img)
    mask_idx = np.argmax(segmaps, axis=0)

    if ignore_idx == -1:
        ignore_idx = N - 1

    if target_idx is not None:
        assert target_idx <= N - \
            1, "{} is out of range {}".format(target_idx, N - 1)
        mask[mask_idx == target_idx] = COLORMAP[target_idx % 8]
    else:
        for n in range(N):
            if n == ignore_idx:
                continue
            mask[mask_idx == n] = COLORMAP[n % 8]

    img = cv2.addWeighted(img, 1, mask, alpha, 0)

    return img



def to_mask(src, idx=0, th=0.8):
    """Convert Bx2xHxW tensor to Bx1xHxW mask
    Args:
        src (torch.Tensor): in shape (B, 2, H, W)
        idx (int, optional): target index to be 1
        th (float, optional): threshold
    Returns:
        mask (torch.Tensor): in shape (B, 1, H, W)
    """
    B, C, H, W = src.shape
    assert C == 2, "expected 2 channels tensor, but got {}".format(C)
    device = src.device
    # assert max(1.0, th) == 1 and min(0.0, th) == 0
    mask = torch.zeros(B, 1, H, W).to(device)
    mask_idx = torch.argmax(src, dim=1, keepdim=True)
    mask[mask_idx == idx] = 1
    return mask


def load_model(model, filename, strict=True, eval=False, freeze=False):
    """Load pre-trained torch.nn.Module
    Args:
        model (torch.nn.Module)
        filename (str)
        strict (bool, optional)
        eval (bool, optional)
        freeze (bool, optional)
    Returns:
        model
    """
    device = next(model.parameters()).device
    state_dict = torch.load(filename, map_location=device)
    if "model" in state_dict.keys():
        model.load_state_dict(state_dict["model"], strict=strict)
    else:
        print(
            "type of state_dict is deprecated, make it contain keys `model` and `optimizer`.")
        model.load_state_dict(state_dict, strict=strict)

    if eval:
        model.eval()
    else:
        model.train()

    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        print("{} is freezed".format(model.name))

    return model



def find_contours(mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE):
    """wrapper of cv2.findContours(), which doesn't depend on opencv version
    Args:
        mask (np.ndarray): in shape HxW
        mode (long): type of mode (default: cv2.RETR_TREE)
        method (long): type of method (default: cv2.CHAIN_APPROX_NONE)
    Returns:
        cnts (list[np.ndarray]): found contours
    """
    if (cv2.__version__)[:3] == "3.4":
        _, cnts, _ = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    else:
        cnts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    return cnts




def get_max_area(mask):
    """Extract max contour area from mask
    Args:
        mask (np.ndarray)
    Returns:
        out (np.ndarray)
    """
    cnts = find_contours(mask)
    cnts.sort(key=lambda x: cv2.contourArea(x), reverse=True)
    out = np.zeros_like(mask)
    cv2.drawContours(out, [cnts[0]], -1, color=255, thickness=-1)
    return out



def mask2roi(mask, max_area=True, offset_factor=(0.1, 0.01)):
    """Convert mask to roi
    Args:
        mask (np.ndarray): mask image
        max_area (bool): whether extract max contour area
        offset_factor (sequence): roi offset factor
    Returns:
        roi (tuple[float]): (top, left, bottom, right) order
    """
    if not isinstance(offset_factor, (list, tuple)):
        offset_factor = (offset_factor, offset_factor)

    if mask.max() <= 1:
        mask *= 255

    if max_area:
        mask = get_max_area(mask)

    h, w = mask.shape[:2]
    off_h = int(h * offset_factor[1])
    off_w = int(w * offset_factor[0])

    foreground = np.where(mask == 255)
    if len(foreground[0]) > 0:
        top = max(np.min(foreground[0]) - off_h, 0)
        bottom = min(np.max(foreground[0]) + 1 + off_h, h)
        left = max(np.min(foreground[1]) - off_w, 0)
        right = min(np.max(foreground[1]) + 1 + off_w, w)
    else:
        top = 0
        bottom = h
        left = 0
        right = w

    return top, left, bottom, right


def imresize(img, size=256, copy=False, interpolation=cv2.INTER_NEAREST):
    """Resize image
    Args:
        img (np.ndarray): source image
        size (int, sequence): target size, if sequence (width, height) order(default: 256)
        copy (bool, optional): indicates whether copy image array(default: False)
        interpolation ():
    Returns:
        np.ndarray: resized image
    """
    if isinstance(size, int):
        size = (size, size)
    elif isinstance(size, (list, tuple)):
        pass
    else:
        raise TypeError("unexpected type of size: {}".format(type(size)))

    img = cv2.resize(img, size, interpolation=interpolation)

    return img


class MaskDetector(ConnectionBasedTransport):

    def __init__(self):
        super(ConnectionBasedTransport, self).__init__()
        # ROS params
        config_path = rospy.get_param("~config_path", None)
        if config_path is not None:
            if not osp.exists(config_path):
                raise IOError("No such file: {}".format(config_path))
            self.config = load_yaml(config_path)
            self.algorithm = self.config["algorithm"]
            self.weight_file = rospy.get_param("~weight_file")
            self.use_dims = rospy.get_param("~use_dims", 3)
            image_size = rospy.get_param("~image_size", 256)
            if isinstance(image_size, int):
                self.image_size = (image_size, image_size)
            elif isinstance(image_size, (list, tuple)):
                self.image_size = image_size
            else:
                raise TypeError(
                    "unexpected type of image_size: {}".format(type(image_size)))

            # Prepare input image for model
            # max : min = 1200 : 940
            max_depth = rospy.get_param("~max_depth", 1200)
            min_depth = rospy.get_param("~min_depth", 920)
            self.prep = ImagePrep(min_depth=min_depth,
                                  max_depth=max_depth)

        # topic being often used
        self.rgb_topic = rospy.get_param(
            "~rgb", "/head_mount_kinect/rgb/image_raw")
        self.depth_topic = rospy.get_param(
            "~depth", "/head_mount_kinect/depth/image_raw")
        self.caminfo_topic = rospy.get_param(
            "~camera_info", "/head_mount_kinect/rgb/camera_info")

        # utils
        self.cv_bridge = CvBridge()
        self.camera_model = PinholeCameraModel()

        super(MaskDetector, self).__init__()
        x_offset_fac = rospy.get_param("~x_offset_factor", 0.1)
        y_offset_fac = rospy.get_param("~y_offset_factor", 0.01)
        self.offset_factor = (x_offset_fac, y_offset_fac)

        self.use_rectified_image = rospy.get_param(
            "~use_rectified_image", True)

        # Load pre-trained model
        self.model = build_model(self.algorithm)
        self.model = load_model(
            self.model, self.weight_file, strict=True, eval=True)
        rospy.loginfo("Model loaded with {}".format(self.weight_file))

        # Subscriber
        sub_rgb = message_filters.Subscriber(self.rgb_topic, Image)
        sub_depth = message_filters.Subscriber(self.depth_topic, Image)
        sub_caminfo = message_filters.Subscriber(
            self.caminfo_topic, CameraInfo)
        ts = message_filters.ApproximateTimeSynchronizer(
            [sub_rgb, sub_depth, sub_caminfo], 100, 10.0)
        ts.registerCallback(self.callback)

        # Publisher
        self.pub_pred = rospy.Publisher(
            "~output/predict/image", Image, queue_size=100)
        self.pub_mask = rospy.Publisher(
            "~output/predict/mask", Image, queue_size=100)
        self.pub_roi_mask = rospy.Publisher(
            "~output/apply_mask/mask", Image, queue_size=100)
        self.pub_roi_img = rospy.Publisher(
            "~output/apply_mask/image", Image, queue_size=100)
        self.pub_roi_depth = rospy.Publisher(
            "~output/apply_mask/depth", Image, queue_size=100)
        self.pub_roi_caminfo = rospy.Publisher(
            "~output/apply_mask/camera_info", CameraInfo, queue_size=100)

        rospy.loginfo("{}: Node initialize is done".format(
            self.__class__.__name__))

    def callback(self, rgb_msg, depth_msg, info_msg):
        """Callback
        Args:
            rgb_msg (sensor_msgs.msg.Image)
            depth_msg (sensor_msgs.msg.Image)
            info_msg (sensor_msgs.msg.CameraInfo)
        """
        try:
            rgb = self.cv_bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, "32FC1")
        except CvBridgeError as e:
            rospy.logerr(e)
            return
        h, w = rgb.shape[:2]

        # Model prediction
        img_in = self._prepare_inputs(rgb=rgb.copy(), depth=depth.copy())
        seg_map = self.model(img_in)

        mask = to_mask(seg_map, idx=0).squeeze(0).cpu().detach().numpy()
        mask = mask.transpose(1, 2, 0) * 255
        mask = mask.astype("uint8")
        mask = imresize(mask, (w, h))
        mask_msg = self.cv_bridge.cv2_to_imgmsg(mask, encoding="mono8")
        mask_msg.header = rgb_msg.header
        mask_msg.header.stamp = rospy.Time.now()

        try:
            roi_img_msg, roi_depth_msg, roi_mask_msg, camera_info = \
                self.apply_mask(mask, rgb, depth, info_msg, rgb_msg)
            roi_img_msg.header = rgb_msg.header
            roi_img_msg.header.stamp = mask_msg.header.stamp
            roi_depth_msg.header = depth_msg.header
            roi_depth_msg.header.stamp = mask_msg.header.stamp
            roi_mask_msg.header = rgb_msg.header
            roi_mask_msg.header.stamp = mask_msg.header.stamp
            camera_info.header.stamp = mask_msg.header.stamp
        except Exception as e:
            rospy.logerr(e)
            return

        # Visualize
        img_vis = imresize(rgb.copy(), size=self.image_size)
        ret_img = imshow_segmentation(img_vis, seg_map[0], img_ord="hwc")
        ret_img = ret_img.astype("uint8")
        ret_img = imresize(ret_img, (w, h))
        img_msg = self.cv_bridge.cv2_to_imgmsg(ret_img, encoding="bgr8")
        img_msg.header = rgb_msg.header

        # Publish predicted image
        self.publish(img_msg, mask_msg, roi_img_msg,
                     roi_depth_msg, roi_mask_msg, camera_info)

    def apply_mask(self, mask, rgb, depth, info_msg, rgb_msg):
        """Crop image and depth
        Args:
            mask (np.ndarray): mask
            rgb (np.ndarray): rgb image
            depth (np.ndarray): depth image
            info_msg (sensor_msgs.msg.CameraInfo): info_msg
        Returns:
            roi_img_msg (sensor_msgs.msg.Image)
            roi_depth_msg (sensor_msgs.msg.Image)
            roi_mask_msg (sensor_msgs.msg.Image)
            camera_info (sensor_msgs.msg.CameraInfo)
        """
        roi = mask2roi(mask, max_area=True, offset_factor=self.offset_factor)
        roi_rgb = rgb[roi[0]:roi[2], roi[1]:roi[3]]
        roi_depth = depth[roi[0]:roi[2], roi[1]:roi[3]]
        roi_mask = mask[roi[0]:roi[2], roi[1]:roi[3]]

        roi_img_msg = self.cv_bridge.cv2_to_imgmsg(roi_rgb, "bgr8")
        roi_depth_msg = self.cv_bridge.cv2_to_imgmsg(roi_depth, "32FC1")
        roi_mask_msg = self.cv_bridge.cv2_to_imgmsg(roi_mask, "mono8")

        camera_info = copy.deepcopy(info_msg)
        # camera_info.roi.x_offset = (roi[2] - roi[0]) // 2
        # camera_info.roi.y_offset = (roi[3] - roi[1]) // 2
        # camera_info.roi.width = roi[2] - roi[0]
        # camera_info.roi.height = roi[3] - roi[1]
        camera_info.roi.x_offset = 0
        camera_info.roi.y_offset = 0
        # camera_info.roi.width = roi[3] - roi[1]
        # camera_info.roi.height = roi[2] - roi[0]
        camera_info.roi.width = roi[3]
        camera_info.roi.height = roi[2]
        camera_info.roi.do_rectify = self.use_rectified_image

        return roi_img_msg, roi_depth_msg, roi_mask_msg, camera_info

    def publish(self,
                img_msg,
                mask_msg,
                roi_img_msg,
                roi_depth_msg,
                roi_mask_msg,
                camera_info):
        """Publish predicted image
        Args:
            img_msg (sensor_msgs.msg.Image)
            mask_msg (sensor_msgs.msg.Image)
            roi_img_msg (sensor_msgs.msg.Image)
            roi_depth_msg (sensor_msgs.msg.Image)
            roi_mask_msg (sensor_msgs.msg.Image)
            camera_info (sensor_msgs.msg.CameraInfo)
        """
        self.pub_pred.publish(img_msg)
        self.pub_mask.publish(mask_msg)
        self.pub_roi_img.publish(roi_img_msg)
        self.pub_roi_depth.publish(roi_depth_msg)
        self.pub_roi_mask.publish(roi_mask_msg)
        self.pub_roi_caminfo.publish(camera_info)
        rospy.logdebug("{}: Publish".format(self.__class__.__name__))

    def _prepare_inputs(self, rgb=None, depth=None):
        """
        Args:
            rgb (np.ndarray): (H, W, C)
            depth (np.ndarray): (H, W)
        Returns:
            img_in (torch.Tensor): (1, N, H, W)
        """
        if self.use_dims == 4:
            rgb_in = imresize(rgb.copy(), size=self.image_size)
            depth_in = imresize(depth.copy(), size=self.image_size)
            img_in = self.prep(rgb=rgb_in, depth=depth_in)
        elif self.use_dims == 1:
            depth_in = imresize(depth.copy(), size=self.image_size)
            img_in = self.prep(depth=depth_in)
        else:
            rgb_in = imresize(rgb.copy(), size=self.image_size)
            img_in = self.prep(rgb=rgb_in)

        return img_in


if __name__ == "__main__":
    rospy.init_node("mask_detector")
    MaskDetector()
    rospy.spin()
