import os
import glob
import xml.etree.ElementTree as ET
import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import frangi
from sklearn.preprocessing import normalize


def get_annotations(ann_dir):
    ann_paths = []
    all_bboxes = []
    ann_list = glob.glob(os.path.join(ann_dir, '*.xml'))

    for ann in ann_list:
        ann_bboxes = []
        ann_xml = ET.parse(ann).getroot()
        ann_filename = ann_xml.find("filename").text

        for ann_object in ann_xml.findall('object'):
            xmin = ann_object.find('bndbox').find('xmin').text
            xmax = ann_object.find('bndbox').find('xmax').text
            ymin = ann_object.find('bndbox').find('ymin').text
            ymax = ann_object.find('bndbox').find('ymax').text

            bbox = [xmin, xmax, ymin, ymax]
            ann_bboxes.append(bbox)
        all_bboxes.append(ann_bboxes)
        ann_paths.append(ann_filename)

    return ann_paths, all_bboxes


def find_no_bbox_imgs(img_dir, ann_dir):
    img_paths = os.listdir(img_dir)
    ann_list = glob.glob(os.path.join(ann_dir, '*.xml'))

    ann_filenames = []

    for ann in ann_list:
        ann_xml = ET.parse(ann).getroot()
        ann_filename = ann_xml.find('filename').text

        ann_filenames.append(ann_filename)

    for img_path in img_paths:
        if img_path in ann_filenames:
            os.replace(os.path.join(img_dir, img_path), os.path.join('data/allianz-bbox-temp', img_path))
        else:
            os.replace(os.path.join(img_dir, img_path), os.path.join('data/allianz-data-no-bbox', img_path))

    return True


def downscale_annotations(ann_dir, ann_out, factor=4):
    ann_paths = []
    all_bboxes = []
    ann_list = glob.glob(os.path.join(ann_dir, '*.xml'))

    for ann in ann_list:
        ann_bboxes = []
        tree = ET.parse(ann)
        ann_xml = tree.getroot()

        ann_filename = ann_xml.find('filename').text

        ann_size = ann_xml.find("size")
        ann_height = ann_size.find("height")
        ann_width = ann_size.find("width")

        ann_height.text = str(math.floor(int(ann_height.text) / 4))
        ann_width.text = str(math.floor(int(ann_width.text) / 4))

        for ann_object in ann_xml.findall('object'):
            xmin = ann_object.find('bndbox').find('xmin')
            xmax = ann_object.find('bndbox').find('xmax')
            ymin = ann_object.find('bndbox').find('ymin')
            ymax = ann_object.find('bndbox').find('ymax')

            xmin.text = str(math.floor(int(xmin.text) / 4))
            xmax.text = str(math.floor(int(xmax.text) / 4))
            ymin.text = str(math.floor(int(ymin.text) / 4))
            ymax.text = str(math.floor(int(ymax.text) / 4))

            bbox = [xmin, xmax, ymin, ymax]
            ann_bboxes.append(bbox)
        all_bboxes.append(ann_bboxes)

        tree.write(os.path.join(ann_out, ann.split('/')[-1]))

    return ann_paths, all_bboxes


def downsample_images(img_dir, img_out):
    imgs = os.listdir(img_dir)

    for img in imgs:
        img_mat = cv2.imread(os.path.join(img_dir, img))
        img_mat = cv2.resize(img_mat, (math.floor(img_mat.shape[1] / 4), math.floor(img_mat.shape[0] / 4)))

        cv2.imwrite(os.path.join(img_out, img), img_mat)


def generate_segmentation_masks(img_dir, ann_dir, seg_out):
    imgs = os.listdir(img_dir)

    ann_paths, all_bboxes = get_annotations(ann_dir)

    for i, img in enumerate(imgs):
        img_mat = cv2.imread(os.path.join(img_dir, img))
        seg_mask = np.zeros(shape=(img_mat.shape[0], img_mat.shape[1]))

        img_mat = cv2.cvtColor(img_mat, cv2.COLOR_BGR2GRAY)
        i = ann_paths.index(img)
        bboxes = all_bboxes[i]

        for bbox in bboxes:
            bbox = list(map(int, bbox))
            img_mat_region = img_mat[bbox[2]:bbox[3], bbox[0]:bbox[1]]

            img_mat_region = frangi(img_mat_region)
            img_mat_region_unique = np.unique(img_mat_region)

            threshold = img_mat_region_unique[int(img_mat_region_unique.shape[0] * 0.8)]

            _, img_mat_region = cv2.threshold(img_mat_region, threshold,
                                              1,
                                              cv2.THRESH_BINARY)

            kernel = np.ones((3, 3), np.uint8)
            img_mat_region = cv2.erode(img_mat_region, kernel, iterations=1)
            img_mat_region = cv2.dilate(img_mat_region, kernel, iterations=10)

            seg_mask[bbox[2]:bbox[3], bbox[0]:bbox[1]] = img_mat_region

        cv2.imwrite(os.path.join(seg_out, img), seg_mask)

        """
        fig, ax = plt.subplots(1)
        ax.imshow(img_mat)
        rect = patches.Rectangle((bboxes[0], bboxes[2]), bboxes[1] - bboxes[0], bboxes[3] - bboxes[2],
                                 linewidth=1,
                                 edgecolor='r',
                                 facecolor='none')
        
        ax.add_patch(rect)
        plt.show()
        """


generate_segmentation_masks('data/data_scaled', 'data/annotations_scaled', 'data/seg_masks')

