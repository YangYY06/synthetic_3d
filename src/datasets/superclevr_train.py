import BboxTools as bbt
import cv2
import numpy as np
import os
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import random

colors = np.array([(205, 92, 92), (255, 160, 122), (255, 0, 0), (255, 192, 203),
                   (255, 105, 180), (255, 20, 147), (255, 69, 0), (255, 165, 0),
                   (255, 215, 0), (255, 255, 0), (255, 218, 185), (238, 232, 170),
                   (189, 183, 107), (230, 230, 250), (216, 191, 216), (238, 130, 238),
                   (255, 0, 255), (102, 51, 153), (75, 0, 130), (123, 104, 238),
                   (127, 255, 0), (50, 205, 50), (0, 250, 154), (60, 179, 113),
                   (154, 205, 50), (102, 205, 170), (32, 178, 170), (0, 255, 255),
                   (175, 238, 238), (127, 255, 212), (70, 130, 180), (176, 196, 222),
                   (135, 206, 250), (30, 144, 255)], dtype=np.uint8)


class SuperCLEVRTrain(Dataset):
    def __init__(self, img_path, anno_path, prefix, image_h=352, image_w=520, rotate=None, crop_object=False,
                 obj_mask_path=None, distance=None,
                 mask_path=None, bg_path=None,
                 category=None, subcategory=None,
                 transform=None, enable_cache=True, max_num_obj=8, partial=1.0):
        super().__init__()
        self.img_path = img_path
        self.mask_path = mask_path
        self.anno_path = anno_path
        self.prefix = prefix
        self.category = category
        self.subcategory = subcategory
        self.transform = transform
        self.enable_cache = enable_cache
        self.max_num_obj = max_num_obj
        self.partial = partial
        self.crop_object = crop_object
        self.obj_mask_path = obj_mask_path
        self.image_h = image_h
        self.image_w = image_w
        self.bg_path = bg_path
        self.rotate = rotate
        self.distance = distance

        assert (category is None and subcategory is not None) or (category is not None and subcategory is None)

        self.cache_img = dict()
        self.cache_anno = dict()

        self.prepare()

    def prepare(self):
        file_list = sorted(
            [x.split('.')[0] for x in os.listdir(self.img_path) if x.startswith(self.prefix) and x.endswith('.png')])
        if self.partial < 1.0:
            file_list = file_list[:int(len(file_list) * self.partial)]
        file_list_cate = []
        for f in tqdm(file_list):
            anno = dict(np.load(os.path.join(self.anno_path, f + '.npz'), allow_pickle=True))
            if self.category:
                cate_list = [obj['category'] for obj in anno['objects']]
                if self.category in cate_list:
                    file_list_cate.append(f)
            if self.subcategory:
                cate_list = [obj['sub_category'] for obj in anno['objects']]
                if self.subcategory in cate_list:
                    file_list_cate.append(f)
        self.file_list = file_list_cate

    def __getitem__(self, item):
        img_name = self.file_list[item]

        if img_name in self.cache_img:
            img = self.cache_img[img_name]
            anno = self.cache_anno[img_name]
        else:
            img = Image.open(os.path.join(self.img_path, img_name + '.png'))
            if img.mode != 'RGB':
                img = img.convert('RGB')
                img_ori = img.convert('RGB')
            anno = dict(np.load(os.path.join(self.anno_path, img_name + '.npz'), allow_pickle=True))
            if self.enable_cache:
                self.cache_img[img_name] = img
                self.cache_anno[img_name] = anno

        if self.mask_path:
            m = np.load(os.path.join(self.mask_path, img_name + '.npy'))

        w, h = img.size

        objects = anno['objects']

        """
        all_kp, all_kpvis, obj_mask, all_distance = None, None, np.zeros((h, w), dtype=np.uint8), np.zeros((self.max_num_obj,), dtype=np.float32)
        count_obj = 0
        for obj in objects:
            if obj['category'] != self.category:
                continue
            if all_kp is None:
                all_kp, all_kpvis = np.zeros((self.max_num_obj, len(obj['kp']), 2), dtype=np.int32), np.zeros((self.max_num_obj, len(obj['kp'])), dtype=np.uint8)
            all_kp[count_obj] = obj['kp']
            all_kpvis[count_obj] = obj['kpvis']
            obj_mask = obj_mask | obj['obj_mask']
            all_distance[count_obj] = obj['distance']
            count_obj += 1
        """

        if self.category:
            objects = [obj for obj in objects if obj['category'] == self.category]
        if self.subcategory:
            objects = [obj for obj in objects if obj['sub_category'] == self.subcategory]
        obj = objects[np.random.randint(0, len(objects))]
        all_kp = obj['kp']
        all_kpvis = obj['kpvis']
        principal = obj['principal']
        all_distance = obj['distance']
        count_obj = 1

        obj_mask = np.zeros((h, w), dtype=np.uint8)
        for obj in objects:
            if self.category:
                if obj['category'] != self.category:
                    continue
            if self.subcategory:
                if obj['sub_category'] != self.subcategory:
                    continue
            obj_mask = obj_mask | obj['obj_mask']

        mask = obj_mask
        if self.obj_mask_path:
            obj_m = Image.open(os.path.join(self.obj_mask_path, img_name + '.png'))
            obj_m = np.array(obj_m)
            mask = np.zeros_like(obj_m[:, :, 0])
            mask = np.where(obj_m[:, :, 0] == 64, 0, 1)
            mask = np.where(obj_m[:, :, 1] == 64, mask, 1)
            mask = np.where(obj_m[:, :, 2] == 64, mask, 1)

        x_0 = np.where(np.sum(mask, axis=0) != 0)[0][0]
        x_1 = np.where(np.sum(mask, axis=0) != 0)[0][-1]
        y_0 = np.where(np.sum(mask, axis=1) != 0)[0][0]
        y_1 = np.where(np.sum(mask, axis=1) != 0)[0][-1]

        img = np.array(img)
        if self.crop_object:
            bbox = [x_0, y_0, x_1, y_1]
            resize_rate = float(all_distance / self.distance)
            center = principal
            center = [int(center[1] * resize_rate), int(center[0] * resize_rate)]
            box = bbt.from_numpy(bbox, sorts=('x0', 'y0', 'x1', 'y1'))
            box_ori = box.copy()
            out_shape = [self.image_h, self.image_w]
            box_ori = box_ori.set_boundary(img.shape[0:2])
            box *= resize_rate
            dsize = (int(img.shape[1] * resize_rate), int(img.shape[0] * resize_rate))
            img = cv2.resize(img, dsize=dsize)
            obj_mask = cv2.resize(obj_mask.astype("float32"), dsize=dsize)
            mask = mask.astype('float32')
            mask = cv2.resize(mask, dsize=dsize)
            box1 = bbt.box_by_shape(out_shape, center)
            all_distance /= resize_rate

            if out_shape[0] // 2 - center[0] > 0 or out_shape[1] // 2 - center[1] > 0 or out_shape[0] // 2 + center[0] - \
                    img.shape[0] > 0 or out_shape[1] // 2 + center[1] - img.shape[1] > 0:
                if len(img.shape) == 2:
                    padding = (
                        (max(out_shape[0] // 2 - center[0], 0), max(out_shape[0] // 2 + center[0] - img.shape[0], 0)),
                        (max(out_shape[1] // 2 - center[1], 0), max(out_shape[1] // 2 + center[1] - img.shape[1], 0)))
                else:
                    padding = (
                        (max(out_shape[0] // 2 - center[0], 0), max(out_shape[0] // 2 + center[0] - img.shape[0], 0)),
                        (max(out_shape[1] // 2 - center[1], 0), max(out_shape[1] // 2 + center[1] - img.shape[1], 0)),
                        (0, 0))

                img = np.pad(img, padding, mode='constant')
                box = box.shift([padding[0][0], padding[1][0]])
                box1 = box1.shift([padding[0][0], padding[1][0]])
                obj_mask = np.pad(obj_mask, padding[0:2], mode='constant')
                mask = np.pad(mask, padding[0:2], mode='constant')

            img_box = box1.set_boundary(img.shape[0:2])
            box_in_cropped = img_box.box_in_box(box)
            img = img_box.apply(img)
            mask = img_box.apply(mask)
            obj_mask = img_box.apply(obj_mask)

            proj_foo = bbt.projection_function_by_boxes(box_ori, box_in_cropped, compose=False)
            (y0, y1), (x0, x1) = box_in_cropped.bbox

            for i in range(len(all_kp)):
                all_kp[i][0] = proj_foo[1](all_kp[i][0])
                all_kp[i][1] = proj_foo[0](all_kp[i][1])
            principal[0] = proj_foo[1](principal[0])
            principal[1] = proj_foo[0](principal[1])

            all_kpvis = np.logical_and(all_kpvis, np.all(all_kp >= np.zeros_like(all_kp), axis=1))
            all_kpvis = np.logical_and(all_kpvis, np.all(all_kp < np.array([self.image_w, self.image_h]), axis=1))
            all_kp = np.max([np.zeros_like(all_kp), all_kp], axis=0)
            all_kp = np.min([np.ones_like(all_kp) * (np.array([self.image_w, self.image_h]) - 1), all_kp], axis=0)

        if self.rotate:
            theta = random.gauss(0, self.rotate)
            while (theta >= 90 or theta <= -90):
                theta = random.gauss(0, self.rotate)

            img = Image.fromarray(np.uint8(img))
            obj_mask = Image.fromarray(np.uint8(obj_mask))
            mask = Image.fromarray(np.uint8(mask))
            img = img.rotate(theta)
            obj_mask = obj_mask.rotate(theta)
            mask = mask.rotate(theta)

            mat = np.array([[np.cos(theta / 180 * np.pi), -np.sin(theta / 180 * np.pi)],
                            [np.sin(theta / 180 * np.pi), np.cos(theta / 180 * np.pi)]])

            all_kp = ((all_kp - principal) @ mat).astype('int') + principal

            all_kpvis = np.logical_and(all_kpvis, np.all(all_kp >= np.zeros_like(all_kp), axis=1))
            all_kpvis = np.logical_and(all_kpvis, np.all(all_kp < np.array([self.image_w, self.image_h]), axis=1))
            all_kp = np.max([np.zeros_like(all_kp), all_kp], axis=0)
            all_kp = np.min([np.ones_like(all_kp) * (np.array([self.image_w, self.image_h]) - 1), all_kp], axis=0)

            img = np.array(img)
            obj_mask = np.array(obj_mask)
            mask = np.array(mask)

        if self.bg_path:
            bg_names = os.listdir(self.bg_path)
            idx = np.random.randint(len(bg_names))
            bg = Image.open(os.path.join(self.bg_path, bg_names[idx]))
            bg = bg.resize([img.shape[1], img.shape[0]])
            bg = np.array(bg)
            img_bg = img.copy()
            for j in range(3):
                img_bg[:, :, j] = np.where(mask == 0, bg[:, :, j], img[:, :, j])
            img = img_bg
        img = Image.fromarray(img)

        sample = {'img_name': img_name, 'img': img, 'kp': all_kp,
                  'kpvis': all_kpvis, 'obj_mask': obj_mask,
                  'distance': all_distance, 'num_objs': count_obj, 'principal': principal}

        if self.mask_path:
            sample['mask_names'] = anno['all_mask_names']
            sample['masks'] = m

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.file_list)

    def debug(self, item):
        transform = self.transform
        self.transform = None

        sample = self.__getitem__(item)

        print('kp', sample['kp'].shape)
        print('kpvis', sample['kpvis'].shape)
        print('obj_mask', sample['obj_mask'].shape)
        print('distance', sample['distance'].shape, sample['distance'])
        print('num_objs', sample['num_objs'])
        if 'mask_names' in sample and 'masks' in sample:
            print('mask_names', len(sample['mask_names']))
            print('masks', sample['masks'].shape)

        if 'masks' in sample:
            car_cates = ['truck', 'suv', 'minivan', 'sedan', 'wagon']
            car_parts = ["back_bumper", "back_left_door", "back_left_wheel", "back_left_window", "back_right_door",
                         "back_right_wheel", "back_right_window", "back_windshield", "front_bumper", "front_left_door",
                         "front_left_wheel", "front_left_window", "front_right_door", "front_right_wheel",
                         "front_right_window", "front_windshield", "hood", "roof", "trunk"]
            mask_img = np.zeros((sample['masks'].shape[1], sample['masks'].shape[2], 3), dtype=np.uint8)
            count_color = 0
            for i, mask_name in enumerate(sample['mask_names']):
                obj_name, part_name = mask_name.split('..')
                obj_cate = obj_name.split('_')[0]
                if '.' in part_name:
                    part_name = part_name.split('.')[0]
                print(obj_cate, part_name)
                if obj_cate not in car_cates:
                    continue
                if part_name not in car_parts:
                    continue
                mask_img[sample['masks'][i] == 1] = colors[count_color]
                count_color += 1
            Image.fromarray(mask_img).save(f'debug_parts.png')

        img = np.array(sample['img'])
        print(img.shape)
        kp = sample['kp']
        kpvis = sample['kpvis']
        principal = sample['principal']
        for i in range(len(kp)):
            if kpvis[i] > 0:
                img = cv2.circle(img, (kp[i, 0], kp[i, 1]), radius=1, color=(0, 255, 0), thickness=1)
                pass
        #  img = cv2.circle(img, principal, 5, [255, 0, 0])
        Image.fromarray(img).save(f'debug_img_{item}.png')
        # Image.fromarray(sample['obj_mask'] * 255).save(f'debug_mask_{item}.png')

        self.transform = transform
