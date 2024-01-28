import os
import sys
import argparse
import json
import os
import pdb
import random
import re
import sys

import bpy
import bmesh


def parse_args():

    parser = argparse.ArgumentParser(description="Pre-process p3d data")

    parser.add_argument('--off_path', type=str, default='/home/jiahao/pretrain_6d_pose/data/Pascal_CAD')
    parser.add_argument('--category', type=str, default='aeroplane')
    parser.add_argument('--save_dir', type=str, default='/home/jiahao/pretrain_6d_pose/generate_data/data/pascal3dp_models')
    print(parser.parse_args())
    args = parser.parse_args()
    return args

def install():
    bpy.ops.wm.addon_install(filepath='/home/jiahao/pretrain_6d_pose/generate_data/load_off.py',overwrite=True)
    bpy.ops.wm.addon_enable(module='load_off')

def main():
    off_path='/home/jiahao/pretrain_6d_pose/data/Pascal_CAD'
    objs = bpy.data.objects
    objs.remove(objs["Cube"], do_unlink=True)
    objs.remove(objs["Camera"], do_unlink=True)
    objs.remove(objs["Lamp"], do_unlink=True)

    for category in ['car','tvmonitor','sofa','bicycle','train','bottle','boat','diningtable']:
        save_dir='/home/jiahao/pretrain_6d_pose/generate_data/data/pascal3dp_models'

        os.makedirs(os.path.join(save_dir, category), exist_ok=True)
        subcates=os.listdir(os.path.join(off_path, category))


        for name in subcates:
            bpy.ops.import_mesh.off(filepath=os.path.join(off_path, category, name),filter_glob='*.off')
            filepath = os.path.join(save_dir, category, category + '_' + name.split('.')[0] + '.blend')
            bpy.ops.wm.save_as_mainfile(filepath=filepath)

            objs = bpy.data.objects
            objs.remove(objs[name.split('.')[0]], do_unlink=True)


if __name__ == '__main__':
    install()
    main()
