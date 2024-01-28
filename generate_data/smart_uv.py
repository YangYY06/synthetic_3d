import os

import bpy


def main():
    in_path = '/home/jiahao/pretrain_6d_pose/generate_data/data/pascal3dp_models'
    out_path = '/home/jiahao/pretrain_6d_pose/generate_data/data/pascal3dp_smartuv'
    for cate in ['tvmonitor', 'sofa', 'bicycle', 'train', 'bottle', 'boat', 'diningtable']:
        path = os.path.join(in_path, cate)

        for f in sorted(os.listdir(path)):

            subcate = f[:-6]
            print(cate, subcate)
            bpy.ops.wm.open_mainfile(filepath=os.path.join(path, f))
            for obj in bpy.context.scene.objects:
                bpy.context.scene.objects.active = obj
                bpy.ops.uv.smart_project()
            bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out_path, f))


if __name__ == '__main__':
    main()
