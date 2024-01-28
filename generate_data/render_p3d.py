import argparse
import argparse
from collections import Counter
from datetime import datetime
import json
import math
import pdb
import os
import random
import subprocess
import sys
import tempfile
import bpy
import bpy_extras
from mathutils import Vector
import numpy as np
import scipy.io as sio

this_dir = os.path.dirname(__file__)
lib_path = os.path.join(this_dir, '..')
if this_dir not in sys.path:
    sys.path.insert(0, this_dir)
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

import utils


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description="Render images with Blender")

    parser.add_argument('--base_scene_blendfile', type=str, default='data/base_scene.blend')
    parser.add_argument('--properties_json', type=str, default='data/properties_cgpart.json')
    parser.add_argument('--shape_dir', type=str, default='../data/CGPart')
    parser.add_argument('--model_dir', type=str, default='data/save_models_1/')
    parser.add_argument('--material_dir', type=str, default='data/pascal3dp_smartuv')
    # camera option 1
    parser.add_argument('--camera_x', type=float, default=None)
    parser.add_argument('--camera_y', type=float, default=None)
    parser.add_argument('--camera_z', type=float, default=None)
    # camera option 2
    parser.add_argument('--distance', type=float, default=None)
    parser.add_argument('--elevation_mean', type=float, default=None)
    parser.add_argument('--elevation_variance', type=float, default=None)
    parser.add_argument('--elevation_max', type=float, default=None)
    parser.add_argument('--elevation_min', type=float, default=None)

    parser.add_argument('--save_part_mask', type=str2bool, default=True)

    # DTD args
    parser.add_argument('--enable_dtd', type=str2bool, default=False)
    parser.add_argument('--dtd_path', type=str, default='../data/dtd')
    parser.add_argument('--dtd_mat_file', type=str, default='../data/dtd/imdb/imdb.mat')
    parser.add_argument('--dtd_split', type=str, default='train', choices=['train', 'val'])

    # Custom args
    parser.add_argument('--categories', type=str, nargs='+', default=None)
    parser.add_argument('--colors', type=str, nargs='+', default=None)
    parser.add_argument('--sizes', type=str, nargs='+', default=None)
    parser.add_argument('--materials', type=str, nargs='+', default=None)
    parser.add_argument('--textures', type=str, nargs='+', default=None)

    # Object args
    parser.add_argument('--min_objects', type=int, default=1)
    parser.add_argument('--max_objects', type=int, default=1)
    parser.add_argument('--min_dist', type=float, default=0.25)
    parser.add_argument('--margin', type=float, default=0.1)
    parser.add_argument('--min_pixels_per_object', type=int, default=200)
    parser.add_argument('--min_pixels_per_part', type=int, default=20)
    parser.add_argument('--max_retries', type=int, default=150)

    # Output args
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--num_images', type=int, default=5)
    parser.add_argument('--filename_prefix', type=str, default='P3D-Diffusion')
    parser.add_argument('--split', type=str, default='new')
    parser.add_argument('--output_dir', type=str, default='../data/car')
    parser.add_argument('--save_blendfiles', type=str2bool, default=False)
    parser.add_argument('--version', type=str, default='1.0')
    parser.add_argument('--license', type=str, default='Creative Commons Attribution (CC-BY 4.0)')
    parser.add_argument('--date', type=str, default=datetime.today().strftime("%m/%d/%Y"))

    # Rendering args
    parser.add_argument('--use_gpu', type=int, default=1)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--key_light_jitter', type=float, default=3.0)
    parser.add_argument('--fill_light_jitter', type=float, default=3.0)
    parser.add_argument('--back_light_jitter', type=float, default=3.0)
    parser.add_argument('--xy_jitter', type=float, default=0.5)
    parser.add_argument('--z_jitter', type=float, default=0.5)
    parser.add_argument('--render_num_samples', type=int, default=512)
    parser.add_argument('--render_min_bounces', type=int, default=8)
    parser.add_argument('--render_max_bounces', type=int, default=8)
    parser.add_argument('--render_tile_size', type=int, default=256)
    parser.add_argument('--clevr_scene_path', type=str, default=None)
    parser.add_argument('--stretch_x', type=float, default=0.0)
    parser.add_argument('--stretch_y', type=float, default=0.0)
    parser.add_argument('--stretch_z', type=float, default=0.0)

    if '--' not in sys.argv:
        args = parser.parse_args([])
    else:
        args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])

    args.output_image_dir = os.path.join(args.output_dir, 'images')
    args.output_mask_dir = os.path.join(args.output_dir, 'masks')
    args.output_scene_dir = os.path.join(args.output_dir, 'scenes')
    args.output_scene_file = os.path.join(args.output_dir, 'CLEVR_scenes.json')
    args.output_blend_dir = os.path.join(args.output_dir, 'blendfiles')

    return args


def main():
    args = parse_args()
    os.makedirs('tmp', exist_ok=True)

    if args.clevr_scene_path is not None:
        print('Loading scenes from ', args.clevr_scene_path)
        clevr_scene = json.load(open(clevr_scene_path))
        clevr_scene = clevr_scene['scenes']

    # Load DTD info
    if args.enable_dtd:
        dtd_mat = sio.loadmat(args.dtd_mat_file)
        images = dtd_mat['images']
        dtd_ids = images[0, 0][0][0]
        dtd_filenames = images[0, 0][1][0]
        dtd_filenames = np.array([os.path.join(args.dtd_path, 'images', f[0]) for f in dtd_filenames])
        dtd_splits = images[0, 0][2][0]

        dtd_filenames_split = []
        if args.dtd_split == 'train':
            target_s = 1
        elif args.dtd_split == 'val':
            target_s = 2
        for j, f, s in zip(dtd_ids, dtd_filenames, dtd_splits):
            if s == target_s:
                dtd_filenames_split.append(f)

    global color_name_to_rgba, size_mapping, material_mapping, textures_mapping, obj_info
    color_name_to_rgba, size_mapping, material_mapping, textures_mapping, obj_info = utils.load_properties_json(
        args.properties_json, os.path.join(args.shape_dir, 'labels'))

    if args.categories is not None:
        info_pth = {k: obj_info['info_pth'][k] for k in obj_info['info_pth'] if k in args.categories}
        obj_info['info_pth'] = info_pth
    if args.sizes is not None:
        size_mapping = [(k, v) for k, v in size_mapping if k in args.sizes]
    if args.colors is not None:
        color_name_to_rgba = {k: color_name_to_rgba[k] for k in args.colors}
    if args.materials is not None:
        material_mapping = [(k, v) for k, v in material_mapping if k in args.materials]
    if args.textures is not None:
        textures_mapping = args.textures

    num_digits = 6
    prefix = '%s_%s_' % (args.filename_prefix, args.split)
    img_template = '%s%%0%dd.png' % (prefix, num_digits)
    scene_template = '%s%%0%dd.json' % (prefix, num_digits)
    blend_template = '%s%%0%dd.blend' % (prefix, num_digits)
    mask_template = '%s%%0%dd.png' % (prefix, num_digits)
    img_template = os.path.join(args.output_image_dir, img_template)
    scene_template = os.path.join(args.output_scene_dir, scene_template)
    blend_template = os.path.join(args.output_blend_dir, blend_template)
    mask_template = os.path.join(args.output_mask_dir, mask_template)

    os.makedirs(args.output_image_dir, exist_ok=True)
    os.makedirs(args.output_scene_dir, exist_ok=True)
    os.makedirs(args.output_blend_dir, exist_ok=True)
    if args.save_part_mask:
        os.makedirs(args.output_mask_dir, exist_ok=True)

    all_scene_paths = []
    for i in range(args.num_images):
        scene_idx = i + args.start_idx if args.clevr_scene_path is not None else -1
        image_idx = clevr_scene[scene_idx]['image_index'] if scene_idx >= 0 else i + args.start_idx

        img_path = img_template % (image_idx)
        mask_path = mask_template % (image_idx)
        scene_path = scene_template % (image_idx)
        all_scene_paths.append(scene_path)
        blend_path = None
        if args.save_blendfiles == 1:
            blend_path = blend_template % (image_idx)
        num_objects = random.randint(args.min_objects, args.max_objects)

        render_scene(args,
                     num_objects=num_objects,
                     output_index=(image_idx),
                     output_split=args.split,
                     output_image=img_path,
                     output_scene=scene_path,
                     output_blendfile=blend_path,
                     output_mask=mask_path,
                     idx=scene_idx,
                     dtd_fnames=random.sample(dtd_filenames_split, num_objects * 10) if args.enable_dtd else None
                     )

    all_scenes = []
    for scene_path in all_scene_paths:
        with open(scene_path, 'r') as f:
            all_scenes.append(json.load(f))
    output = {
        'info': {
            'date': args.date,
            'version': args.version,
            'split': args.split,
            'license': args.license,
        },
        'scenes': all_scenes
    }
    with open(args.output_scene_file, 'w') as f:
        json.dump(output, f)


def render_scene(args,
                 num_objects=5,
                 output_index=0,
                 output_split='none',
                 output_image='render.png',
                 output_scene='render_json',
                 output_blendfile=None,
                 output_mask=None,
                 idx=-1,
                 dtd_fnames=None
                 ):
    # Load the main blendfile
    bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)
    # utils.set_layer(bpy.data.objects['Ground'], 2)
    # Load materials
    utils.load_materials(args.material_dir)

    # Set render arguments so we can get pixel coordinates later.
    # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
    # cannot be used.
    render_args = bpy.context.scene.render
    render_args.engine = "CYCLES"  # BLENDER_RENDER, CYCLES
    render_args.filepath = output_image
    render_args.resolution_x = args.width
    render_args.resolution_y = args.height
    render_args.resolution_percentage = 100
    render_args.tile_x = args.render_tile_size
    render_args.tile_y = args.render_tile_size
    if args.use_gpu == 1:
        # Blender changed the API for enabling CUDA at some point
        if bpy.app.version < (2, 78, 0):
            bpy.context.user_preferences.system.compute_device_type = 'CUDA'
            bpy.context.user_preferences.system.compute_device = 'CUDA_0'
        else:
            cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
            cycles_prefs.compute_device_type = 'CUDA'

    # Some CYCLES-specific stuff
    bpy.data.worlds['World'].cycles.sample_as_light = True
    bpy.context.scene.cycles.blur_glossy = 2.0
    bpy.context.scene.cycles.samples = args.render_num_samples
    bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
    bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces
    if args.use_gpu == 1:
        bpy.context.scene.cycles.device = 'GPU'

    # This will give ground-truth information about the scene and its objects
    scene_struct = {
        'split': output_split,
        'image_index': output_index,
        'image_filename': os.path.basename(output_image),
        'objects': [],
        'directions': {},
    }

    # Put a plane on the ground so we can compute cardinal directions
    bpy.ops.mesh.primitive_plane_add(radius=5)
    plane = bpy.context.object

    def rand(L):
        return 2.0 * L * (random.random() - 0.5)

    # Add random jitter to camera position
    if args.camera_x is not None:
        bpy.data.objects['Camera'].location[0] = args.camera_x
    if args.camera_y is not None:
        bpy.data.objects['Camera'].location[1] = args.camera_y
    if args.camera_z is not None:
        bpy.data.objects['Camera'].location[2] = args.camera_z
    if args.xy_jitter > 0:
        for i in range(2):
            bpy.data.objects['Camera'].location[i] += rand(args.xy_jitter)
    if args.z_jitter > 0:
        bpy.data.objects['Camera'].location[2] += rand(args.z_jitter)

    if args.distance is not None:
        elevation_mean = args.elevation_mean / 180 * np.pi
        elevation_variance = args.elevation_variance / 180 * np.pi
        elevation_max = args.elevation_max / 180 * np.pi
        elevation_min = args.elevation_min / 180 * np.pi

        elev = random.gauss(elevation_mean, elevation_variance)

        while not (elev <= elevation_max and elev >= elevation_min):
            elev = random.gauss(elevation_mean, elevation_variance)

        #elev = 10 / 180 * np.pi
        bpy.data.objects['Camera'].location[0] = args.distance * np.cos(elev)
        bpy.data.objects['Camera'].location[2] = args.distance * np.sin(elev)
        bpy.data.objects['Camera'].location[1] = 0

    # Figure out the left, up, and behind directions along the plane and record
    # them in the scene structure
    camera = bpy.data.objects['Camera']

    camera.data.lens = 180

    plane_normal = plane.data.vertices[0].normal
    cam_behind = camera.matrix_world.to_quaternion() * Vector((0, 0, -1))
    cam_left = camera.matrix_world.to_quaternion() * Vector((-1, 0, 0))
    cam_up = camera.matrix_world.to_quaternion() * Vector((0, 1, 0))
    plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
    plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
    plane_up = cam_up.project(plane_normal).normalized()

    # Delete the plane; we only used it for normals anyway. The base scene file
    # contains the actual ground plane.
    # utils.delete_object(plane)

    # Save all six axis-aligned directions in the scene struct
    scene_struct['directions']['behind'] = tuple(plane_behind)
    scene_struct['directions']['front'] = tuple(-plane_behind)
    scene_struct['directions']['left'] = tuple(plane_left)
    scene_struct['directions']['right'] = tuple(-plane_left)
    scene_struct['directions']['above'] = tuple(plane_up)
    scene_struct['directions']['below'] = tuple(-plane_up)

    mat = camera.matrix_world
    scene_struct['matrix_world'] = [list(mat[0]), list(mat[1]), list(mat[2]), list(mat[3])]

    mat = camera.matrix_world.inverted()
    scene_struct['matrix_world_inverted'] = [list(mat[0]), list(mat[1]), list(mat[2]), list(mat[3])]

    mat = camera.calc_matrix_camera(render_args.resolution_x, render_args.resolution_y, render_args.pixel_aspect_x,
                                    render_args.pixel_aspect_y)
    scene_struct['projection_matrix'] = [list(mat[0]), list(mat[1]), list(mat[2]), list(mat[3])]

    scene_struct['camera_location'] = tuple(camera.location)


    # Now make some random objects

    objects, blender_objects = add_random_objects(scene_struct, num_objects, args, camera, idx, dtd_fnames)

    if False:
        print(path, output_mask)
        os.system('cp {} {}'.format(path, output_mask))
        os.remove(path)
    '''
    def get_mat_pass_index():
        mat_indices = {}
        mm_idx = 1
        for i, obj in enumerate(blender_objects):
            obj_name = obj.name.split('_')[0]
            mat_indices[obj.name] = (i, -1, mm_idx)
            mm_idx += 1
            obj.pass_index = i + 1
            obj_name = 'truck'
            for pi, part_name in enumerate(obj_info['info_part'][obj_name]):
                mat_indices[obj.name + '.' + part_name] = (i, pi, mm_idx)
                mm_idx += 1

            for mi in range(len(obj.data.materials)):
                mat = obj.data.materials[mi]
                if not mat.name.startswith(obj_name):  # original materials
                    mat.pass_index = mat_indices[obj.name][2]
                else:
                    part_name = mat.name.split('.')[1]
                    mat.pass_index = mat_indices[obj.name + '.' + part_name][2]
        mat_indices = {v[2]: (v[0], v[1], k) for k, v in mat_indices.items()}
        return mat_indices

    def build_rendermask_graph(mat_indices):
        # switch on nodes
        bpy.context.scene.use_nodes = True
        tree = bpy.context.scene.node_tree
        links = tree.links

        # clear default nodes
        for n in tree.nodes:
            tree.nodes.remove(n)

        # create input render layer node
        rl = tree.nodes.new('CompositorNodeRLayers')
        rl.location = 185, 285

        scene = bpy.context.scene
        nodes = scene.node_tree.nodes

        render_layers = nodes['Render Layers']

        num_mat = len(mat_indices)
        num_obj = len(blender_objects)

        ofile_node = nodes.new("CompositorNodeOutputFile")
        path = 'tmp'
        ofile_node.base_path = path
        ofile_node.file_slots.remove(ofile_node.inputs[0])

        idmask_nodes = [nodes.new("CompositorNodeIDMask") for _ in range(num_mat)]
        for _i, o_node in enumerate(idmask_nodes):
            o_node.index = _i + 1

        idmask_obj_nodes = [nodes.new("CompositorNodeIDMask") for _ in range(num_obj)]
        for _i, o_node in enumerate(idmask_obj_nodes):
            o_node.index = _i + 1

        part_colors = {}
        rgb_nodes = [nodes.new("CompositorNodeRGB") for _ in range(num_mat)]
        for _i, rgb_node in enumerate(rgb_nodes):
            obj_idx, mat_idx, mat_name = mat_indices[_i + 1]
            r, g, b = 0.05 * (obj_idx + 1), 0.1 * (mat_idx // 5 + 1), 0.1 * (mat_idx % 5 + 1)
            part_colors[mat_name] = (r, g, b)
            rgb_node.outputs[0].default_value[:3] = (r, g, b)

        mix_nodes = [nodes.new("CompositorNodeMixRGB") for _ in range(num_mat)]
        for _i, o_node in enumerate(mix_nodes):
            o_node.blend_type = "MULTIPLY"

        add_nodes = [nodes.new("CompositorNodeMixRGB") for _ in range(num_mat - 1)]
        for _i, o_node in enumerate(add_nodes):
            o_node.blend_type = "ADD"

        bpy.data.scenes['Scene'].render.layers['RenderLayer'].use_pass_material_index = True
        bpy.data.scenes['Scene'].render.layers['RenderLayer'].use_pass_object_index = True

        for mat_idx in range(num_mat):
            scene.node_tree.links.new(
                render_layers.outputs['IndexMA'],
                idmask_nodes[mat_idx].inputs[0]
            )
            scene.node_tree.links.new(
                idmask_nodes[mat_idx].outputs[0],
                mix_nodes[mat_idx].inputs[1]
            )
            scene.node_tree.links.new(
                rgb_nodes[mat_idx].outputs[0],
                mix_nodes[mat_idx].inputs[2]
            )
            # ofile_node.file_slots.new("part_" + mat_indices[mat_idx+1] + '_')
            # scene.node_tree.links.new(
            #     idmask_nodes[mat_idx].outputs[0],
            #     ofile_node.inputs[mat_idx]
            #     )

        # for obj_idx in range(num_obj):
        #     scene.node_tree.links.new(
        #         render_layers.outputs['IndexOB'],
        #         idmask_obj_nodes[obj_idx].inputs[0]
        #         )
        #     ofile_node.file_slots.new("obj_" + str(blender_objects[obj_idx].name) + '_')
        #     scene.node_tree.links.new(
        #         idmask_obj_nodes[obj_idx].outputs[0],
        #         ofile_node.inputs[num_mat+obj_idx]
        #         )

        mat_idx = 0
        scene.node_tree.links.new(
            mix_nodes[mat_idx + 1].outputs[0],
            add_nodes[mat_idx].inputs[1]
        )
        scene.node_tree.links.new(
            mix_nodes[mat_idx].outputs[0],
            add_nodes[mat_idx].inputs[2]
        )
        for mat_idx in range(1, num_mat - 1):
            scene.node_tree.links.new(
                mix_nodes[mat_idx + 1].outputs[0],
                add_nodes[mat_idx].inputs[1]
            )
            scene.node_tree.links.new(
                add_nodes[mat_idx - 1].outputs[0],
                add_nodes[mat_idx].inputs[2]
            )

        ofile_node.file_slots.new("mask_all_{}_".format(output_index))
        scene.node_tree.links.new(
            add_nodes[-1].outputs[0],
            ofile_node.inputs[0]
        )

        return part_colors

    mat_indices = get_mat_pass_index()
    json_pth = 'tmp/mat_indices_{}.json'.format(output_index)
    json.dump(mat_indices, open(json_pth, 'w'))
    part_colors = build_rendermask_graph(mat_indices)
    '''
    # Render the scene and dump the scene data structure
    if False:
        scene_struct['part_colors'] = p_colors
    scene_struct['objects'] = objects
    scene_struct['relationships'] = compute_all_relationships(scene_struct)
    while True:
        try:
            utils.set_layer(bpy.data.objects['Ground'], 2)

            bpy.data.objects['Lamp_Fill'].location = bpy.data.objects['Camera'].location
            bpy.data.objects['Lamp_Back'].location = bpy.data.objects['Camera'].location
            bpy.data.objects['Lamp_Key'].location = bpy.data.objects['Camera'].location

            # Add random jitter to lamp positions
            if args.key_light_jitter > 0:
                for i in range(3):
                    bpy.data.objects['Lamp_Key'].location[i] += rand(args.key_light_jitter)
            if args.back_light_jitter > 0:
                for i in range(3):
                    bpy.data.objects['Lamp_Back'].location[i] += rand(args.back_light_jitter)
            if args.fill_light_jitter > 0:
                for i in range(3):
                    bpy.data.objects['Lamp_Fill'].location[i] += rand(args.fill_light_jitter)

            bpy.ops.render.render(write_still=True)

            render_args = bpy.context.scene.render

            bpy.context.scene.render.filepath = output_mask
            render_args.engine = 'BLENDER_RENDER'
            render_args.use_antialiasing = False
            utils.set_layer(bpy.data.objects['Ground'], 2)
            utils.set_layer(bpy.data.objects['Lamp_Key'], 2)
            utils.set_layer(bpy.data.objects['Lamp_Fill'], 2)
            utils.set_layer(bpy.data.objects['Lamp_Back'], 2)

            bpy.ops.material.new()
            # obj = bpy.data.objects[0]

            obj = bpy.context.active_object

            mat = bpy.data.materials['Material']
            mat.name = 'Material_0'
            r, g, b = 1, 1, 1
            mat.diffuse_color = [r, g, b]
            mat.use_shadeless = True
            obj.data.materials[0] = mat
            # for mi in range(len(bpy.context.object.data.materials)):
            #     bpy.context.object.data.materials[mi] = mat

            bpy.ops.render.render(write_still=True)
            break
        except Exception as e:
            print(e)
            exit()

    # save_as_json
    '''
    cmd = ['python', './restore_img2json.py', str(output_index)]
    res = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    res.wait()
    if res.returncode != 0:
        print("  os.wait:exit status != 0\n")
        result = res.stdout.read()
        print("after read: {}".format(result))
        raise Exception('error in img2json')

    obj_mask_box = json.load(open('/tmp/obj_mask_{}.json'.format(output_index)))
    _path = '/tmp/obj_mask_{}.json'.format(output_index)
    os.system('rm ' + _path)


    scene_struct['obj_mask_box'] = obj_mask_box
    '''
    with open(output_scene, 'w') as f:
        # json.dump(scene_struct, f, indent=2)
        json.dump(scene_struct, f)

    if output_blendfile is not None:
        bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)


def add_random_objects(scene_struct, num_objects, args, camera, idx=-1, dtd_fnames=None):
    """
    Add random objects to the current blender scene
    """
    print('idx0=', idx)
    positions = []
    objects = []
    blender_objects = []
    obj_pointer = []
    if idx >= 0:
        cob = clevr_scene[idx]['objects']
        num_objects = len(cob)

    print('adding', num_objects, 'objects.')
    # num_objects = 5
    for i in range(num_objects):
        if idx >= 0:
            sf = cob[i]
            theta = sf['rotation']
            obj_name = sf['shape']
            obj_pth = obj_info['info_pth'][obj_name]
            size_name = sf['size']
            r = {a[0]: a[1] for a in size_mapping}[size_name]
            x, y = sf['3d_coords'][:2]
        else:
            # Choose a random size
            # size_name, r = random.choice(size_mapping)
            size_name, r = 'large', 1
            # Choose random color and shape
            obj_name, obj_pth = random.choice(list(obj_info['info_pth'].items()))
            print(obj_name, obj_pth)
            # obj_name, obj_pth = "suv", "car/473dd606c5ef340638805e546aa28d99"

            # Try to place the object, ensuring that we don't intersect any existing
            # objects and that we are more than the desired margin away from all existing
            # objects along all cardinal directions.
            num_tries = 0
            while True:
                # If we try and fail to place an object too many times, then delete all
                # the objects in the scene and start over.
                num_tries += 1
                if num_tries > args.max_retries:
                    for obj in blender_objects:
                        utils.delete_object(obj)
                    return add_random_objects(scene_struct, num_objects, args, camera, dtd_fnames=dtd_fnames)
                # x = random.uniform(-3, 3)
                # y = random.uniform(-3, 3)
                x = 0
                y = 0
                # Choose random orientation for the object.
                theta = 360.0 * random.random()
                #theta = 225
                # Check to make sure the new object is further than min_dist from all
                # other objects, and further than margin along the four cardinal directions
                dists_good = True
                margins_good = True

                def dist_map(x, y, t):
                    theta = t / 180. * math.pi
                    dx1 = x * math.cos(theta) - y * math.sin(theta)
                    dy1 = x * math.sin(theta) + y * math.cos(theta)
                    dx2 = x * math.cos(theta) + y * math.sin(theta)
                    dy2 = x * math.sin(theta) - y * math.cos(theta)
                    return dx1, dy1, dx2, dy2

                def ccw(A, B, C):
                    # return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)
                    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

                # Return true if line segments AB and CD intersect
                def intersect(A, B, C, D):
                    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

                def check(xx, yy, box_xx, box_yy, rr, tt, x, y, box_x, box_y, r, theta):
                    xx1, yy1, xx2, yy2 = dist_map(box_xx / 2 * rr, box_yy / 2 * rr, tt)
                    AA = (xx + xx1, yy + yy1)
                    BB = (xx + xx2, yy + yy2)
                    CC = (xx - xx1, yy - yy1)
                    DD = (xx - xx2, yy - yy2)
                    x1, y1, x2, y2 = dist_map(box_x / 2 * r, box_y / 2 * r, theta)
                    A = (x + x1, y + y1)
                    B = (x + x2, y + y2)
                    C = (x - x1, y - y1)
                    D = (x - x2, y - y2)
                    for (p1, p2) in [(AA, BB), (BB, CC), (CC, DD), (DD, AA), (AA, CC), (BB, DD)]:
                        for (p3, p4) in [(A, B), (B, C), (C, D), (D, A), (A, C), (B, D)]:
                            if intersect(p1, p2, p3, p4):
                                return True
                    return False

                for (objobj, xx, yy, rr, tt) in positions:
                    box_x, box_y, _ = obj_info['info_box'][obj_name]
                    box_xx, box_yy, _ = obj_info['info_box'][objobj]
                    if check(xx, yy, box_xx, box_yy, rr * 1.1, tt, x, y, box_x, box_y, r * 1.1, theta):
                        margins_good = False
                        break

                if dists_good and margins_good:
                    break

        # Actually add the object to the scene
        x_scale = 1 + args.stretch_x * (random.random() - 0.5) * 2
        y_scale = 1 + args.stretch_y * (random.random() - 0.5) * 2
        z_scale = 1 + args.stretch_z * (random.random() - 0.5) * 2

        r = (x_scale, y_scale, z_scale)

        #loc = (x, y, -obj_info['info_z'][obj_name] * z_scale)
        loc = (0, 0, 0)
        current_obj = utils.add_object(args.model_dir, obj_name, obj_pth, r, loc, theta=theta)
        print(args.model_dir, obj_name, obj_pth, r, loc, theta)

        obj = bpy.context.object
        blender_objects.append(obj)
        positions.append((obj_name, x, y, r, theta))

        # Attach a random color
        # rgba=(1,0,0,1)
        if idx >= 0:
            mat_name_out = sf['material']
            mat_name = {a[1]: a[0] for a in material_mapping}[mat_name_out]
            color_name = sf['color']
            rgba = color_name_to_rgba[color_name]
            texture = sf.get('texture', None)
        else:
            mat_name, mat_name_out = random.choice(material_mapping)
            color_name, rgba = random.choice(list(color_name_to_rgba.items()))
            texture = random.choice(textures_mapping)
        mat_freq = {"large": 60, "small": 30}[size_name]
        if texture == 'checkered':
            mat_freq = mat_freq / 2
        utils.modify_color(current_obj, material_name=mat_name, mat_list=obj_info['info_material'][obj_name],
                           color=rgba, texture=texture, mat_freq=mat_freq)

        # Record data about the object in the scene data structure
        pixel_coords = utils.get_camera_coords(camera, obj.location)
        objects.append({
            'shape': obj_name,
            'size': size_name,
            '3d_coords': tuple(obj.location),
            'rotation': theta,
            'pixel_coords': pixel_coords,
            'color': color_name,
            'material': mat_name_out,
            'texture': texture,
            'location': loc,
            'theta': theta,
            'size_r': r
        })

        obj_pointer.append(current_obj)

        print('idx:', idx)
        if idx >= 0:
            part_record = cob[i]['parts']
            print('part_record:', part_record)
            for part_name in part_record:
                part_verts_idxs = obj_info['info_part_labels'][obj_name][part_name]
                part_color_name = part_record[part_name]['color']
                part_rgba = color_name_to_rgba[part_color_name]
                mat_name_out = part_record[part_name]['material']
                mat_name = {a[1]: a[0] for a in material_mapping}[mat_name_out]
                part_texture = part_record[part_name]['texture']
                mat_freq = {"large": 60, "small": 30}[size_name]
                if texture == 'checkered':
                    mat_freq = mat_freq / 2
                utils.modify_part_color(current_obj, part_name, part_verts_idxs,
                                        mat_list=obj_info['info_material'][obj_name],
                                        material_name=mat_name, color_name=part_color_name, color=part_rgba,
                                        texture=part_texture, mat_freq=mat_freq)

            objects[i]['parts'] = part_record

    # if idx >= 0:
    #     return objects, blender_objects

    # Check that all objects are at least partially visible in the rendered image
    '''
    if args.save_part_mask:
        all_visible, visible_parts, all_count, part_colors, path = check_visibility(blender_objects,
                                                                                    args.min_pixels_per_object,
                                                                                    args.min_pixels_per_part,
                                                                                    is_part=True, obj_info=obj_info,
                                                                                    save_part_mask=True)
    else:
        all_visible, visible_parts, all_count = check_visibility(blender_objects, args.min_pixels_per_object,
                                                                 args.min_pixels_per_part, is_part=True,
                                                                 obj_info=obj_info, save_part_mask=False)
    '''
    # pdb.set_trace()
    print('check vis done')
    all_visible = True
    if not all_visible:
        # If any of the objects are fully occluded then start over; delete all
        # objects from the scene and place them all again.
        print('Some objects are occluded; replacing objects')
        for obj in blender_objects:
            utils.delete_object(obj)
        return add_random_objects(scene_struct, num_objects, args, camera, dtd_fnames=dtd_fnames)
    '''
    for i in range(num_objects):
        # randomize part material

        current_obj = obj_pointer[i]
        obj_name = current_obj.name.split('_')[0]
        color_name = objects[i]['color']
        size_name = objects[i]['size']
        objects[i]['pixels'] = all_count[current_obj.name]
        part_list = visible_parts[current_obj.name]
        part_names = random.sample(part_list, min(3, len(part_list)))
        # part_name = random.choice(obj_info['info_part'][obj_name])
        part_record = {}
        for part_name in part_names:
            while True:
                part_color_name, part_rgba = random.choice(list(color_name_to_rgba.items()))
                if part_color_name != color_name:
                    break
            part_name = part_name.split('.')[0]
            # if part_name not in obj_info['info_part_labels'][obj_name]:
            #     print(part_name, obj_name, '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            #     continue
            part_verts_idxs = obj_info['info_part_labels'][obj_name][part_name]
            mat_name, mat_name_out = random.choice(material_mapping)
            texture = random.choice(textures_mapping)
            mat_freq = {"large": 60, "small": 30}[size_name]
            if texture == 'checkered':
                mat_freq = mat_freq / 2
            utils.modify_part_color(current_obj, part_name, part_verts_idxs,
                                    mat_list=obj_info['info_material'][obj_name],
                                    material_name=mat_name, color_name=part_color_name, color=part_rgba,
                                    texture=texture, mat_freq=mat_freq)
            part_record[part_name] = {
                "color": part_color_name,
                "material": mat_name_out,
                "size": objects[i]['size'],
                "texture": texture
            }

        objects[i]['parts'] = part_record
    '''
    if dtd_fnames is not None:
        for i in range(num_objects):
            current_obj = obj_pointer[i]

            # for j in range(len(current_obj.data.materials)):
            for i in range(1):
                img_mat = bpy.data.materials.new(name='dtd')
                img_mat.use_nodes = True
                nodes = img_mat.node_tree.nodes
                links = img_mat.node_tree.links

                material_output = None
                for node in nodes:
                    if node.type == "OUTPUT_MATERIAL":
                        material_output = node
                        break
                if material_output is None:
                    material_output = nodes.new("ShaderNodeOutputMaterial")

                bsdf = img_mat.node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
                bsdf.inputs[3].default_value[0:3] = [1.0, 1.0, 1.0]
                texture_img = img_mat.node_tree.nodes.new('ShaderNodeTexImage')
                texture_img.image = bpy.data.images.load(dtd_fnames[i])

                links.new(bsdf.inputs["Base Color"], texture_img.outputs["Color"])
                links.new(material_output.inputs["Surface"], bsdf.outputs["BSDF"])

                current_obj.data.materials.append(img_mat)

    return objects, blender_objects


def compute_all_relationships(scene_struct, eps=0.2):
    """
    Computes relationships between all pairs of objects in the scene.
    
    Returns a dictionary mapping string relationship names to lists of lists of
    integers, where output[rel][i] gives a list of object indices that have the
    relationship rel with object i. For example if j is in output['left'][i] then
    object j is left of object i.
    """
    all_relationships = {}
    for name, direction_vec in scene_struct['directions'].items():
        if name == 'above' or name == 'below': continue
        all_relationships[name] = []
        for i, obj1 in enumerate(scene_struct['objects']):
            coords1 = obj1['3d_coords']
            related = set()
            for j, obj2 in enumerate(scene_struct['objects']):
                if obj1 == obj2: continue
                coords2 = obj2['3d_coords']
                diff = [coords2[k] - coords1[k] for k in [0, 1, 2]]
                dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
                if dot > eps:
                    related.add(j)
            all_relationships[name].append(sorted(list(related)))
    return all_relationships


def check_visibility(blender_objects, min_pixels_per_object, min_pixels_per_part=None, is_part=False, obj_info=None,
                     save_part_mask=False):
    """
    Check whether all objects in the scene have some minimum number of visible
    pixels; to accomplish this we assign random (but distinct) colors to all
    objects, and render using no lighting or shading or antialiasing; this
    ensures that each object is just a solid uniform color. We can then count
    the number of pixels of each color in the output image to check the visibility
    of each object.

    Returns True if all objects are visible and False otherwise.
    """
    f, path = tempfile.mkstemp(suffix='.png')
    # path = 'output/tmp.

    object_colors, part_colors = render_shadeless(blender_objects, path=path, is_part=is_part, obj_info=obj_info)
    img = bpy.data.images.load(path)

    def srgb_to_linear(x, mod=0.1):
        if x <= 0.04045:
            y = x / 12.92
        else:
            y = ((x + 0.055) / 1.055) ** 2.4
        if mod is not None:
            y = round(y / mod) * mod
        return y

    p = list(img.pixels)
    color_count_raw = Counter((p[i], p[i + 1], p[i + 2], p[i + 3]) for i in range(0, len(p), 4))
    color_count_raw.pop(color_count_raw.most_common(1)[0][0])
    color_count_part = {(srgb_to_linear(k[0]), srgb_to_linear(k[1]), srgb_to_linear(k[2])): v for k, v in
                        color_count_raw.items()}
    color_count_part = Counter(color_count_part)
    color_count_obj = Counter()
    for k, v in color_count_part.items():
        color_count_obj[k[0]] += v
    if not save_part_mask:
        os.remove(path)
    all_visible = True
    visible_parts = {obj.name: [] for obj in blender_objects}
    all_count = {obj.name: 0 for obj in blender_objects}
    if len(color_count_obj) != len(blender_objects):
        all_visible = False
        # return False, visible_parts
    for _, count in color_count_obj.most_common():
        if count < min_pixels_per_object:
            all_visible = False
            # return False, visible_parts
    if is_part:
        for p_name, p_color in part_colors.items():
            try:
                obj_name, part_name = p_name.split('..')
            except:
                print(
                    '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print(p_name)
                pdb.set_trace()
            if color_count_part[p_color] > min_pixels_per_part:
                visible_parts[obj_name].append(part_name)
                all_count[obj_name] += color_count_part[p_color]
    if save_part_mask:
        return all_visible, visible_parts, all_count, part_colors, path
    else:
        return all_visible, visible_parts, all_count


def render_shadeless(blender_objects, path='flat.png', is_part=False, obj_info=None):
    """
    Render a version of the scene with shading disabled and unique materials
    assigned to all objects, and return a set of all colors that should be in the
    rendered image. The image itself is written to path. This is used to ensure
    that all objects will be visible in the final rendered scene.
    """
    render_args = bpy.context.scene.render

    # Cache the render args we are about to clobber
    old_filepath = render_args.filepath
    old_engine = render_args.engine
    old_use_antialiasing = render_args.use_antialiasing

    # Override some render settings to have flat shading
    render_args.filepath = path
    render_args.engine = 'BLENDER_RENDER'
    render_args.use_antialiasing = False

    # Move the lights and ground to layer 2 so they don't render
    utils.set_layer(bpy.data.objects['Lamp_Key'], 2)
    utils.set_layer(bpy.data.objects['Lamp_Fill'], 2)
    utils.set_layer(bpy.data.objects['Lamp_Back'], 2)
    utils.set_layer(bpy.data.objects['Ground'], 2)

    # Add random shadeless materials to all objects
    object_colors = set()
    part_colors = {}
    old_materials = []
    for i, obj in enumerate(blender_objects):
        # need to use iteration to copy by value, otherwise just a pointer is copied
        old_materials.append([])
        for mi in range(len(obj.data.materials)):
            old_materials[i].append(obj.data.materials[mi])
        bpy.ops.material.new()
        mat = bpy.data.materials['Material']
        mat.name = 'Material_%d' % i
        r, g, b = 0.1 * i, 0, 0
        mat.diffuse_color = [r, g, b]
        object_colors.add((r, g, b))
        mat.use_shadeless = True
        if not is_part:
            for mi in range(len(obj.data.materials)):
                obj.data.materials[mi] = mat
        else:
            assert obj_info is not None
            obj_name = obj.name.split('_')[0]
            for pi, part_name in enumerate(obj_info['info_part'][obj_name]):
                bpy.ops.material.new()
                new_mat = bpy.data.materials['Material']
                new_mat.name = obj.name + '..' + part_name
                # pcolor = obj_info['colors'][pi][1]
                # new_mat.diffuse_color = (r/2.+pcolor[0]/2., g/2.+pcolor[1]/2., b/2.+pcolor[2]/2.)
                if i == 1:
                    r, g, b = 0.1 * 10, 0.1 * (pi // 5 + 1), 0.1 * (pi % 5 + 1)
                else:
                    r, g, b = 0.1 * i, 0.1 * (pi // 5 + 1), 0.1 * (pi % 5 + 1)
                new_mat.diffuse_color = (r, g, b)
                pc = new_mat.diffuse_color
                part_colors[new_mat.name] = (r, g, b)
                new_mat.use_shadeless = True
            for mi in range(len(obj.data.materials)):
                orig_mat = obj.data.materials[mi]
                if not orig_mat.name.startswith(obj_name):  # original materials
                    obj.data.materials[mi] = mat
                else:
                    part_name = orig_mat.name.split('.')[1]
                    obj.data.materials[mi] = bpy.data.materials[obj.name + '..' + part_name]

    # Render the scene
    bpy.ops.render.render(write_still=True)
    print('render still done 1')

    # Undo the above; first restore the materials to objects
    for mat, obj in zip(old_materials, blender_objects):
        for mi in range(len(obj.data.materials)):
            obj.data.materials[mi] = mat[mi]
        # obj.data.materials = mat

    print('render still done 2')

    # Move the lights and ground back to layer 0
    utils.set_layer(bpy.data.objects['Lamp_Key'], 0)
    utils.set_layer(bpy.data.objects['Lamp_Fill'], 0)
    utils.set_layer(bpy.data.objects['Lamp_Back'], 0)
    utils.set_layer(bpy.data.objects['Ground'], 0)

    # Set the render settings back to what they were
    render_args.filepath = old_filepath
    render_args.engine = old_engine
    render_args.use_antialiasing = old_use_antialiasing

    # # delete the new created materials
    # for mat in bpy.data.materials:
    #     # if mat.name.startswith('Material_'):
    #     if not mat.users:
    #         bpy.data.materials.remove(mat)

    print('render still done 3')
    return object_colors, part_colors


def _render_shadeless(blender_objects, path='flat.png', is_part=False, obj_info=None):
    # compositor masks  
    def get_mat_pass_index():
        mat_indices = {}
        mm_idx = 1
        for i, obj in enumerate(blender_objects):
            obj_name = obj.name.split('_')[0]
            mat_indices[obj.name] = mm_idx
            obj.pass_index = i + 1
            for pi, part_name in enumerate(obj_info['info_part'][obj_name]):
                mat_indices[obj.name + '.' + part_name] = mm_idx
                # mat_indices[mm_idx] = obj.name+'.'+part_name
                mm_idx += 1

            for mi in range(len(obj.data.materials)):
                mat = obj.data.materials[mi]
                if not mat.name.startswith(obj_name):  # original materials
                    mat.pass_index = mat_indices[obj.name]
                else:
                    part_name = mat.name.split('.')[1]
                    mat.pass_index = mat_indices[obj.name + '.' + part_name]
        mat_indices = {v: k for k, v in mat_indices.items()}
        return mat_indices

    def build_rendermask_graph(mat_indices):
        # switch on nodes
        bpy.context.scene.use_nodes = True
        tree = bpy.context.scene.node_tree
        links = tree.links

        # clear default nodes
        for n in tree.nodes:
            tree.nodes.remove(n)

        # create input render layer node
        rl = tree.nodes.new('CompositorNodeRLayers')
        rl.location = 185, 285

        scene = bpy.context.scene
        nodes = scene.node_tree.nodes

        render_layers = nodes['Render Layers']

        num_mat = len(mat_indices)
        num_obj = len(blender_objects)

        ofile_node = nodes.new("CompositorNodeOutputFile")
        ofile_node.base_path = path
        ofile_node.file_slots.remove(ofile_node.inputs[0])

        idmask_nodes = [nodes.new("CompositorNodeIDMask") for _ in range(num_mat)]
        for _i, o_node in enumerate(idmask_nodes):
            o_node.index = _i + 1

        idmask_obj_nodes = [nodes.new("CompositorNodeIDMask") for _ in range(num_obj)]
        for _i, o_node in enumerate(idmask_obj_nodes):
            o_node.index = _i + 1

        part_colors = {}
        object_colors = set()
        for obj in blender_objects:
            while True:
                r, g, b = [random.random() for _ in range(3)]
                if (r, g, b) not in object_colors: break
            object_colors.add((r, g, b))
        object_colors = list(object_colors)
        object_colors = {obj.name: object_colors[i] for i, obj in enumerate(blender_objects)}

        # rgb_nodes = [nodes.new("CompositorNodeRGB") for _ in range(num_mat)]
        # for _i, rgb_node in enumerate(rgb_nodes):
        #     pcolor = obj_info['colors'][_i][1]
        #     r, g, b = object_colors[mat_indices[_i+1].split('.')[0]]
        #     rgb = (r/2.+pcolor[0]/2., g/2.+pcolor[1]/2., b/2.+pcolor[2]/2.)
        #     part_colors[mat_indices[_i+1]] = rgb
        #     rgb_node.outputs[0].default_value[:3] = rgb

        # mix_nodes = [nodes.new("CompositorNodeMixRGB") for _ in range(num_mat)]
        # for _i, o_node in enumerate(mix_nodes):    
        #     o_node.blend_type = "MULTIPLY"

        # add_nodes = [nodes.new("CompositorNodeMixRGB") for _ in range(num_mat-1)]
        # for _i, o_node in enumerate(add_nodes):    
        #     o_node.blend_type = "ADD"

        bpy.data.scenes['Scene'].render.layers['RenderLayer'].use_pass_material_index = True
        bpy.data.scenes['Scene'].render.layers['RenderLayer'].use_pass_object_index = True

        for mat_idx in range(num_mat):
            scene.node_tree.links.new(
                render_layers.outputs['IndexMA'],
                idmask_nodes[mat_idx].inputs[0]
            )
            # scene.node_tree.links.new(
            #     idmask_nodes[mat_idx].outputs[0],
            #     mix_nodes[mat_idx].inputs[1]
            #     )
            # scene.node_tree.links.new(
            #     rgb_nodes[mat_idx].outputs[0],
            #     mix_nodes[mat_idx].inputs[2]
            #     )
            ofile_node.file_slots.new("part_" + mat_indices[mat_idx + 1] + '_')
            scene.node_tree.links.new(
                idmask_nodes[mat_idx].outputs[0],
                ofile_node.inputs[mat_idx]
            )

        for obj_idx in range(num_obj):
            scene.node_tree.links.new(
                render_layers.outputs['IndexOB'],
                idmask_obj_nodes[obj_idx].inputs[0]
            )
            ofile_node.file_slots.new("obj_" + str(blender_objects[obj_idx].name) + '_')
            scene.node_tree.links.new(
                idmask_obj_nodes[obj_idx].outputs[0],
                ofile_node.inputs[num_mat + obj_idx]
            )

        # mat_idx = 0
        # scene.node_tree.links.new(
        #     mix_nodes[mat_idx+1].outputs[0],
        #     add_nodes[mat_idx].inputs[1]
        #     )
        # scene.node_tree.links.new(
        #     mix_nodes[mat_idx].outputs[0],
        #     add_nodes[mat_idx].inputs[2]
        #     )
        # for mat_idx in range(1, num_mat-1):
        #     scene.node_tree.links.new(
        #         mix_nodes[mat_idx+1].outputs[0],
        #         add_nodes[mat_idx].inputs[1]
        #         )
        #     scene.node_tree.links.new(
        #         add_nodes[mat_idx-1].outputs[0],
        #         add_nodes[mat_idx].inputs[2]
        #         )

        # scene.node_tree.links.new(
        #     add_nodes[-1].outputs[0],
        #     ofile_node.inputs['Image']
        #     )

        return object_colors, part_colors

    mat_indices = get_mat_pass_index()
    object_colors, part_colors = build_rendermask_graph(mat_indices)

    return object_colors, part_colors


if __name__ == '__main__':
    main()
