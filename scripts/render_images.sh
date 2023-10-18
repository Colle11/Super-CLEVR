cd image_generation

# The CUDA_VISIBLE_DEVICES environment variable restricts your CUDA application
# execution to a specific device or set of devices for debugging and testing.
# To use it, set CUDA_VISIBLE_DEVICES to a comma-separated list of device IDs
# to make only those devices visible to the application.
# CUDA_VISIBLE_DEVICES=0 selects GPU 0 to perform any CUDA tasks.
# See also https://developer.nvidia.com/blog/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/
# for more details.
CUDA_VISIBLE_DEVICES=0 \
blender --background \
    --python super_restore_render_images.py -- \
    --start_idx 0 \
    --num_images 2 \
    --shape_dir ../../CGPart \
    --model_dir data/save_models_1/ \
    --properties_json data/properties_cgpart.json \
    --margin 0.1 \
    --save_blendfiles 0 \
    --max_retries 150 \
    --width 640 \
    --height 480 \
    --use_gpu 1 \
    # Remove the arg '--use_gpu 1' to render images with the CPU.


    #--output_image_dir ../output/ver_texture_same/images/ \
    #--output_scene_dir ../output/ver_texture_same/scenes/ \
    #--output_blend_dir ../output/ver_texture_same/blendfiles \
    #--output_scene_file ../output/ver_texture_same/superCLEVR_scenes.json \
    #--is_part 1 \
    #--load_scene 1 \
    #--clevr_scene_path ../output/ver_mask/superCLEVR_scenes.json 

    # --shape_color_co_dist_pth data/dist/shape_color_co_super.npz \

    # --clevr_scene_path ../output/superCLEVR_scenes_5.json

    # --color_dist_pth data/dist/color_dist.npz \
    # --mat_dist_pth data/dist/mat_dist.npz \
    # --shape_dist_pth data/dist/shape_dist.npz \


cd ..

