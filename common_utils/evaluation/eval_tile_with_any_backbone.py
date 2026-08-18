import os
import torch
import argparse

import sys
sys.path.append(os.getcwd())

from loguru import logger
from torch.cuda.amp import autocast as autocast
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

from utils.network_loader import load_network
from utils.util import tensor2img

image_size = (960, 512)
npy_save_as = 'png'

# Split images
@torch.no_grad()
def split_image_to_tile(batch_image: torch.Tensor, patch_size=128, overlap_pixels=8):
    """
    Split batch image to tiles in row-first format
    :param batch_image: (b, c, h, w) batch image
    :param patch_size:
    :param overlap_pixels:
    :return:
        tiles: (s, c, patch_size, patch_size),
        factor_tensor: (h, w)
    """
    b, c, h, w = batch_image.shape

    tile_size = patch_size - overlap_pixels

    expand_right = (w // tile_size + 1) * tile_size + overlap_pixels - w
    expand_bottom = (h // tile_size + 1) * tile_size + overlap_pixels - h
    img_padded = torch.nn.functional.pad(batch_image, pad=(0, expand_right, 0, expand_bottom), mode='reflect')

    factor_tensor = torch.zeros((img_padded.shape[2], img_padded.shape[3]),
                                device=img_padded.device)

    h_offsets = list(range(0, h, tile_size))
    w_offsets = list(range(0, w, tile_size))

    tile_list = []

    for h_offset in h_offsets:
        for w_offset in w_offsets:
            tile_img = img_padded[:, :, h_offset: h_offset + patch_size, w_offset: w_offset + patch_size]
            tile_list.append(tile_img)

            factor_tensor[
                h_offset: (h_offset + patch_size),
                w_offset: (w_offset + patch_size)
            ] += 1

    return torch.cat(tile_list, dim=0), factor_tensor[0:h, 0:w]

@torch.no_grad()
def regroup_tile_to_image(batch_tile: torch.Tensor, factor_tensor: torch.Tensor, overlap_pixels=8):
    """
    Regroup tiles to batch image
    :param batch_tile: (s, c, patch_size, patch_size)
    :param factor_tensor: (h, w)
    :param overlap_pixels:
    :return: (b, c, h, w)
    """
    h, w = factor_tensor.shape
    s, c, _, patch_size = batch_tile.shape

    tile_size = patch_size - overlap_pixels
    total_width = (w // tile_size + 1) * tile_size + overlap_pixels
    total_height = (h // tile_size + 1) * tile_size + overlap_pixels

    h_offsets = list(range(0, h, tile_size))
    w_offsets = list(range(0, w, tile_size))

    b = s // (len(h_offsets) * len(w_offsets))

    result_tensor = torch.zeros(
        b, c, total_height, total_width,
        device=batch_tile.device
    )

    batch_count = 0

    for h_offset in h_offsets:
        for w_offset in w_offsets:
            result_tensor[
                :,
                :,
                h_offset: h_offset + patch_size,
                w_offset: w_offset + patch_size
            ] += batch_tile[batch_count * b: (batch_count + 1) * b]

            batch_count += 1

    result_tensor = result_tensor[:, :, 0:h, 0:w]
    return result_tensor / factor_tensor



def process_image(retinol_model, src_image_path, dst_image_path):
    if src_image_path.split('.')[-1].lower() == 'npy':
        npy_data = np.load(src_image_path)
        npy_data = cv2.cvtColor(npy_data, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(npy_data)
    else:
        image = Image.open(src_image_path).convert('RGB')

    # SNRAware
    # img_nf = transform_fn(image).permute(1, 2, 0).numpy() * 255.0
    # img_nf = cv2.blur(img_nf, (5, 5))
    # img_nf = img_nf * 1.0 / 255.0
    # img_nf = torch.Tensor(img_nf).float().permute(2, 0, 1).unsqueeze(0).cuda()

    image = transform_fn(image).unsqueeze(0).cuda()

    with torch.no_grad():
        with autocast(args.fast_eval):
            tile_batch, factor_tensor = split_image_to_tile(image, patch_size=192, overlap_pixels=16)
            # Tile process
            tile_result = retinol_model(tile_batch)
            result = regroup_tile_to_image(tile_result, factor_tensor, overlap_pixels=16)

    if dst_image_path.split('.')[-1].lower() == 'npy':
        Image.fromarray(tensor2img(result[0], min_max=(0, 1))).save(dst_image_path[:-3] + npy_save_as)
    else:
        Image.fromarray(tensor2img(result[0], min_max=(0, 1))).save(dst_image_path)


if __name__ == '__main__':
    logger.info('Start Execution...')
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights', type=str, help='Path of Retinol model weights')
    parser.add_argument('-n', '--network', type=str, help='Network Type')
    parser.add_argument('-o', '--output-dir', type=str, help='Path of the directory that saves result')
    parser.add_argument('-f', '--fast-eval', action='store_true', help='Enable fast eval based on AMP')

    parser.add_argument('-i', '--image', type=str, help='Path of the single image')
    parser.add_argument('-v', '--video', type=str, help='Path of the video')
    parser.add_argument('-s', '--image-set', type=str, help='Path of directory that saves image set')

    args = parser.parse_args()

    # initial model
    model = load_network(args.network, args.weights).cuda()
    model.eval()

    # load
    logger.success('Load pretrained model {} success.'.format(model.__class__.__name__))

    os.makedirs(args.output_dir, exist_ok=True)

    transform_fn = transforms.Compose([
        # transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])

    if args.image is not None:
        # image part
        image_path = args.image
        logger.info('Image: Processing {}'.format(image_path))
        image_name = os.path.basename(image_path)
        dst_path = os.path.join(args.output_dir, image_name)

        process_image(model, image_path, dst_path)

        logger.success('Image: Result saved as {}'.format(dst_path))

    if args.video is not None:
        # video part
        video_path = args.video
        logger.info('Video: Processing {}'.format(video_path))

        video_name = os.path.basename(video_path)
        dst_path = os.path.join(args.output_dir, video_name)

        capture = cv2.VideoCapture(video_path)

        if capture.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = capture.get(cv2.CAP_PROP_FPS)

            video_writer = cv2.VideoWriter(dst_path, fourcc, fps, image_size, True)

            cnt = 0
            while True:
                not_finish, frame = capture.read()

                if not not_finish:
                    logger.info('Video: Process done, save to {}'.format(dst_path))
                    video_writer.release()
                    logger.success('Video: Result saved as {}'.format(dst_path))
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = transform_fn(frame).unsqueeze(0).cuda()
                with torch.no_grad():
                    with autocast(args.fast_eval):
                        result, _ = model(frame)

                    frame_result = tensor2img(result[0], min_max=(0, 1))
                    frame_result = cv2.cvtColor(frame_result, cv2.COLOR_RGB2BGR)
                    video_writer.write(frame_result)

                    # cv2.imshow('frame', frame_result)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                    logger.info('Frame {} done'.format(cnt))
                    cnt += 1

        else:
            logger.error('Video: Failed to open {}'.format(video_path))

    if args.image_set is not None:
        # image set part
        set_path = args.image_set
        image_list = os.listdir(set_path)

        for image_file in image_list:
            image_path = os.path.join(set_path, image_file)
            image_file_name = os.path.basename(image_file)
            logger.info('Set: Processing {}'.format(image_path))
            dst_path = os.path.join(args.output_dir, image_file_name)

            process_image(model, image_path, dst_path)

            logger.success('Set: Result saved as {}'.format(dst_path))

        logger.success('Set: All file in {} processed'.format(set_path))

    logger.info('End execution')
