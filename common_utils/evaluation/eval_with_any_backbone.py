import os
import torch
import argparse
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
            # SNRAware
            # result = retinol_model(image, img_nf)

            # Tile process
            result = retinol_model(image)

    if dst_image_path.split('.')[-1].lower() == 'npy':
        Image.fromarray(tensor2img(result[0], min_max=(0, 1))).save(dst_image_path[:-3] + npy_save_as)
    else:
        Image.fromarray(tensor2img(result[0], min_max=(0, 1))).save(dst_image_path)


if __name__ == '__main__':
    logger.info('Start Execution...')
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights', type=str, help='Path of weights')
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
