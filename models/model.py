import random
import numpy as np
import torch
import tqdm
import time
from loguru import logger
import os
from PIL import Image

import data
import utils.util as RetinolUtil
import utils.ramps as ramps
import json
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
import csv
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from utils.scheduler import GradualWarmupScheduler
from utils.Profiler import Profiler


class Retinol:
    def __init__(self, networks, paired_loader, unpaired_loader, val_loaders_dict, losses, metrics, optimizer,
                 visualize_ena,
                 resume_state, init_method, tensorboard_log_dir, options):
        assert len(networks) == 2
        self.network = networks[0].cuda()
        self.ema_network = networks[1].cuda()

        self.paired_loader = paired_loader
        self.unpaired_loader = unpaired_loader
        self.val_loaders_dict = val_loaders_dict

        assert len(losses) > 0
        self.loss_function_dict = losses
        self.epoch_loss_recorder = {}
        self.clear_epoch_loss_recorder()

        self.metrics = metrics
        self.iqa_csv_path = options['path']['iqa_csv']
        self.init_csv()

        optimizer['args'].update({
            "params": list(filter(lambda p: p.requires_grad, self.network.parameters()))
        })
        self.optimizer = RetinolUtil.init_obj(optimizer, default_file_name='torch.optim', init_type='Optimizer')

        # warmup
        if options['train']['enable_warmup']:
            logger.info("Using warmup and cosine strategy")
            warmup_epochs = options['train']['warmup_epochs']
            scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                    options['train']['n_epoch'] - warmup_epochs,
                                                                    eta_min=1e-6)
            self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=warmup_epochs,
                                                    after_scheduler=scheduler_cosine)
            # self.scheduler.step()
        else:
            logger.info("Using cosine strategy")
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                  options['train']['n_epoch'],
                                                                  eta_min=1e-6)
            # self.scheduler.step()

        # resume training state
        if resume_state is None:
            self.global_step = 0
            self.start_epoch = 0

            self.init_weights(self.network, init_type=init_method)
            self.init_weights(self.ema_network, init_type=init_method)
        else:
            self.load_from_disk(resume_state, options)

        self.summery_writer = SummaryWriter(log_dir=tensorboard_log_dir)

        # init current_best_info
        self.current_best_info = {}
        for val_set_name in self.val_loaders_dict:
            self.current_best_info[val_set_name] = {}
            for metric_name in self.metrics:
                self.current_best_info[val_set_name][metric_name] = {
                    'value': -1,
                    'network': 'student',
                    'epoch': -1,
                    'step': -1
                }

        self.global_best_info = {
            'value': -1,
            'network': 'student',
            'epoch': -1,
            'step': -1
        }

    def train(self, options):
        logger.info('Train start')
        if options['speed_up']['enable_amp']:
            logger.info('AMP enable.')
        else:
            logger.info('AMP disable.')

        result_path = os.path.join(options['path']['results'], options['phase'])
        os.makedirs(result_path, exist_ok=True)

        # switch to train mode
        self.network.train()
        self.ema_network.train()

        scaler = GradScaler(enabled=options['speed_up']['enable_amp'])
        torch.cuda.empty_cache()

        epoch_index = 0
        for epoch_index_raw in range(self.start_epoch, options['train']['n_epoch']):
            epoch_index = epoch_index_raw + 1
            if epoch_index > options['train']['finish_epoch']:
                logger.info('Reach finish epoch {}'.format(options['train']['finish_epoch']))
                break
            logger.info('Train of epoch {} start'.format(epoch_index))
            self.optimizer.zero_grad(set_to_none=True)
            if epoch_index < options['train']['semi_start_epoch']:
                self.supervise_train_step(options, epoch_index, scaler)
            else:
                self.semi_supervise_train_step(options, epoch_index, scaler)

            logger.info('Current LearningRate {}'.format(self.scheduler.get_last_lr()[0]))
            self.summery_writer.add_scalar('Learning rate', self.scheduler.get_last_lr()[0], epoch_index)

            self.print_loss_per_epoch()

            self.scheduler.step()

            if epoch_index % options['train']['val_save_epoch'] == 0:
                # val part
                logger.info('Validation of epoch {} start'.format(epoch_index))
                self.eval_and_save_to_tensorboard(epoch_index, options)

                logger.info('Validation of epoch {} end'.format(epoch_index))
            elif epoch_index % options['train']['val_epoch'] == 0:
                if options['train']['reproduce_target'] is not None and epoch_index % options['train']['reproduce_target'] != 0:
                    self.fake_eval(epoch_index, options, self.val_loaders_dict)
                else:
                    self.eval_step_without_save(epoch_index, options, self.val_loaders_dict)

            if epoch_index % options['train']['save_checkpoint_epoch'] == 0 or epoch_index == options['train']['reproduce_target']:
                # save checkpoint
                self.save_everything(epoch_index, epoch_index, options)
                logger.info('Checkpoint of epoch {} saved'.format(epoch_index))

            logger.info('Train of epoch {} end'.format(epoch_index))

            if epoch_index == options['train']['reproduce_target']:
                break

        logger.info('Train end')

        # eval all best checkpoint and save to dir
        # logger.info('Final evaluation begin')
        # for eval_set_name in self.val_loaders_dict:
        #     self.load_from_disk(os.path.join(options['path']['checkpoint'], '{}_{}'.format(eval_set_name, 'psnr')))
        #     self.val_loaders_dict[eval_set_name].full_set = True
        #     self.eval_and_save_to_dir(options, eval_set_name)
        #
        #     logger.success('Final eval of {} done'.format(eval_set_name))
        #     self.val_loaders_dict[eval_set_name].full_set = False

    def test(self, options):
        logger.info('Test start')
        if options['speed_up']['enable_amp']:
            logger.info('AMP enable.')
        else:
            logger.info('AMP disable.')

        result_path = os.path.join(options['path']['results'])
        os.makedirs(result_path, exist_ok=True)

        for eval_set_name in self.val_loaders_dict:
            self.eval_and_save_to_dir(options, eval_set_name)

            logger.success('Final eval of {} done'.format(eval_set_name))
        # self.ema_network.train()
        logger.info('Test end')

    def supervise_train_step(self, options, epoch, scaler):
        supervise_pbar = tqdm.tqdm(self.paired_loader)
        supervise_pbar.set_description('Supervised')
        current_epoch_displayed_input = False

        # eval_iter = (len(supervise_pbar) // options['train']['iter_per_optim_step']) // options['train'][
        #     'val_per_epoch']
        with Profiler(options['path']['profile_path']) as prof:
            for iter_index, train_data in enumerate(supervise_pbar):
                train_student_input = train_data['student_input']
                # train_teacher_input = train_data['teacher_input']
                train_ground_truth = train_data['ground_truth']

                if options['debug'] and not current_epoch_displayed_input:
                    self.display_train_data_in_tensorboard(train_student_input, train_ground_truth, epoch)
                    current_epoch_displayed_input = True

                train_student_input = train_student_input.cuda()
                # train_teacher_input = train_teacher_input.cuda()
                train_ground_truth = train_ground_truth.cuda()

                with autocast(options['speed_up']['enable_amp']):
                    student_result = self.network(train_student_input)
                    student_result = torch.clamp(student_result, 0, 1)

                    loss_supervise = self.loss_fn(student_result, train_ground_truth, 'supervised_loss', epoch,
                                                  'Supervised')

                    loss = loss_supervise

                # calc gradient and backward
                if options['speed_up']['enable_amp']:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (iter_index % options['train']['iter_per_optim_step'] == (options['train']['iter_per_optim_step'] - 1)) or \
                        iter_index + 1 == len(supervise_pbar):
                    if options['speed_up']['enable_amp']:
                        scaler.step(self.optimizer)
                        scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                    else:
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)

                    # EMA
                    update_ema_variables(self.network, self.ema_network, get_ema_decay(epoch, options),
                                         self.global_step)
                    self.global_step += 1

                    # visualization
                    # supervise_pbar.set_description('Loss {:.4f}'.format(loss.item()))

                prof.step()

    def semi_supervise_train_step(self, options, epoch, scaler):
        total_iter_cnt = min(len(self.paired_loader), len(self.unpaired_loader))
        paired_loader_iter = iter(self.paired_loader)
        unpaired_loader_iter = iter(self.unpaired_loader)

        # consistency_weight = options['train']['semi_loss_weight']

        semi_loss_weight = options['train']['semi_loss_weight']
        semi_loss_consistency_ramp_up = options['train']['semi_loss_consistency_ramp_up']

        pbar = tqdm.tqdm(range(total_iter_cnt))
        pbar.set_description('Semi-supervised')
        current_epoch_displayed_input = False

        consistency_weight = get_current_consistency_weight(semi_loss_weight, semi_loss_consistency_ramp_up,
                                                            epoch)
        # eval_iter = (total_iter_cnt // options['train']['iter_per_optim_step']) // options['train']['val_per_epoch']
        for iter_index in pbar:
            # paired data here
            paired_train_data = next(paired_loader_iter)
            paired_train_student_input = paired_train_data['student_input']
            paired_train_teacher_input = paired_train_data['teacher_input']
            paired_train_ground_truth = paired_train_data['ground_truth']

            if options['debug'] and not current_epoch_displayed_input:
                self.display_train_data_in_tensorboard(paired_train_student_input, paired_train_ground_truth, epoch)

            # unpaired data here
            unpaired_train_data = next(unpaired_loader_iter)
            unpaired_train_student_input = unpaired_train_data['student_input']
            unpaired_train_teacher_input = unpaired_train_data['teacher_input']
            # CutMix
            cutmix_train_student_input, cutmix_train_teacher_input, paired_mask = data.RandomCutMix(
                unpaired_train_student_input,
                unpaired_train_teacher_input,
                paired_train_student_input,
                # torch.zeros_like(unpaired_train_student_input),
                1 - (epoch - options['train']['semi_start_epoch']) / (
                        options['train']['n_epoch'] - options['train']['semi_start_epoch']))

            if options['debug'] and not current_epoch_displayed_input:
                self.display_train_data_in_tensorboard(cutmix_train_student_input, None, epoch)
                current_epoch_displayed_input = True

            paired_train_student_input = paired_train_student_input.cuda()
            paired_train_teacher_input = paired_train_teacher_input.cuda()
            paired_train_ground_truth = paired_train_ground_truth.cuda()

            unpaired_train_student_input = unpaired_train_student_input.cuda()
            unpaired_train_teacher_input = unpaired_train_teacher_input.cuda()
            cutmix_train_student_input = cutmix_train_student_input.cuda()
            cutmix_train_teacher_input = cutmix_train_teacher_input.cuda()

            total_pixel_per_patch = paired_train_student_input.shape[1] * paired_train_student_input.shape[2] * \
                                    paired_train_student_input.shape[3]
            paired_mask = paired_mask.cuda()
            unpaired_mask = 1 - paired_mask
            paired_pixel_count = torch.sum(paired_mask, dim=[1, 2, 3])

            with autocast(options['speed_up']['enable_amp']):
                paired_student_result = self.network(paired_train_student_input)
                paired_student_result = torch.clamp(paired_student_result, 0, 1)
                loss_supervise = self.loss_fn(paired_student_result, paired_train_ground_truth, 'supervised_loss',
                                              epoch,
                                              'Paired-student-GT')

            with autocast(options['speed_up']['enable_amp']):
                paired_teacher_result = self.ema_network(paired_train_teacher_input)
                paired_teacher_result = torch.clamp(paired_teacher_result, 0, 1)
                loss_paired_contrastive = self.loss_fn([paired_teacher_result,
                                                        paired_student_result,
                                                        torch.ones_like(paired_student_result), total_pixel_per_patch],
                                                       paired_train_ground_truth,
                                                       'paired_contrastive_loss', epoch, 'Paired-contrastive')
            #     # paired_teacher_result = paired_teacher_result.detach_()
            #     # for key, feature in paired_teacher_feature_dict.items():
            #     #     paired_teacher_feature_dict[key].detach_()
            #     loss_paired_consist = self.loss_fn(paired_student_result, paired_teacher_result, 'paired_consist_loss',
            #                                        epoch, 'Paired-student-teacher')
            #     loss_paired_feature = self.loss_fn(paired_student_feature_dict, paired_teacher_feature_dict,
            #                                        'paired_feature_loss', epoch, 'Paired-feature')

            with autocast(options['speed_up']['enable_amp']):
                # loss_paired_part = loss_supervise + consistency_weight * loss_paired_consist + \
                # consistency_weight * loss_paired_feature
                loss_paired_part = loss_supervise + loss_paired_contrastive

            if options['speed_up']['enable_amp']:
                scaler.scale(loss_paired_part).backward()
            else:
                loss_paired_part.backward()

            with autocast(options['speed_up']['enable_amp']):
                # noise_student = torch.clamp(torch.randn_like(cutmix_train_student_input), -1, 1) * options['train'][
                #     'noise_scale']
                unpaired_student_result = self.network(cutmix_train_student_input)
                unpaired_student_result = torch.clamp(unpaired_student_result, 0, 1)

            with autocast(options['speed_up']['enable_amp']):
                with torch.no_grad():
                    # noise_teacher = torch.clamp(torch.randn_like(cutmix_train_teacher_input), -1, 1) * \
                    #                 options['train'][
                    #                     'noise_scale']
                    unpaired_teacher_result = self.ema_network(
                        cutmix_train_teacher_input)

                    unpaired_teacher_result = torch.clamp(unpaired_teacher_result, 0, 1)

                loss_unpaired_consist = self.loss_fn([unpaired_student_result, unpaired_mask,
                                                      total_pixel_per_patch - paired_pixel_count],
                                                     unpaired_teacher_result,
                                                     'unpaired_consist_loss', epoch, 'Unpaired-student-teacher')

                loss_unpaired_cutmix = self.loss_fn([unpaired_student_result, paired_mask, paired_pixel_count],
                                                    paired_train_ground_truth, 'cutmix_loss', epoch, 'CutMix-feature')

                loss_unpaired_contrastive = self.loss_fn([cutmix_train_student_input *
                                                          (torch.mean(unpaired_teacher_result, dim=[1, 2, 3]) /
                                                           torch.mean(cutmix_train_student_input, dim=[1, 2, 3])).view((-1, 1, 1, 1)),
                                                          unpaired_student_result,
                                                          unpaired_mask, total_pixel_per_patch - paired_pixel_count],
                                                          # paired_mask, paired_pixel_count],
                                                          # torch.ones_like(unpaired_train_student_input), total_pixel_per_patch],
                                                         unpaired_teacher_result,
                                                         'unpaired_contrastive_loss', epoch, 'Unpaired-contrastive')

                # loss_unsupervise = loss_semi_supervise + loss_paired_semi
                loss_unpaired_part = loss_unpaired_cutmix + consistency_weight * \
                                     (loss_unpaired_consist + loss_unpaired_contrastive)

            # calc gradient and backward
            if options['speed_up']['enable_amp']:
                scaler.scale(loss_unpaired_part).backward()
            else:
                loss_unpaired_part.backward()

            if iter_index % options['train']['iter_per_optim_step'] == (options['train']['iter_per_optim_step'] - 1) or\
                    iter_index + 1 == total_iter_cnt:
                if options['speed_up']['enable_amp']:
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                else:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                # EMA
                update_ema_variables(self.network, self.ema_network, get_ema_decay(epoch, options),
                                     self.global_step)
                self.global_step += 1

                # visualization
                # pbar.set_description(
                #     'Supervised Loss {:.4f}, Unsupervised Loss {:.4f}'.format(loss_supervise.item(),
                #                                                               loss_unsupervise.item()))

                # eval
                # if (iter_index // options['train']['iter_per_optim_step'] + 1) % eval_iter == 0:
                #     self.eval_step_without_save(epoch, options, self.val_loaders_dict)

    def eval_and_save_to_tensorboard(self, epoch_index, options):
        self.network.eval()
        self.ema_network.eval()

        with torch.no_grad():
            eval_iqa_result = {
                'epoch': epoch_index,
                'step': self.global_step,
                'result': {}
            }

            for eval_set_name, eval_loader in self.val_loaders_dict.items():
                eval_iqa_result['result'][eval_set_name] = {}

                eval_loader_pbar = tqdm.tqdm(eval_loader)
                image_count = 0
                raw_metrics_result = {}
                ema_metrics_result = {}
                for metric_name in self.metrics:
                    raw_metrics_result[metric_name] = 0
                    ema_metrics_result[metric_name] = 0

                val_image_size = eval_loader.dataset.image_size
                for val_data in eval_loader_pbar:
                    file_name = val_data['file_name']
                    val_student_input = val_data['student_input']
                    val_teacher_input = val_data['teacher_input']
                    val_ground_truth = val_data['ground_truth'].cuda()
                    val_ground_truth = data.crop_image_batch(val_ground_truth, val_image_size)

                    with autocast(options['speed_up']['enable_amp'] or options['speed_up']['fast_eval']):
                        val_student_input = val_student_input.cuda()
                        student_result = self.network(val_student_input)
                        student_result = torch.clamp(student_result, 0, 1).float()
                        student_result = data.crop_image_batch(student_result, val_image_size)

                        val_teacher_input = val_teacher_input.cuda()
                        teacher_result = self.ema_network(val_teacher_input)
                        teacher_result = torch.clamp(teacher_result, 0, 1).float()
                        teacher_result = data.crop_image_batch(teacher_result, val_image_size)

                    with autocast(options['speed_up']['enable_amp']):
                        for index, file in enumerate(file_name):
                            image_count += 1
                            for metric_name, metric in self.metrics.items():
                                # iqa
                                student_metric_single_result = metric(student_result[index].unsqueeze(0), val_ground_truth[index].unsqueeze(0))
                                raw_metrics_result[metric_name] += student_metric_single_result

                                teacher_metric_single_result = metric(teacher_result[index].unsqueeze(0), val_ground_truth[index].unsqueeze(0))
                                ema_metrics_result[metric_name] += teacher_metric_single_result

                            # visualization
                            self.summery_writer.add_image("[{}] {}".format('Raw', file), student_result[index],
                                                          self.global_step,
                                                          dataformats='CHW')
                            self.summery_writer.add_image("[{}] {}".format('EMA', file), teacher_result[index],
                                                          self.global_step,
                                                          dataformats='CHW')

                for metric_name in self.metrics:
                    raw_metrics_result[metric_name] /= image_count
                    ema_metrics_result[metric_name] /= image_count

                eval_iqa_result['result'][eval_set_name]['student'] = raw_metrics_result
                eval_iqa_result['result'][eval_set_name]['teacher'] = ema_metrics_result

            # log & visualization
            self.iqa_result_log_and_visual(eval_iqa_result)

            # save best checkpoint
            self.check_and_save_best_checkpoint(epoch_index, eval_iqa_result, options)

        self.network.train()
        self.ema_network.train()

    def eval_and_save_to_dir(self, options, eval_name):
        self.network.eval()
        self.ema_network.eval()

        image_dir = os.path.join(options['path']['results'], eval_name)
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(os.path.join(image_dir, 'Student'), exist_ok=True)
        os.makedirs(os.path.join(image_dir, 'Teacher'), exist_ok=True)

        with torch.no_grad():
            eval_loader = self.val_loaders_dict[eval_name]

            val_tqdm = tqdm.tqdm(eval_loader)

            teacher_metric_detail = {}  # teacher_metric_detail[metric_name][file_name] = metric_result
            student_metric_detail = {}  # student_metric_detail[metric_name][file_name] = metric_result

            val_image_size = eval_loader.dataset.image_size

            for val_data in val_tqdm:
                file_name = val_data['file_name']
                val_student_input = val_data['student_input']
                val_teacher_input = val_data['teacher_input']
                val_ground_truth = val_data['ground_truth'].cuda()
                val_ground_truth = data.crop_image_batch(val_ground_truth, val_image_size)

                with autocast(options['speed_up']['enable_amp'] or options['speed_up']['fast_eval']):
                    val_student_input = val_student_input.cuda()
                    student_result = self.network(val_student_input)
                    # # the shameful peeking, cite from LLFlow
                    # peeking_value = torch.mean(val_ground_truth) / torch.mean(student_result)
                    # student_result = student_result * peeking_value
                    # # end of peeking
                    student_result = torch.clamp(student_result, 0, 1).float()
                    student_result = data.crop_image_batch(student_result, val_image_size)

                    val_teacher_input = val_teacher_input.cuda()
                    teacher_result = self.ema_network(val_teacher_input)
                    # # the shameful peeking, cite from LLFlow
                    # peeking_value = torch.mean(val_ground_truth) / torch.mean(teacher_result)
                    # teacher_result = teacher_result * peeking_value
                    # # end of peeking
                    teacher_result = torch.clamp(teacher_result, 0, 1).float()
                    teacher_result = data.crop_image_batch(teacher_result, val_image_size)

                with autocast(options['speed_up']['enable_amp']):
                    for index, file in enumerate(file_name):
                        student_metric_detail[file] = {}
                        teacher_metric_detail[file] = {}
                        file_name_list = os.path.basename(file).split('.')[:-1]
                        file_name = ""
                        for part in file_name_list:
                            file_name += part
                            file_name += '.'
                        for metric_name, metric in self.metrics.items():
                            # iqa
                            student_metric_single_result = metric(student_result[index], val_ground_truth[index])
                            student_metric_detail[file][metric_name] = student_metric_single_result.item()

                            teacher_metric_single_result = metric(teacher_result[index], val_ground_truth[index])
                            teacher_metric_detail[file][metric_name] = teacher_metric_single_result.item()

                        Image.fromarray(RetinolUtil.tensor2img(student_result[index], min_max=(0, 1))).save(
                            os.path.join(
                                os.path.join(image_dir, 'Student'), file_name + 'png'
                            )
                        )
                        Image.fromarray(RetinolUtil.tensor2img(teacher_result[index], min_max=(0, 1))).save(
                            os.path.join(
                                os.path.join(image_dir, 'Teacher'), file_name + 'png'
                            )
                        )

            # save metric csv
            csv_header = ['file']
            for metric in self.metrics:
                csv_header.append(metric)

            with open(os.path.join(os.path.join(image_dir, 'Student'), 'metric.csv'), 'w',
                      encoding='utf-8') as csv_file:
                dict_writer = csv.DictWriter(csv_file, csv_header)
                dict_writer.writeheader()

                for file_name, file_dict in student_metric_detail.items():
                    file_dict['file'] = file_name
                    dict_writer.writerow(file_dict)
            logger.success('Student metric csv written.')

            with open(os.path.join(os.path.join(image_dir, 'Teacher'), 'metric.csv'), 'w',
                      encoding='utf-8') as csv_file:
                dict_writer = csv.DictWriter(csv_file, csv_header)
                dict_writer.writeheader()

                for file_name, file_dict in teacher_metric_detail.items():
                    file_dict['file'] = file_name
                    dict_writer.writerow(file_dict)
            logger.success('Teacher metric csv written.')

    def eval_step_without_save(self, epoch_index, options, dataloaders_dict):
        self.network.eval()
        self.ema_network.eval()

        with torch.no_grad():
            eval_iqa_result = {
                'epoch': epoch_index,
                'step': self.global_step,
                'result': {}
            }
            for eval_set_name, eval_loader in dataloaders_dict.items():
                eval_count = len(eval_loader)
                eval_iqa_result['result'][eval_set_name] = {}

                eval_loader_pbar = tqdm.tqdm(eval_loader)
                raw_metrics_result = {}
                ema_metrics_result = {}
                for metric_name in self.metrics:
                    raw_metrics_result[metric_name] = 0
                    ema_metrics_result[metric_name] = 0

                val_image_size = eval_loader.dataset.image_size
                for val_data in eval_loader_pbar:
                    # file_name = val_data['file_name']
                    val_student_input = val_data['student_input']
                    val_teacher_input = val_data['teacher_input']
                    val_ground_truth = val_data['ground_truth']
                    val_ground_truth = data.crop_image_batch(val_ground_truth, val_image_size).cuda()

                    with autocast(options['speed_up']['enable_amp'] or options['speed_up']['fast_eval']):
                        val_student_input = val_student_input.cuda()
                        student_result = self.network(val_student_input)
                        student_result = torch.clamp(student_result, 0, 1).float()
                        student_result = data.crop_image_batch(student_result, val_image_size)

                        val_teacher_input = val_teacher_input.cuda()
                        teacher_result = self.ema_network(val_teacher_input)
                        teacher_result = torch.clamp(teacher_result, 0, 1).float()
                        teacher_result = data.crop_image_batch(teacher_result, val_image_size)

                    with autocast(options['speed_up']['enable_amp']):
                        for metric_name, metric in self.metrics.items():
                            student_metric_result = metric(student_result, val_ground_truth)
                            raw_metrics_result[metric_name] += student_metric_result

                            teacher_metric_result = metric(teacher_result, val_ground_truth)
                            ema_metrics_result[metric_name] += teacher_metric_result

                for metric_name in self.metrics:
                    raw_metrics_result[metric_name] /= eval_count
                    ema_metrics_result[metric_name] /= eval_count

                eval_iqa_result['result'][eval_set_name]['student'] = raw_metrics_result
                eval_iqa_result['result'][eval_set_name]['teacher'] = ema_metrics_result

            # log & visualization
            self.iqa_result_log_and_visual(eval_iqa_result)

            # save best checkpoint
            self.check_and_save_best_checkpoint(epoch_index, eval_iqa_result, options)

        self.network.train()
        self.ema_network.train()

    def check_and_save_best_checkpoint(self, epoch_index, eval_iqa_result, options):
        watching_metrics = options['model']['watching_metrics']

        student_metric_sum = 0
        teacher_metric_sum = 0
        # discover best
        for val_set_name, val_result_info in self.current_best_info.items():
            for metric_name, metric_result_info in val_result_info.items():
                val_set_metric_name = '{}_{}'.format(val_set_name, metric_name)
                if val_set_metric_name not in watching_metrics:
                    continue

                student_metric_sum += eval_iqa_result['result'][val_set_name]['student'][metric_name]
                teacher_metric_sum += eval_iqa_result['result'][val_set_name]['teacher'][metric_name]

                if eval_iqa_result['result'][val_set_name]['student'][metric_name] > \
                        eval_iqa_result['result'][val_set_name]['teacher'][metric_name]:
                    temp_better = eval_iqa_result['result'][val_set_name]['student'][metric_name]
                    better_network = 'student'
                else:
                    temp_better = eval_iqa_result['result'][val_set_name]['teacher'][metric_name]
                    better_network = 'teacher'

                if metric_result_info['value'] < temp_better:
                    # new best
                    logger.info('New Best {} under {} : {} from {}'.format(metric_name, val_set_name, temp_better,
                                                                           better_network))
                    self.current_best_info[val_set_name][metric_name]['value'] = float(temp_better)
                    self.current_best_info[val_set_name][metric_name]['network'] = better_network
                    self.current_best_info[val_set_name][metric_name]['epoch'] = epoch_index
                    self.current_best_info[val_set_name][metric_name]['step'] = self.global_step

                    # save network
                    self.save_everything(val_set_metric_name, epoch_index, options)

                    # save best_iqa.json
                    with open(os.path.join(options['path']['results'], 'best_iqa.json'), 'w') as best_json_file:
                        json.dump(self.current_best_info, best_json_file, indent=2)

        if student_metric_sum > teacher_metric_sum:
            temp_better = student_metric_sum
            better_network = 'student'
        else:
            temp_better = teacher_metric_sum
            better_network = 'teacher'

        if self.global_best_info['value'] < temp_better:
            self.global_best_info['value'] = float(temp_better)
            self.global_best_info['network'] = better_network
            self.global_best_info['epoch'] = epoch_index
            self.global_best_info['step'] = self.global_step

            logger.info('New Global Best : {} from {}'.format(temp_better, better_network))
            # save network
            self.save_everything('global', epoch_index, options)

    def fake_eval(self, epoch_index, options, dataloaders_dict):
        # do nothing
        # this function is designed for skipping eval when reproducing to get weights
        logger.info('Fake eval at epoch {}'.format(epoch_index))
        self.network.eval()
        self.ema_network.eval()

        with torch.no_grad():
            for eval_set_name, eval_loader in dataloaders_dict.items():
                eval_loader_pbar = tqdm.tqdm(eval_loader)
                for val_data in eval_loader_pbar:
                    pass

        self.network.train()
        self.ema_network.train()

    def save_everything(self, prefix, epoch_index, options):
        network_label = self.network.__class__.__name__
        save_network(prefix, network=self.network, network_label=network_label,
                     save_dir=options['path']['checkpoint'])
        save_network(prefix, network=self.ema_network, network_label=network_label + '_ema',
                     save_dir=options['path']['checkpoint'])

        # Train info
        train_info = {
            'epoch': epoch_index,
            'step': self.global_step,

            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler_last_epoch': self.scheduler.last_epoch,
            # 'lr_scheduler': self.scheduler.state_dict(),

            # random generator save
            'torch_rng': torch.get_rng_state(),
            'torch_cuda_rng': torch.cuda.get_rng_state(),
            'numpy_rng': np.random.get_state(),
            'python_rng': random.getstate()
        }
        torch.save(train_info, os.path.join(options['path']['checkpoint'], "{}_train_info.pth".format(prefix)))

    def load_from_disk(self, resume_state, options):
        network_label = self.network.__class__.__name__
        load_network(resume_state, network=self.network, network_label=network_label)
        load_network(resume_state, network=self.ema_network, network_label=network_label + '_ema')

        # Train info
        train_info = torch.load(resume_state + "_train_info.pth")
        self.global_step = train_info['step']
        self.start_epoch = train_info['epoch']

        self.optimizer.load_state_dict(train_info['optimizer'])
        # self.scheduler.load_state_dict(train_info['lr_scheduler'])

        # There will be an unexpected result if using code above to resume the scheduler, use last_epoch instead,
        # as mentioned in https://blog.csdn.net/qq_45860671/article/details/124324597
        # If using code above, the result of next 3-4 epoch will be the same as trained without stop/resume,
        # but will be different from the expected later.
        if options['train']['enable_warmup']:
            # logger.info("Using warmup and cosine strategy")
            warmup_epochs = options['train']['warmup_epochs']
            scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                    options['train']['n_epoch'] - warmup_epochs,
                                                                    eta_min=1e-6,
                                                                    last_epoch=train_info[
                                                                                   'lr_scheduler_last_epoch'] - 1)
            self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=warmup_epochs,
                                                    after_scheduler=scheduler_cosine)
            # self.scheduler.step()
        else:
            # logger.info("Using cosine strategy")
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                  options['train']['n_epoch'],
                                                                  eta_min=1e-6,
                                                                  last_epoch=train_info['lr_scheduler_last_epoch'] - 1)
            # self.scheduler.step()

        torch.set_rng_state(train_info['torch_rng'])
        try:
            torch.cuda.set_rng_state(train_info['torch_cuda_rng'])
        except Exception:
            logger.warning('Incorrect torch CUDA RNG')
        np.random.set_state(train_info['numpy_rng'])
        random.setstate(train_info['python_rng'])

    def init_weights(self, net, init_type='kaiming', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    torch.nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    torch.nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'kaiming_uniform':
                    torch.nn.init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    torch.nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    logger.critical(
                        'Initialization method [{}] of {} is not implemented'.format(init_type, m.__class__.__name__))
                    raise NotImplementedError(
                        'Initialization method [{}] of {} is not implemented'.format(init_type, m.__class__.__name__))
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    torch.nn.init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    torch.nn.init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)

        if init_type is None:
            logger.info('{} initialized as default'.format(net.__class__.__name__))
        else:
            net.apply(init_func)
            logger.success("{} initialized as {}".format(net.__class__.__name__, init_type))

    def get_average_csv_writer(self, options):
        exist_csv = os.path.exists(
            os.path.join(os.path.join(options['path']['results'], options['phase']), 'average.csv'))
        average_csv_file = open(os.path.join(os.path.join(options['path']['results'], options['phase']), 'average.csv'),
                                mode='a+', encoding='utf-8-sig')
        average_csv_writer = csv.writer(average_csv_file)

        if not exist_csv:
            average_csv_writer.writerow(['epoch'] + [metric_name for metric_name in self.metrics])

        return average_csv_file, average_csv_writer

    def loss_fn(self, x, y, loss_name, epoch_index, tag):
        loss_sum = 0
        for loss in self.loss_function_dict[loss_name]:
            single_loss = loss(x, y)
            loss_sum = loss_sum + single_loss * loss.weight_from_epoch(epoch_index)

            with torch.no_grad():
                # visualization
                self.summery_writer.add_scalar('[{}] {}'.format(tag, loss.__class__.__name__), single_loss.item(),
                                               self.global_step)

        with torch.no_grad():
            if not isinstance(loss_sum, type(0)):
                loss_sum_item = loss_sum.item()
                self.summery_writer.add_scalar('[{}] Total Loss'.format(tag), loss_sum_item, self.global_step)
                self.epoch_loss_recorder[loss_name].append(loss_sum_item)

        return loss_sum

    def print_loss_per_epoch(self):
        logger.info('Current Loss:')
        for loss_name, loss_value in self.epoch_loss_recorder.items():
            if len(loss_value) > 0:
                logger.info('\t{}: {}'.format(loss_name, np.mean(loss_value)))

        self.clear_epoch_loss_recorder()

    def clear_epoch_loss_recorder(self):
        for loss_name, loss_funcs in self.loss_function_dict.items():
            if len(loss_funcs) > 0:
                self.epoch_loss_recorder[loss_name] = []

    def display_train_data_in_tensorboard(self, input_x, gt_y, epoch_index):
        with torch.no_grad():
            index = random.randint(0, input_x.shape[0] - 1)
            if gt_y is None:
                self.summery_writer.add_image('Unpaired input', torch.clamp(input_x[index], 0, 1), epoch_index,
                                              dataformats='CHW')
            else:
                self.summery_writer.add_image('Paired input', torch.clamp(input_x[index], 0, 1), epoch_index,
                                              dataformats='CHW')
                self.summery_writer.add_image('Paired ground truth', torch.clamp(gt_y[index], 0, 1), epoch_index,
                                              dataformats='CHW')

    def iqa_result_log_and_visual(self, iqa_result_dict):
        logger.info('Eval result at step {}, epoch {}:'.format(iqa_result_dict['step'], iqa_result_dict['epoch']))

        csv_line = '\n{},'.format(iqa_result_dict['epoch'])
        for dataset_name, dataset_result in iqa_result_dict['result'].items():
            for network_name, network_result in dataset_result.items():
                for metric_name, metric_value in network_result.items():
                    logger.info('\t{} of {} at {}: {}'.format(metric_name, network_name, dataset_name, metric_value))

                    self.summery_writer.add_scalar('[{} @ {}] {}'.format(network_name, dataset_name, metric_name),
                                                   metric_value, iqa_result_dict['step'])
                    csv_line += str(float(metric_value)) + ','

        csv_line = csv_line[:-1]
        with open(self.iqa_csv_path, 'a+') as csv_file:
            csv_file.write(csv_line)

    def init_csv(self):
        csv_header = 'epoch,'
        two_model = ['student', 'teacher']
        for val_set_name in self.val_loaders_dict:
            for model_name in two_model:
                for metric in self.metrics:
                    csv_header += '{}_{}_{},'.format(val_set_name, model_name, metric)

        csv_header = csv_header[:-1]
        with open(self.iqa_csv_path, 'w') as csv_file:
            csv_file.write(csv_header)


def get_ema_decay(epoch, options):
    if epoch < options['train']['decay_change_epoch']:
        return options['train']['ema_decay_0']
    else:
        return options['train']['ema_decay_1']


def load_network(resume_state, network, network_label):
    if resume_state is None:
        return
    logger.info('Begin loading pretrained model [{:s}] ...'.format(network_label))

    model_path = "{}_{}.pth".format(resume_state, network_label)

    if not os.path.exists(model_path):
        logger.warning('Pretrained model in [{:s}] is not existed, Skip it'.format(model_path))
        return

    logger.info('Loading pretrained model from [{:s}] ...'.format(model_path))

    network.load_state_dict(torch.load(model_path))

    logger.success('Load pretrained model {} success.'.format(network_label))


def save_network(prefix, network, network_label, save_dir):
    """ save network structure """
    save_filename = '{}_{}.pth'.format(prefix, network_label)
    save_path = os.path.join(save_dir, save_filename)

    state_dict = network.state_dict()
    for key, param in state_dict.items():
        state_dict[key] = param.cpu()
    torch.save(state_dict, save_path)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        # ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


def get_current_consistency_weight(consistency, consistency_rampup, epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242

    return consistency * ramps.sigmoid_rampup(epoch, consistency_rampup)
