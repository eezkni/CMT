from loguru import logger
from utils.util import init_obj



def create_model(options, paired_loader, unpaired_loader, val_loaders_dict):
    """ create model """

    # set metrics and loss
    metrics = {str(item_opt): create_metric(item_opt) for item_opt in options['model']['which_metrics']}

    losses = {}
    for loss_name, loss_content in options['model']['which_losses'].items():
        losses[loss_name] = [create_loss(item_opt) for item_opt in loss_content]
    # losses = [create_loss(item_opt) for item_opt in options['model']['which_losses']]

    # set network
    # create two same network, [0] as student model and [1] as teacher model
    if len(options['model']['which_networks']) == 1:
        networks = [create_network(options, options['model']['which_networks'][0], ema=False),
                    create_network(options, options['model']['which_networks'][0], ema=True)]
    else:
        networks = [create_network(options, options['model']['which_networks'][0], ema=False),
                    create_network(options, options['model']['which_networks'][1], ema=False)]

    network_param_size = (sum(param.numel() for param in networks[0].parameters())) / 1e6
    logger.info("{} param {:.2f} M".format(networks[0].__class__.__name__, network_param_size))

    # set model
    model_opt = options['model']['which_model']
    model_opt['args'].update(
        {'networks': networks,
         'paired_loader': paired_loader,
         'unpaired_loader': unpaired_loader,
         'val_loaders_dict': val_loaders_dict,
         'losses': losses,
         'metrics': metrics,
         'visualize_ena': options['visual'],
         'resume_state': options['path']['resume_state'],
         'tensorboard_log_dir': options['path']['tensorboard_log_dir'],
         'options': options
         }
    )

    model = init_obj(model_opt, default_file_name='models.model', init_type='Model')

    return model


def create_metric(metric_opt):
    return init_obj(metric_opt, default_file_name='models.metric', init_type='Metric')


def create_loss(loss_opt):
    return init_obj(loss_opt, default_file_name='models.loss', init_type='Loss').to('cuda').eval()


def create_network(options, network_opt, ema):
    """ define network with weights initialization """
    net = init_obj(network_opt, default_file_name='models.network', init_type='Network')
    #
    # if options['phase'] == 'train':
    #     logger.info('Network [{}] weights initialize using [{:s}] method.'.format(net.__class__.__name__,
    #                                                                               network_opt['args'].get('init_method',
    #                                                                                                       'default')))
    #     # net.init_weights()

    if ema:
        # the parameters of ema network never need to calculate the grad during the training
        for param in net.parameters():
            param.detach_()
    return net

def create_optimizer(optimizer_opt):
    return init_obj(optimizer_opt, default_file_name='torch.optim', init_type='Adam')
