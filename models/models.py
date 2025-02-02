from .conditional_gan_model import ConditionalGAN
import pdb

def create_model(opt):
    model = None
    if opt.model == 'test':
        assert (opt.dataset_mode == 'single')
        from .test_model import TestModel
        model = TestModel(opt)
    else:
        model = ConditionalGAN(opt)
    # model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    # pdb.set_trace()
    # (Pdb) pp opt
    # Namespace(batchSize=1, beta1=0.5, checkpoints_dir='./checkpoints', 
    # 	continue_train=False, dataroot='D:\\Photos\\TrainingData\\BlurredSharp\\combined', 
    # 	dataset_mode='aligned', display_freq=100, display_id=1, display_port=8097, 
    # 	display_single_pane_ncols=0, display_winsize=256, epoch_count=1, fineSize=256, 
    # 	gan_type='gan', gpu_ids=[0], identity=0.0, input_nc=3, isTrain=True, lambda_A=100.0, 
    # 	lambda_B=10.0, learn_residual=True, loadSizeX=640, loadSizeY=360, lr=0.0001, 
    # 	max_dataset_size=inf, model='content_gan', nThreads=2, n_layers_D=3, name='experiment_name', 
    # 	ndf=64, ngf=64, niter=150, niter_decay=150, no_dropout=False, no_flip=False, no_html=False, 
    # 	norm='instance', output_nc=3, phase='train', pool_size=50, print_freq=20, resize_or_crop='crop',
    # 	 save_epoch_freq=5, save_latest_freq=100, serial_batches=False, which_direction='AtoB',
    # 	  which_epoch='latest', which_model_netD='basic', which_model_netG='resnet_9blocks')

    return model
