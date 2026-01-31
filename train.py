import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.util import init_ddp, cleanup_ddp


if __name__ == "__main__":
    opt = TrainOptions().parse()  # 自定义train的参数类，获取命令行参数
    opt.device = init_ddp()
    # dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    # dataset_size = len(dataset)  # get the number of images in the dataset.
    dataloader = create_dataset(opt) # 改个名字更清晰
    dataset_size = len(dataloader.dataset) # 获取原始数据集的大小
    print(f"The number of training images = {dataset_size}")

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    
    # Print mixed precision info if enabled
    if hasattr(opt, 'use_amp') and opt.use_amp:
        print(f"Mixed precision training enabled with dtype: {opt.amp_dtype}")
    
    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    total_iters = 0  # the total number of training iterations

    # warmup_start_epoch = opt.n_epochs
    # warmup_end_epoch = opt.n_epochs + opt.n_epochs_decay//2
    # target_lambda = opt.lambda_label
    # base_lambda = opt.lambda_label/2.0


    # for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):

    #     if epoch < warmup_start_epoch:
    #         current_label_lambda = 0.0
    #     elif warmup_start_epoch <= epoch <= warmup_end_epoch:
    #         progress = (epoch - warmup_start_epoch) / (warmup_end_epoch - warmup_start_epoch)
    #         current_label_lambda = (target_lambda-base_lambda) * progress + base_lambda
    #     else:
    #         current_label_lambda = target_lambda


    #     model.opt.lambda_label = current_label_lambda
    #     # print(f"Epoch {epoch}: Setting lambda_label to {current_label_lambda:.4f}")

    #     epoch_start_time = time.time()  # timer for entire epoch
    #     iter_data_time = time.time()  # timer for data loading per iteration
    #     epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
    #     visualizer.reset()
    #     # Set epoch for DistributedSampler
    #     if hasattr(dataset, "set_epoch"):
    #         dataset.set_epoch(epoch)

    #     for i, data in enumerate(dataset):  # inner loop within one epoch
    #         iter_start_time = time.time()  # timer for computation per iteration
    #         if total_iters % opt.print_freq == 0:
    #             t_data = iter_start_time - iter_data_time

    #         total_iters += opt.batch_size
    #         epoch_iter += opt.batch_size
    #         model.set_input(data)  # unpack data from dataset and apply preprocessing
    #         model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

    #         if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
    #             save_result = total_iters % opt.update_html_freq == 0
    #             model.compute_visuals()
    #             visualizer.display_current_results(model.get_current_visuals(), epoch, total_iters, save_result)

    #         if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
    #             losses = model.get_current_losses()
    #             t_comp = (time.time() - iter_start_time) / opt.batch_size
    #             visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
    #             visualizer.plot_current_losses(total_iters, losses)

    #         if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
    #             print(f"saving the latest model (epoch {epoch}, total_iters {total_iters})")
    #             save_suffix = f"iter_{total_iters}" if opt.save_by_iter else "latest"
    #             model.save_networks(save_suffix)

    #         iter_data_time = time.time()

    #     model.update_learning_rate()  # update learning rates at the end of every epoch

    #     if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
    #         print(f"saving the model at the end of epoch {epoch}, iters {total_iters}")
    #         model.save_networks("latest")
    #         model.save_networks(epoch)

    #     print(f"End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay} \t Time Taken: {time.time() - epoch_start_time:.0f} sec")


    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        model.opt.lambda_label = opt.lambda_label
        
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()
        # Set epoch for DistributedSampler
        if hasattr(dataloader, "set_epoch"):
            dataloader.set_epoch(epoch)

        for i, data in enumerate(dataloader):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += 1 # opt.batch_size
            epoch_iter += 1 # opt.batch_size
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, total_iters, save_result)

            if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                visualizer.plot_current_losses(total_iters, losses)

            if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                print(f"saving the latest model (epoch {epoch}, total_iters {total_iters})")
                save_suffix = f"iter_{total_iters}" if opt.save_by_iter else "latest"
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        model.update_learning_rate()  # update learning rates at the end of every epoch

        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print(f"saving the model at the end of epoch {epoch}, iters {total_iters}")
            model.save_networks("latest")
            model.save_networks(epoch)

        print(f"End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay} \t Time Taken: {time.time() - epoch_start_time:.0f} sec")


    cleanup_ddp()