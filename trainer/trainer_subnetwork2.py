import numpy as np
import torch
from torchvision.utils import make_grid

from base.base_trainer import BaseTrainer


class DefaultTrainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, config, model, loss, metrics, optimizer, lr_scheduler, resume, data_loader,
                 valid_data_loader=None, train_logger=None, **extra_args):
        super(DefaultTrainer, self).__init__(config, model, loss, metrics, optimizer, lr_scheduler, resume,
                                             train_logger)

        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.log_step = int(np.sqrt(data_loader.batch_size))

    def _eval_metrics(self, p_pred, p, theta_pred, theta):
        acc_metrics = np.zeros(len(self.metrics) * 2)
        for i, metric in enumerate(self.metrics):
            acc_metrics[i * 2] += metric(p_pred, p)
            self.writer.add_scalar('{}_p'.format(metric.__name__), acc_metrics[i * 2])
            acc_metrics[i * 2 + 1] += metric(theta_pred, theta)
            self.writer.add_scalar('{}_theta'.format(metric.__name__), acc_metrics[i * 2 + 1])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        # set the model to train mode
        self.model.train()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics) * 2)

        # start training
        for batch_idx, sample in enumerate(self.data_loader):
            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)

            # get data and send them to GPU
            # (N, 4*C, H, W) GPU tensor
            I_cat = sample['I_cat'].to(self.device)
            # (N, C, H, W) GPU tensor
            p_hat = sample['p_hat'].to(self.device)
            theta_hat = sample['theta_hat'].to(self.device)
            p_and_theta_weight = sample['p_and_theta_weight'].to(self.device)

            # (N, C, H, W) GPU tensor
            p = sample['p'].to(self.device)
            theta = sample['theta'].to(self.device)

            # get network output
            # (N, C, H, W) GPU tensor
            p_pred, theta_pred = self.model(I_cat, p_hat, theta_hat, p_and_theta_weight)

            # visualization
            with torch.no_grad():
                if batch_idx % 100 == 0:
                    # save images to tensorboardX
                    self.writer.add_image('p_hat', make_grid(p_hat))
                    self.writer.add_image('theta_hat', make_grid(theta_hat))
                    self.writer.add_image('p_and_theta_weight', make_grid(p_and_theta_weight))

                    self.writer.add_image('p_pred', make_grid(p_pred))
                    self.writer.add_image('theta_pred', make_grid(theta_pred))

                    self.writer.add_image('p', make_grid(p))
                    self.writer.add_image('theta', make_grid(theta))

            # train model
            self.optimizer.zero_grad()
            model_loss = self.loss(I_cat, p_pred, p, theta_pred, theta)
            model_loss.backward()
            self.optimizer.step()

            # calculate total loss/metrics and add scalar to tensorboard
            self.writer.add_scalar('loss', model_loss.item())
            total_loss += model_loss.item()
            total_metrics += self._eval_metrics(p_pred, p, theta_pred, theta)

            # show current training step info
            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] loss: {:.6f}'.format(
                        epoch,
                        batch_idx * self.data_loader.batch_size,
                        self.data_loader.n_samples,
                        100.0 * batch_idx / len(self.data_loader),
                        model_loss.item(),  # it's a tensor, so we call .item() method
                    )
                )

        # turn the learning rate
        self.lr_scheduler.step()

        # get batch average loss/metrics as log and do validation
        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        # set the model to validation mode
        self.model.eval()

        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics) * 2)

        # start validating
        with torch.no_grad():
            for batch_idx, sample in enumerate(self.valid_data_loader):
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')

                # get data and send them to GPU
                # (N, 4*C, H, W) GPU tensor
                I_cat = sample['I_cat'].to(self.device)
                # (N, C, H, W) GPU tensor
                p_hat = sample['p_hat'].to(self.device)
                theta_hat = sample['theta_hat'].to(self.device)
                p_and_theta_weight = sample['p_and_theta_weight'].to(self.device)

                # (N, C, H, W) GPU tensor
                p = sample['p'].to(self.device)
                theta = sample['theta'].to(self.device)

                # get network output
                # (N, C, H, W) GPU tensor
                p_pred, theta_pred = self.model(I_cat, p_hat, theta_hat, p_and_theta_weight)

                loss = self.loss(I_cat, p_pred, p, theta_pred, theta)

                # calculate total loss/metrics and add scalar to tensorboardX
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(p_pred, p, theta_pred, theta)

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }
