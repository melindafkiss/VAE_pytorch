import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os, sys
import gc

import torch
import torch.optim

import PIL

import neptune
import gin
import gin.torch
from absl import flags, app

import trainers
import models

from torchvision import datasets, transforms
import torchvision.utils as vutils

import utils

from datasets import DatasetWithIndices

from scipy.stats import norm
from scipy.stats import multivariate_normal
import math

@gin.configurable('ExperimentRunner')
class ExperimentRunner():
    def __init__(self, seed=1, no_cuda=False, num_workers=2,
                 epochs=None, log_interval=100, plot_interval=100,
                 outdir='out', datadir='~/datasets', batch_size=200,
                 num_iterations= None, prefix='', dataset='mnist',
                 ae_model_class=gin.REQUIRED, f_model = gin.REQUIRED,
                 learn_func = False, baseline = True, limit_train_size=None,
                 trail_label_idx=0, input_normalize_sym = False):
        self.seed = seed
        self.no_cuda = no_cuda
        self.num_workers = num_workers
        if epochs and num_iterations:
            assert False, 'Please only specify either epochs or iterations.'
        elif epochs == None and num_iterations == None:
            assert False, 'Please specify epochs or iterations.'
        else:
            self.epochs = epochs
            self.num_iterations = num_iterations
        self.log_interval = log_interval
        self.plot_interval = plot_interval
        self.outdir = outdir
        self.datadir = datadir
        self.batch_size = batch_size
        self.prefix = prefix
        self.dataset = dataset
        self.ae_model_class = ae_model_class
        self.f_model = f_model
        self.learn_func = learn_func
        self.baseline = baseline
        self.limit_train_size = limit_train_size
        self.trail_label_idx = trail_label_idx
        self.input_normalize_sym = input_normalize_sym
        
        self.setup_environment()
        self.setup_torch()

    def setup_environment(self):
        self.imagesdir = os.path.join(self.outdir, self.prefix, 'images')
        self.chkptdir = os.path.join(self.outdir, self.prefix, 'models')
        os.makedirs(self.datadir, exist_ok=True)
        os.makedirs(self.imagesdir, exist_ok=True)
        os.makedirs(self.chkptdir, exist_ok=True)

    def setup_torch(self):
        use_cuda = not self.no_cuda and torch.cuda.is_available()

        torch.manual_seed(self.seed)
        if use_cuda:
            torch.cuda.manual_seed(self.seed)

        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.dataloader_kwargs = {'num_workers': 3, 'pin_memory': True} if use_cuda else {'num_workers': self.num_workers, 'pin_memory': False}
        print(self.device)

    def setup_trainers(self):
        if self.dataset in ('mnist', 'fashionmnist', 'kmnist'):
            input_dims = (28, 28, 1)
            nc = 1
        else:
            input_dims = (64, 64, 3)
            nc = 3

        self.model = self.ae_model_class(nc=nc, input_dims=input_dims, input_normalize_sym=self.input_normalize_sym)
        self.model.to(self.device)

        self.trainer = trainers.VAETrainer(self.model, self.f_model, self.learn_func, self.baseline, self.device, batch_size = self.batch_size, train_loader=self.train_loader, test_loader=self.test_loader, trail_label_idx = self.trail_label_idx)

    
    def setup_data_loaders(self):
        if self.dataset == 'celeba':
            transform_list = [transforms.CenterCrop(140), transforms.Resize((64,64),PIL.Image.ANTIALIAS), transforms.ToTensor()]
            #if self.input_normalize_sym:
                #D = 64*64*3
                #transform_list.append(transforms.LinearTransformation(2*torch.eye(D), -.5*torch.ones(D)))
            transform = transforms.Compose(transform_list)
            train_dataset = datasets.CelebA(self.datadir, split='train', target_type='attr', download=True, transform=transform)
            test_dataset = datasets.CelebA(self.datadir, split='test', target_type='attr', download=True, transform=transform)
            self.nlabels = 0
        elif self.dataset == 'mnist':
            train_dataset = datasets.MNIST(self.datadir, train=True, target_transform=None, download=True, transform=transforms.Compose([transforms.ToTensor()]))
            test_dataset = datasets.MNIST(self.datadir, train=False, target_transform=None, download=True, transform=transforms.Compose([transforms.ToTensor()]))
            self.nlabels = 10
        elif self.dataset == 'fashionmnist':
            train_dataset = datasets.FashionMNIST(self.datadir, train=True, target_transform=None, download=True, transform=transforms.Compose([transforms.ToTensor()]))
            test_dataset = datasets.FashionMNIST(self.datadir, train=False, target_transform=None, download=True, transform=transforms.Compose([transforms.ToTensor()]))
            self.nlabels = 10
        elif self.dataset == 'kmnist':
            train_dataset = datasets.KMNIST(self.datadir, train=True, target_transform=None, download=True, transform=transforms.Compose([transforms.ToTensor()]))
            test_dataset = datasets.KMNIST(self.datadir, train=False, target_transform=None, download=True, transform=transforms.Compose([transforms.ToTensor()]))
            self.nlabels = 10
        else:
            raise Exception("Dataset not found: " + dataset)

        if self.limit_train_size is not None:
            train_dataset = torch.utils.data.random_split(train_dataset, [self.limit_train_size, len(train_dataset)-self.limit_train_size])[0]

        self.train_loader = torch.utils.data.DataLoader(DatasetWithIndices(train_dataset, self.input_normalize_sym), batch_size=self.batch_size, shuffle=True, **self.dataloader_kwargs)
        self.test_loader = torch.utils.data.DataLoader(DatasetWithIndices(test_dataset, self.input_normalize_sym), batch_size=self.batch_size, shuffle=False, **self.dataloader_kwargs)

    def train(self): 
        self.setup_data_loaders()
        self.setup_trainers()

        self.global_iters = 0

        #epochs now configurable through num_iterations as well (for variable batch_size grid search)
        if self.epochs == None:
            iterations_per_epoch = int(len(self.train_loader.dataset) / self.batch_size)
            self.epochs = int(self.num_iterations/iterations_per_epoch)
        for self.epoch in range(self.epochs):
            for batch_idx, (x, y, idx) in enumerate(self.train_loader, start=0):
                print(self.epoch, batch_idx, self.global_iters, len(x), len(self.train_loader))
                self.global_iters += 1
                batch = self.trainer.train_on_batch(x)

                if self.baseline:
                                        
                    if self.global_iters % self.log_interval == 0:                        
                        print("Global iter: {}, Train epoch: {}, batch: {}/{}, loss: {}, reg_loss: {}, rec_loss:{}".format(self.global_iters, self.epoch, batch_idx+1, len(self.train_loader), batch['loss'], batch['reg_loss'], batch['rec_loss']))
                        
                        neptune.send_metric('train_loss', x=self.global_iters, y=batch['loss'])
                        neptune.send_metric('train_kl_loss', x=self.global_iters, y=batch['reg_loss'])
                        neptune.send_metric('train_rec_loss', x=self.global_iters, y=batch['rec_loss'])

                    if self.global_iters % self.plot_interval == 0:
                        self.test()

                else:
                    for self.learn_func in {False, True}:
                        if self.global_iters % self.log_interval == 0:                        
                            print("Global iter: {}, Train epoch: {}, batch: {}/{}, loss: {}".format(self.global_iters, self.epoch, batch_idx+1, len(self.train_loader), batch['loss']))
                        
                        neptune.send_metric('train_loss', x=self.global_iters, y=batch['loss'])

                        if self.global_iters % self.plot_interval == 0:
                            self.test()


                
    def plot_images(self, x, train_rec, test_rec, gen):
        with torch.no_grad():
            plot_x, plot_train, plot_test, plot_gen = x, train_rec, test_rec, gen

            if self.input_normalize_sym:
                x_range = (-1., 1.)
            else:
                x_range = (0., 1.)
                
            utils.save_image(plot_x, 'test_samples', self.global_iters, '{}/test_samples_epoch_{}.png'.format(self.imagesdir, self.epoch + 1), x_range = x_range)
            utils.save_image(plot_train, 'train_reconstructions', self.global_iters, '{}/train_reconstructions_epoch_{}.png'.format(self.imagesdir, self.epoch + 1), normalize=True, x_range = x_range)
            utils.save_image(plot_test, 'test_reconstructions', self.global_iters, '{}/test_reconstructions_epoch_{}.png'.format(self.imagesdir, self.epoch + 1), normalize=True, x_range = x_range)
            utils.save_image(plot_gen, 'generated', self.global_iters, '{}/generated_epoch_{}.png'.format(self.imagesdir, self.epoch + 1), normalize=True, x_range = x_range)


    def test(self):
        if self.baseline:
            test_loss, test_reg_loss, test_rec_loss = 0.0, 0.0, 0.0
        
            with torch.no_grad():
                for _, (x_test, y_test, idx) in enumerate(self.test_loader, start=0):                
                    test_evals = self.trainer.rec_loss_on_test(x_test)
                    test_rec_loss += test_evals['rec_loss'].item()

                    test_rec_loss /= len(self.test_loader)
                    test_reg_loss = self.trainer.reg_loss_on_test().item()
                    test_loss = test_rec_loss + test_reg_loss

            #with open('ratio_{}_{}.txt'.format(self.trainer.trainer_type, self.trainer.reg_lambda), 'a') as file:
            #    file.write(str(ratio) + '\n')

            print('Test Epoch: {} ({:.2f}%)\tLoss: {:.6f}'.format(self.epoch + 1, float(self.epoch + 1) / (self.epochs) * 100., test_loss))
            #neptune.send_metric('test_loss', x=self.global_iters, y=test_loss)
            #neptune.send_metric('test_reg_loss', x=self.global_iters, y=test_reg_loss)
            #neptune.send_metric('test_rec_loss', x=self.global_iters, y=test_rec_loss)
            #neptune.send_metric('test_covered_area', x=self.global_iters, y=covered)

            with torch.no_grad():
                _, (x, _, _) = enumerate(self.test_loader, start=0).__next__()
                test_reconstruct = self.trainer.reconstruct(x)

                _, (x, _, _) = enumerate(self.train_loader, start=0).__next__()
                train_reconstruct = self.trainer.reconstruct(x)
                gen_batch = self.trainer.decode_batch(self.trainer.sample_pz(n=self.batch_size))
                self.plot_images(x, train_reconstruct, test_reconstruct, gen_batch['decode'])
        else:
            with torch.no_grad():
                _, (x, _, _) = enumerate(self.test_loader, start=0).__next__()
                test_reconstruct = self.trainer.reconstruct(x)

                _, (x, _, _) = enumerate(self.train_loader, start=0).__next__()
                train_reconstruct = self.trainer.reconstruct(x)
                gen_batch = self.trainer.decode_batch(self.trainer.sample_pz(n=self.batch_size))
                self.plot_images(x, train_reconstruct, test_reconstruct, gen_batch['decode'])


def main(argv):
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param, skip_unknown=True)
    op_config_str = gin.config._CONFIG

    use_neptune = "NEPTUNE_API_TOKEN" in os.environ
    if use_neptune:

        params = utils.get_gin_params_as_dict(gin.config._CONFIG)
        neptune.init(project_qualified_name="melindafkiss/sandbox")

        exp = neptune.create_experiment(params=params, name="exp")
        #ONLY WORKS FOR ONE GIN-CONFIG FILE
        with open(FLAGS.gin_file[0]) as ginf:
            param = ginf.readline()
            while param:
                param = param.replace('.','-').replace('=','-').replace(' ','').replace('\'','').replace('\n','').replace('@','')
                #neptune.append_tag(param)
                param = ginf.readline()
        #for tag in opts['tags'].split(','):
        #  neptune.append_tag(tag)
    else:
        neptune.init('shared/onboarding', api_token='ANONYMOUS', backend=neptune.OfflineBackend())

    er = ExperimentRunner(prefix=exp.id)
    er.train()

    params = utils.get_gin_params_as_dict(gin.config._OPERATIVE_CONFIG)
    for k, v in params.items():
        neptune.set_property(k, v)
    neptune.stop()
    print('fin')

if __name__ == '__main__':
    flags.DEFINE_multi_string('gin_file', None, 'List of paths to the config files.')
    flags.DEFINE_multi_string('gin_param', None, 'Newline separated list of Gin parameter bindings.')
    FLAGS = flags.FLAGS
    app.run(main)
