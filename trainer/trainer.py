from abc import ABC, ABCMeta, abstractmethod
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
from deeplearning.model.init import init_normal


class ITrainer(ABCMeta):
    """Interface for training models."""

    @staticmethod
    @abstractmethod
    def train(num_epochs: int) -> None:
        """This method trains a model.

        Parameters
        ----------
        num_epochs : number of epochs the model will be trained."""

    @staticmethod
    @abstractmethod
    def reset() -> None:
        """This method resets the parameters of a model and the optimizer."""


class Trainer(ITrainer):

    def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Adam,
            loader: torch.utils.data.DataLoader,
            loss_fn,
            metrics_fn,
            init_fn=init_normal) -> None:
        """
        Parameters
        ----------
        model : PyTorch neural network
        optimizer : torch.optim object
        loader : a torch dataloader
        loss_fn : the loss function
        metrics_fn : function reporting a metric
        init_fn : function that initializes the parameters of the model
        """

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.epochs_trained = 0

        # Components
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loader = loader
        self.loss_fn = loss_fn
        # self.metrics_fn = metrics_fn
        self.init_fn = init_fn

        # Hyperparameters
        self.HPARAMS = {
            "lr": self.optimizer.param_groups[0]['lr'],
            "weight_decay": self.optimizer.param_groups[0]['weight_decay'],
            "batch_size": self.loader.batch_size
        }

        self.reset()

    # def __call__(self):
    #     print(f"Model: {self.model}")
    #     print(f"Loss function: {self.loss_fn}")
    #     print(f"Optimizer: {self.optimizer}")

    def train(self, num_epochs: int) -> None:
        """Trains self.model for num_epochs epochs.

        Parameters
        ----------
        num_epochs : int
        """
        self.model.train()

        for epoch in range(num_epochs):
            print(
                f"[{self.epochs_trained + epoch + 1}/"
                f"{self.epochs_trained + num_epochs}]")

            for batch_idx, (data, targets) in enumerate(self.loader.train_loader):
                data = data.to(self.device)
                targets = targets.to(self.device)

                # Forward
                scores = self.model(data)
                loss = self.loss_fn(scores, targets)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if batch_idx == 0:
                    print(loss)

        self.epochs_trained += num_epochs

        # print(f"Validation ROC: {self.metrics_fn(actuals, predictions)}")
        # self.model.train()

    def reset(self) -> None:
        """Resets model.parameters() with self.init_fn, and
        self.optimizer."""

        self.model.apply(self.init_fn)
        self.optimizer = optim.Adam(
            params=self.model.parameters(),
            lr=self.HPARAMS["lr"],
            weight_decay=self.HPARAMS["weight_decay"]
        )
        self.epochs_trained = 0

    def set_hparams(self, hparams: dict) -> None:
        """Method to change the hyperparamter setup."""

        self.HPARAMS["lr"] = hparams.get("lr", self.HPARAMS["lr"])
        self.optimizer.param_groups[0]["lr"] = self.HPARAMS["lr"]

        self.HPARAMS["weight_decay"] = hparams.get(
            "weight_decay", self.HPARAMS["weight_decay"])
        self.optimizer.param_groups[0]["weight_deday"] = self.HPARAMS["weight_deday"]

        self.HPARAMS["batch_size"] = hparams.get(
            "batch_size", self.HPARAMS["batch_size"])
        self.loader.batch_size = self.HPARAMS["batch_size"]

    # @abstractmethod
    # def out(self):
    #     pass

    # @abstractmethod
    # def save_model(self) -> None:
    #     pass

    # @abstractmethod
    # def load_model(self, file) -> None:
    #     pass


class GANTrainer(ITrainer):

    def __init__(self, model_g, model_d, optim_g, optim_d, loader, loss_fn,
                 init_fn, z_dim, img_size) -> None:
        """Interface for training GAN models.

        Parameters
        ----------
        model_g : PyTorch generator network
        model_d : PyTorch discriminator network
        optimizer : torch.optim object
        loader : a container class for torch DataLoaders
        loss_fn : the loss function, binary crossentropy
        # metrics_fn : function reporting a metric
        init_fn : function that initializes parameters of the models
        z_dim : dimension of latent noise for generator
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.epochs_trained = 0

        # Parameters
        self.model_d = model_d.to(self.device)
        self.model_g = model_g.to(self.device)
        self.optim_d = optim_d
        self.optim_g = optim_g
        self.loader = loader
        self.loss_fn = loss_fn
        self.init_fn = init_fn
        self.z_dim = z_dim
        self.img_size = img_size

        self.lr = self.optim_d.param_groups[0]['lr']
        self.weight_decay = self.optim_d.param_groups[0]['weight_decay']

    def __call__(self):
        print(f"Generator: {self.model_g}")
        print(f"Discriminator: {self.model_d}")
        print(f"Loss function: {self.loss_fn}")
        print(f"Optimizer: {self.optim_d}")

    def train(self, num_epochs, num_critic_iterations=1) -> None:
        """Trains self.model_g for num_epochs epochs, and self.model_c
        for num_epochs * num_critic_iterations epochs.

        Parameters
        ----------
        num_epochs : number of epochs to be trained
        num_critic_iterations : number of iterations the critic will be
            trained in each epoch
        """

        print('Start training.')
        # fake_writer = SummaryWriter(log_dir=f'{TBLOGPATH}/fake')
        # real_writer = SummaryWriter(log_dir=f'{TBLOGPATH}/real')
        self.model_g.train()
        self.model_d.train()
        tb_step = 0

        for epoch in range(num_epochs):

            for batch_idx, (real_data, _) in enumerate(self.loader):
                current_batch_size = real_data.shape[0]
                x = real_data.to(device=self.device)
                # Random uniform latent noise
                z = torch.rand(current_batch_size, self.z_dim,
                               1, 1).to(self.device)
                fake_data = self.model_g(z)

                # Discriminator forward
                real_scores = self.model_d(x).reshape(-1)
                fake_scores = self.model_d(self.model_g(z)).reshape(-1)
                real_loss_D = self.loss_fn(
                    real_scores, torch.ones_like(real_scores))
                fake_loss_D = self.loss_fn(
                    fake_scores, torch.zeros_like(fake_scores))
                loss_d = (real_loss_D + fake_loss_D) / 2

                # Discriminator backward
                self.optim_d.zero_grad()
                loss_d.backward(retain_graph=True)  # retain_graph=True
                self.optim_d.step()

                # Generator forward
                fake_scores_updated = self.model_d(self.model_g(z))
                loss_g = self.loss_fn(fake_scores_updated,
                                      torch.ones_like(fake_scores_updated))

                # Generator backward
                self.optim_g.zero_grad()
                loss_g.backward()
                self.optim_g.step()

                if batch_idx == 0:
                    with torch.no_grad():
                        print(
                            f'Epoch [{epoch + 1}/{num_epochs}]'
                            f'\tLoss D: {loss_d:.6f}\tLoss G: {loss_g:.6f}')
                        fake_data = self.model_g(z).reshape(-1, 1,
                                                            self.img_size, self.img_size)
                        fake_grid = torchvision.utils.make_grid(
                            fake_data[:32], normalize=True)
                        # fake_writer.add_image(
                        #     'Fake Img', fake_grid, global_step=tb_step)
                        real_data = real_data.reshape(-1,
                                                      1, self.img_size, self.img_size)
                        real_grid = torchvision.utils.make_grid(
                            real_data[:32], normalize=True)
                        # real_writer.add_image(
                        #     'Real Img', real_grid, global_step=tb_step)
                        tb_step += 1
        print('Finished training.')

        # print('Start training.')
        # # tb_step = 0
        # self.model_c.train()
        # self.model_g.train()

        # for epoch in range(num_epochs):
        #     for batch_idx, (data, _) in enumerate(self.loader):
        #         current_bs = data.shape[0]
        #         real_img = data.to(device=self.device)

        #         # Critic will be trained more
        #         for _ in range(num_critic_iterations):
        #             z = torch.rand(current_bs, Z_DIM, 1, 1).to(
        #                 device=self.device)
        #             fake_img = self.model_g(z)
        #             scores_real = self.model_c(real_img)
        #             scores_fake = self.model_c(fake_img)
        #             loss_C = -(torch.mean(scores_real) -
        #                        torch.mean(scores_fake))
        #             self.optim_c.zero_grad()
        #             loss_C.backward(retain_graph=True)
        #             self.optim_c.step()

        #             for p in self.model_c.parameters():
        #                 p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

        #         # Generator
        #         # scores_fake_update = C(G(z)) # if we don't use retain_graph
        #         scores_fake_update = self.model_c(fake_img)
        #         loss_G = -torch.mean(scores_fake_update)

        #         self.optim_g.zero_grad()
        #         loss_G.backward()
        #         self.optim_g.step()

        #         if batch_idx == 0:
        #             with torch.no_grad():
        #                 print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}]\t'
        #                       f'Loss C: {loss_C}\tLoss G: {loss_G}')
        #                 # fake_imgs = torchvision.utils.make_grid(
        #                 #     tensor=fake_img[:32], padding=2, normalize=True)
        #                 # real_imgs = torchvision.utils.make_grid(
        #                 #     tensor=real_img[:32], padding=2, normalize=True)
        #                 # writer_fake.add_image(
        #                 #     tag='Fake', img_tensor=fake_imgs, global_step=tb_step)
        #                 # writer_real.add_image(
        #                 #     tag='Real', img_tensor=real_imgs, global_step=tb_step)
        #     # tb_step = 0
        # print('Finished training.')

        self.epochs_trained += num_epochs

    def validate(self) -> None:
        pass

    def reset(self) -> None:
        self.model_g.apply(self.init_fn)
        self.model_c.apply(self.init_fn)
        self.optim_g = optim.Adam(
            params=self.model_g.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.optim_d = optim.Adam(
            params=self.model_d.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.epochs_trained = 0


class TensorboardLogger:
    pass


class ScalarLogger(TensorboardLogger):
    pass


class ImageLogger(TensorboardLogger):
    pass
