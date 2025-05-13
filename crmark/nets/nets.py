import argparse
from crmark.compressor.utils import *


class ConvTP2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=2, stride=2, padding=0, bias=True):
        super(ConvTP2d, self).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding,
                               bias=bias),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class Conv2D(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=True):
        super(Conv2D, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding, bias=bias),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class SpatialAttention(nn.Module):
    def __init__(self, input_dim, bias):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.ave_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.conv = nn.Conv2d(input_dim * 2, input_dim, 3, 1, 1, bias=bias)
        self.sigmoid = nn.Sigmoid()

    def __call__(self, x):
        max_out = self.max_pool(x)
        avg_out = self.ave_pool(x)
        cat_out = torch.cat([max_out, avg_out], dim=1)
        conv_out = torch.sigmoid(self.conv(cat_out))
        return conv_out * x


class BaseFuncDown(nn.Module):
    def __init__(self, img_size, input_dim, bit_length, min_size=16, base_conv=8, bias=True, fc=True):
        super(BaseFuncDown, self).__init__()
        self.min_size = min_size
        self.activation = nn.LeakyReLU(inplace=True)
        conv_list = []
        self.first_conv = Conv2D(input_dim, base_conv, 3, 1, 1, bias=bias)
        conv_input_dim = base_conv
        conv_num = int(np.log2(img_size // self.min_size))
        for i in range(1, conv_num + 1):
            conv_list += [
                Conv2D(conv_input_dim, (i + 1) * base_conv, 3, 1, 1, bias=bias),
                SpatialAttention((i + 1) * base_conv, bias=bias),
                Conv2D((i + 1) * base_conv, (i + 1) * base_conv, 3, 2, 1, bias=bias),
            ]
            conv_input_dim = (i + 1) * base_conv

        if fc:
            self.feature_down = nn.Sequential(
                nn.Flatten(),
                nn.Linear(conv_input_dim * min_size * min_size, bit_length, bias=bias),
                nn.LeakyReLU(inplace=True),
            )
        else:
            sqrt_bit_length = int(np.sqrt(bit_length))
            stride = min_size // sqrt_bit_length
            kernel_size = min_size - sqrt_bit_length * stride + stride
            padding = 0

            self.feature_down = nn.Sequential(
                Conv2D((conv_num + 1) * base_conv, 1, kernel_size, stride, padding, bias=bias),
                nn.Flatten()
            )

        self.downsampleblock = nn.Sequential(*conv_list)

    def forward(self, x):
        next_out = self.first_conv(x)
        next_out = self.downsampleblock(next_out)
        out = self.feature_down(next_out)
        return out


class BaseFuncUp(nn.Module):
    def __init__(self, img_size, output_dim, bit_length, min_size=16, base_conv=8, bias=True, fc=True):
        super(BaseFuncUp, self).__init__()
        self.min_size = min_size
        self.activation = nn.LeakyReLU(inplace=True)
        upsample_list = []
        feature_dim = (int(np.log2(img_size // self.min_size)) + 1) * base_conv

        if fc:
            self.feature_up = nn.Sequential(
                nn.Linear(bit_length, feature_dim * min_size * min_size, bias=bias),
                nn.Unflatten(1, (feature_dim, min_size, min_size)),
                nn.LeakyReLU(inplace=True),
            )
        else:
            sqrt_bit_length = int(np.sqrt(bit_length))
            stride = min_size // sqrt_bit_length
            kernel_size = min_size - sqrt_bit_length * stride + stride
            padding = 0

            self.feature_up = nn.Sequential(
                nn.Unflatten(1, (1, sqrt_bit_length, sqrt_bit_length)),
                ConvTP2d(1, feature_dim, kernel_size, stride, padding, bias=bias)
            )

        upsample_num = int(np.log2(img_size // self.min_size))
        for i in range(upsample_num, 0, -1):
            upsample_list += [
                ConvTP2d(feature_dim, (i - 1 + 1) * base_conv, 4, 2, 1, bias=bias),
                SpatialAttention((i - 1 + 1) * base_conv, bias=bias),
                Conv2D((i - 1 + 1) * base_conv, (i - 1 + 1) * base_conv, 3, 1, 1, bias=bias),
            ]
            feature_dim = (i - 1 + 1) * base_conv

        self.final_conv = Conv2D(feature_dim, output_dim, 3, 1, 1, bias=bias)
        self.upsampleblock = nn.Sequential(*upsample_list)

    def forward(self, x):
        next_out = self.feature_up(x)
        next_out = self.upsampleblock(next_out)
        out = self.final_conv(next_out)
        return out


class InvertibleBlock(nn.Module):
    def __init__(self, img_size, input_dim, bit_length, min_size=16, base_conv=8, bias=True, clamp=4., fc=False):
        super().__init__()
        self.clamp = clamp
        self.qi = BaseFuncDown(img_size, input_dim, bit_length, min_size, base_conv, bias, fc)
        self.ui = BaseFuncUp(img_size, input_dim, bit_length, min_size, base_conv, bias, fc)
        self.si = BaseFuncDown(img_size, input_dim, bit_length, min_size, base_conv, bias, fc)
        self.round_img = StochasticRound(scale=1. / 255.)
        self.round_bit = StochasticRound(scale=1.)

    def sigmiod(self, s):
        return self.clamp * (torch.sigmoid(s))

    def exp(self, s):
        return torch.exp(s)

    def forward(self, x1, x2, hard_round, reverse=False):
        """
        Args:
            x1: torch.Tensor, the first part of input (transformed by phi)
            x2: torch.Tensor, the second part of input (affine transformed)
            hard_round:
            reverse: bool, if True, applies inverse transformation
        Returns:
            y1, y2: Transformed outputs
        """
        if not reverse:
            y1 = x1 + self.round_img(self.ui(x2), hard_round)
            y2 = x2 * (self.round_bit(self.exp(self.sigmiod(self.si(y1))), hard_round)) + self.round_bit(self.qi(y1),
                                                                                                         hard_round)
        else:
            y2 = (x2 - self.round_bit(self.qi(x1), hard_round)) / (
                self.round_bit(self.exp(self.sigmiod(self.si(x1))), hard_round))
            y1 = x1 - self.round_img(self.ui(y2), hard_round)
        return y1, y2


class Model(nn.Module):
    def __init__(self, img_size, channel_dim, bit_length, k, min_size=16, fc=False):
        super().__init__()
        self.k = k
        self.fc = fc
        self.min_size = min_size
        self.img_size = img_size
        self.bit_length = bit_length
        self.round = StochasticRound()
        self.channel_dim = channel_dim
        self.inn_blocks = nn.Sequential(
            *[InvertibleBlock(img_size, channel_dim, bit_length, min_size=min_size, fc=fc) for _ in range(k)])
        self.MSE = nn.MSELoss(reduce=True)
        self.PMSE = PenalityLoss()
        self.LLoss = LPIPSLoss()

    def forward(self, x1, x2, hard_round, reverse=False):
        if not reverse:
            for inn_block in self.inn_blocks:
                x1, x2 = inn_block(x1, x2, hard_round, False)
        else:
            for inn_block in self.inn_blocks[::-1]:
                x1, x2 = inn_block(x1, x2, hard_round, True)
        return x1, x2

    def train_batch(self, args, noise_layer, cover, secret, now_step):
        stego, drop_z = self.forward(cover, secret, args.hard_round, False)
        noised_stego = noise_layer(stego, cover, now_step)
        drop_z_backward = torch.randn_like(drop_z)
        recon_cover, recon_secret = self.forward(noised_stego, drop_z_backward, args.hard_round, True)
        loss_penalty = self.PMSE(stego)
        loss_stego = self.MSE(stego, cover)
        loss_lpips = self.LLoss(stego, cover)
        loss_secret = self.MSE(recon_secret, secret)
        loss = args.lambda_penalty * loss_penalty + args.lambda_stego * loss_stego + args.lambda_lpips * loss_lpips + args.lambda_secret * loss_secret
        result = {
            "train_values": {
                "train_total_loss": loss,
                "train_accuracy": extract_accuracy(recon_secret, secret),
                'train_lambda_secret': args.lambda_secret,
                "train_loss_lpips": loss_lpips.item(),
                "train_loss_stego": loss_stego.item(),
                "train_loss_secret": loss_secret.item(),
                "train_loss_penalty": loss_penalty.item(),
                "train_overflow_0": overflow_num(stego, 0),
                "train_overflow_255": overflow_num(stego, 255),
                "train_stego_psnr": compute_psnr(stego, cover)
            }
        }
        return result

    def val_batch(self, args, noise_layer, intensity, val_cover, val_secret):
        secret_shape = (args.batch_size, 1, int(args.bit_length ** 0.5), int(args.bit_length ** 0.5))
        val_stego, val_drop_z = self.forward(val_cover, val_secret, True, False)
        val_noised_stego = noise_layer.test(quantize_image(val_stego), val_cover, intensity)
        round_val_noised_stego = self.round(val_noised_stego, True)
        val_recon_cover, val_recon_secret = self.forward(round_val_noised_stego, torch.randn_like(val_drop_z), True,
                                                         True)
        result = {
            "val_values": {
                "val_accuracy": extract_accuracy(val_recon_secret, val_secret),
                "val_overflow_0": overflow_num(val_stego, 0),
                "val_overflow_255": overflow_num(val_stego, 255),
                "val_stego_psnr": compute_psnr(val_stego, val_cover)
            },
            "val_images": {
                "val_cover": val_cover,
                "val_noised_stego": quantize_image(val_noised_stego),
                "val_stego": quantize_image(val_stego),
                "val_residual": quantize_residual_image(val_stego, val_cover),
                "val_recon_cover": quantize_image(val_recon_cover),
            }
        }
        if int(args.bit_length ** 0.5) ** 2 == args.bit_length:
            result["val_images"]["val_z"] = quantize_image(val_drop_z.view(secret_shape))
            result["val_images"]["val_secret"] = quantize_image(val_secret.view(secret_shape))
            result["val_images"]["val_recon_secret"] = quantize_image(val_recon_secret.view(secret_shape))
        return result

    def load_model(self, model_path, optim_blocks=None, scheduler_blocks=None):
        try:
            if os.path.exists(model_path):
                save_dict = torch.load(model_path, weights_only=False)
                start_epoch = save_dict['param_dict']['epoch']
                global_step = save_dict['param_dict']['global_step']
                lambda_secret = save_dict['param_dict']['lambda_secret']
                model_state_dict = save_dict['model_state_dict']
                optimizer_state_dict = save_dict['optimizer_state_dict']
                scheduler_state_dict = save_dict['scheduler_state_dict']
                if optim_blocks is not None:
                    optim_blocks.load_state_dict(optimizer_state_dict)
                if optim_blocks is not None:
                    scheduler_blocks.load_state_dict(scheduler_state_dict)
                self.load_state_dict(model_state_dict)
                print(f"encoder_net pretrained model parameters loaded: {model_path}")
                return start_epoch, global_step, lambda_secret
            else:
                print("encoder_net pretrained model parameters not found.")
        except FileNotFoundError:
            print("No pretrained model parameters found.")

    def save_model(self, args, optim_blocks=None, scheduler_blocks=None, now_global_step=0, epoch=0):

        # Create the save directory if it does not exist
        model_dir = f"{args.checkpoint_path}/{args.train_name}"
        os.makedirs(model_dir, exist_ok=True)

        save_dict = {
            'model_state_dict': self.state_dict(),
            "param_dict": {'epoch': epoch,
                           'global_step': now_global_step,
                           'lambda_secret': args.lambda_secret,
                           'bit_length': self.bit_length,
                           'img_size': self.img_size,
                           'channel_dim': self.channel_dim,
                           'min_size': self.img_size,
                           'fc': self.fc,
                           'k': self.k
                           }
        }
        if optim_blocks is not None:
            save_dict['optimizer_state_dict'] = optim_blocks.state_dict()
        if scheduler_blocks is not None:
            save_dict['scheduler_state_dict'] = scheduler_blocks.state_dict()

        torch.save(save_dict, f"{model_dir}/model_{epoch}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=5)
    parser.add_argument("--device", type=str, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument('--im_size', type=int, default=400)
    parser.add_argument('--hard_round', type=bool, default=False)
    parser.add_argument('--no_img_loss_step', type=int, default=64)
    parser.add_argument('--bit_length', type=int, default=100)
    parser.add_argument('--min_size', type=int, default=25)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--channel_dim', type=int, default=1)
    parser.add_argument('--seed', type=int, default=99)
    args = parser.parse_args()
    # Set the device based on gpu_id
    if torch.cuda.is_available():
        args.device = torch.device(f"cuda:{args.gpu_id}")
    else:
        args.device = torch.device("cpu")
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float64)
    model = Model(img_size=args.im_size, channel_dim=args.channel_dim, bit_length=args.bit_length, k=args.k,
                  min_size=args.min_size).to(args.device)

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    cover = torch.randint(0, 256, size=(args.batch_size, args.channel_dim, args.im_size, args.im_size)).to(
        args.device) / 255.
    secret = torch.randint(0, 2, size=(args.batch_size, args.bit_length)).to(args.device) / 1.
    from thop import profile

    flops, params = profile(model, inputs=(cover, secret, True, False))
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
    print(f"Parameters: {params / 1e6:.2f} M")
    x1, x2 = model.forward(cover, secret, True, False)
    print(x2)
    x1 = torch.round(x1 * 255.) / 255.
    x2 = torch.round(x2)
    cover_, secret_ = model.forward(x1, x2, True, True)
    print(torch.mean(torch.round(cover_ * 255) - torch.round(cover * 255)),
          torch.mean(torch.round(secret_) - torch.round(secret)))
