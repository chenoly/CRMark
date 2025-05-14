import warnings
import argparse
from tqdm import tqdm
from torch.optim import AdamW
from crmark.nets import Model
from dataloader import HideImage
from prettytable import PrettyTable
from crmark.compressor.utils import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from crmark.compressor.utils import find_latest_model
from watermarklab.noiselayers.noiselayerloader import DigitalDistortion

warnings.filterwarnings("ignore")


def train_batch(model, args, noise_layer, cover, secret, now_step):
    stego, drop_z = model.forward(cover, secret, args.hard_round, False)
    noised_stego = noise_layer(stego, cover, now_step)
    drop_z_backward = torch.randn_like(drop_z)
    recon_cover, recon_secret = model.forward(noised_stego, drop_z_backward, args.hard_round, True)
    loss_penalty = model.PMSE(stego)
    loss_stego = model.MSE(stego, cover)
    loss_lpips = model.LLoss(stego, cover)
    loss_secret = model.MSE(recon_secret, secret)
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


def val_batch(model, args, noise_layer, intensity, val_cover, val_secret):
    secret_shape = (args.batch_size, 1, int(args.bit_length ** 0.5), int(args.bit_length ** 0.5))
    val_stego, val_drop_z = model.forward(val_cover, val_secret, True, False)
    val_noised_stego = noise_layer.test(quantize_image(val_stego), val_cover, intensity)
    round_val_noised_stego = model.round(val_noised_stego, True)
    val_recon_cover, val_recon_secret = model.forward(round_val_noised_stego, torch.randn_like(val_drop_z), True, True)
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


def train(args):
    torch.manual_seed(args.seed)
    # torch.set_default_tensor_type(torch.DoubleTensor)
    args_dict = vars(args)
    table = PrettyTable(["Argument", "Value"])
    for arg, value in args_dict.items():
        table.add_row([arg, value])
    # torch.set_default_dtype(torch.float64)
    print(table)
    # logs
    log_path = os.path.join(args.logs_path, args.train_name)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(f"{args.checkpoint_path}/{args.train_name}", exist_ok=True)
    writer = SummaryWriter(log_path)
    # create model
    model = Model(img_size=args.im_size, channel_dim=args.channel_dim, bit_length=args.bit_length, k=args.k,
                  min_size=args.min_size, fc=args.fc).to(args.device)

    # noiselayer
    train_noiselayer = DigitalDistortion(noise_dict=args.train_noise_dict, max_step=args.max_step)
    test_noiselayer = DigitalDistortion(noise_dict=args.test_noise_dict, max_step=args.max_step)
    # datasets
    train_dataset = HideImage(args.dataset_path, args.im_size, args.bit_length, args.channel_dim)
    val_dataset = HideImage(args.val_dataset_path, args.im_size, args.bit_length, args.channel_dim)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)

    # optimizer
    optim_blocks = AdamW(model.inn_blocks.parameters(), lr=args.lr, betas=(0.5, 0.999), eps=1e-6, weight_decay=1e-5)
    scheduler_blocks = torch.optim.lr_scheduler.StepLR(optim_blocks, 200, gamma=0.5)

    if args.continue_train:
        model_path = find_latest_model(f"{args.checkpoint_path}/{args.train_name}")
        load_params = model.load_model(model_path, optim_blocks, scheduler_blocks)
        start_epoch, global_step, lambda_secret = load_params
        args.lambda_secret = lambda_secret
    else:
        start_epoch = 0
        global_step = 0

    model.train()
    inter_result = []
    average_acc_for_down_list = []
    for epoch in tqdm(range(args.num_epoch), position=0, desc="Epoch", ncols=100):
        now_epoch = epoch + start_epoch + 1
        if args.lambda_secret < 5.:
            args.v = 0.95
            args.delta = 0.005

        if len(average_acc_for_down_list) >= args.queue_len:
            if np.mean(average_acc_for_down_list) > 1. - args.delta:
                args.lambda_secret = args.lambda_secret * args.v
                average_acc_for_down_list.clear()

        loss_list = []
        acc_epoch_list = []
        tqdm_epoch = tqdm(iter(train_loader), position=1, desc=f"Iteration", ncols=140)
        for cover, secret in tqdm_epoch:
            cover = cover.to(args.device)
            secret = secret.to(args.device)
            result = train_batch(model, args, train_noiselayer, cover, secret, now_epoch)
            total_loss = result["train_values"]["train_total_loss"]
            acc_epoch_list.append(result["train_values"]["train_accuracy"])
            optim_blocks.zero_grad()
            total_loss.backward()
            optim_blocks.step()
            inter_result.append([result, global_step + 1])
            global_step += 1
            loss_list.append(total_loss.item())
            tqdm_epoch.set_description(
                f"Epoch: {now_epoch}/{args.num_epoch}, Ave acc: {np.mean(acc_epoch_list):.5f}, lambda_secret: {args.lambda_secret}, Ave Loss: {np.mean(loss_list):.5f}")

        average_acc_for_down_list.append(np.mean(acc_epoch_list))
        if len(average_acc_for_down_list) > args.queue_len:
            average_acc_for_down_list.pop(0)

        if (now_epoch - 1) % args.val_save_epoch == 0:
            # Initialize lists to store metrics for each batch
            ave_acc_list = []
            stego_psnr_list = []
            overflow_0_list = []
            overflow_255_list = []

            model.eval()
            with torch.no_grad():
                acc_dict = {}
                for key in test_noiselayer.noise_dict.keys():
                    accuracy_list = []
                    val_tqdm_epoch = tqdm(iter(val_loader), position=1, desc=f"Val (noise model: {key})", ncols=100)
                    for val_cover, val_secret in val_tqdm_epoch:
                        val_cover = val_cover.to(args.device)
                        val_secret = val_secret.to(args.device)
                        # Call the val_batch function for each noise layer and intensity
                        val_result = val_batch(model, args, test_noiselayer.noise_layers[key],
                                                     test_noiselayer.noise_dict[key], val_cover, val_secret)

                        # Append values to the lists
                        accuracy_list.append(val_result["val_values"]["val_accuracy"])
                        ave_acc_list.append(val_result["val_values"]["val_accuracy"])
                        stego_psnr_list.append(val_result["val_values"]["val_stego_psnr"])
                        overflow_0_list.append(val_result["val_values"]["val_overflow_0"])
                        overflow_255_list.append(val_result["val_values"]["val_overflow_255"])

                        # Save the last batch's images
                        last_batch_images = val_result["val_images"]

                        tqdm_epoch.set_description(f"Val: (noise mode: {key}), Accuracy: {np.mean(accuracy_list)}")

                    acc_dict.update({key: np.mean(accuracy_list)})
            model.train()
            # Create final result dictionary
            val_result = {
                "val_values": {
                    'train_loss': np.mean(loss_list),
                    "val_accuracy": np.mean(ave_acc_list),
                    "val_stego_psnr": np.mean(stego_psnr_list),
                    "val_overflow_0": np.mean(overflow_0_list),
                    "val_overflow_255": np.mean(overflow_255_list),
                },
                "val_images": last_batch_images,  # Only the last batch's images
                "val_accuracy": acc_dict
            }
            for one_res in inter_result:
                result, iter_step = one_res
                logs_train_save(writer, result=result, now_epoch=iter_step)
            inter_result.clear()

            logs_train_save(writer, result=val_result, now_epoch=now_epoch)
            model.save_model(args, optim_blocks, scheduler_blocks, global_step, now_epoch)


def logs_train_save(writer, result=None, now_epoch=1):
    """

    :param result:
    :param writer:
    :param now_epoch:
    :return:
    """
    for key in result.keys():
        if "values" in key:
            for key_loss in result[key].keys():
                writer.add_scalar(f'{key}/{key_loss}', result[key][key_loss], now_epoch)
        if "accuracy" in key:
            for key_loss in result[key].keys():
                writer.add_scalar(f'{key}/{key_loss}', result[key][key_loss], now_epoch)
        if "images" in key:
            for key_output in result[key].keys():
                writer.add_images(f'{key}/{key_output}', result[key][key_output], now_epoch)




def train_print():
    train_noise_dict = {"Jpeg": 50, "GaussianBlur": 1.5, "GaussianNoise": 0.05, "SaltPepperNoise": 0.15,
                        "MedianFilter": 7, "Dropout": 0.3}
    test_noise_dict = {"Jpeg": 50, "GaussianBlur": 3., "GaussianNoise": 0.3, "SaltPepperNoise": 0.3, "MedianFilter": 9,
                       "Dropout": 0.5}
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=6, help="ID of the GPU to use")
    parser.add_argument("--device", type=str, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument('--val_dataset_path', type=str, default=r'/data/chenjiale/datasets/DIV2K/val')
    parser.add_argument('--dataset_path', type=str, default=r'/data/chenjiale/datasets/DIV2K/train')  # MiniVOCAL2012 DIV2K_train_HR DIV2K_valid_HR
    parser.add_argument('--im_size', type=int, default=512)
    parser.add_argument('--train_noise_dict', type=dict, default=train_noise_dict)
    parser.add_argument('--test_noise_dict', type=dict, default=test_noise_dict)
    parser.add_argument('--hard_round', type=bool, default=False)
    parser.add_argument('--fc', type=bool, default=False)
    parser.add_argument("--train_name", type=str, default="gray")
    parser.add_argument('--max_step', type=int, default=1)
    parser.add_argument('--k_max', type=int, default=1)
    parser.add_argument('--bit_length', type=int, default=256)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--min_size', type=int, default=32)
    parser.add_argument('--queue_len', type=int, default=3)
    parser.add_argument('--val_save_epoch', type=int, default=5)
    parser.add_argument('--channel_dim', type=int, default=1)
    parser.add_argument("--lr", type=float, default=10 ** -4)
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/")
    parser.add_argument("--lambda_stego", type=float, default=1.)
    parser.add_argument("--lambda_lpips", type=float, default=5.)
    parser.add_argument("--lambda_secret", type=float, default=1e4)
    parser.add_argument("--lambda_z", type=float, default=1e-3)
    parser.add_argument("--v", type=float, default=0.75)
    parser.add_argument("--lambda_penalty", type=float, default=1e6)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument('--num_epoch', type=int, default=1005)
    parser.add_argument('--seed', type=int, default=99)
    parser.add_argument('--logs_path', type=str, default=r"logs")
    parser.add_argument('--continue_train', type=bool, default=False)
    args = parser.parse_args()
    if torch.cuda.is_available():
        args.device = torch.device(f"cuda:{args.gpu_id}")
    else:
        args.device = torch.device("cpu")
    train(args)


if __name__ == "__main__":
    train_print()
