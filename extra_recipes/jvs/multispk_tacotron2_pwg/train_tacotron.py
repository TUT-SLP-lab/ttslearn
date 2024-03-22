from functools import partial
from logging import Logger
from pathlib import Path

import torch
import wandb
from hydra.utils import to_absolute_path
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from torch import nn
from tqdm import tqdm

import hydra
from ttslearn.contrib.multispk_util import collate_fn_ms_tacotron, setup
from ttslearn.tacotron.frontend.openjtalk import sequence_to_text
from ttslearn.train_util import (get_epochs_with_optional_tqdm, plot_2d_feats,
                                 plot_attention, save_checkpoint)
from ttslearn.util import make_non_pad_mask

logger: Logger = None


@torch.no_grad()
def eval_model(
    step,
    model,
    writer,
    in_feats,
    in_lens,
    out_feats,
    out_lens,
    spk_ids,
    is_inference,
):
    # 最大3つまで
    N = min(len(in_feats), 3)

    if is_inference:
        outs, outs_fine, att_ws, out_lens = [], [], [], []
        for idx in range(N):
            out, out_fine, _, att_w = model.inference(
                in_feats[idx][: in_lens[idx]], spk_ids[idx]
            )
            outs.append(out)
            outs_fine.append(out_fine)
            att_ws.append(att_w)
            out_lens.append(len(out))
    else:
        outs, outs_fine, _, att_ws, _, _ = model(in_feats, in_lens, out_feats, spk_ids)

    for idx in range(N):
        text = "".join(
            sequence_to_text(in_feats[idx][: in_lens[idx]].cpu().data.numpy())
        )
        if is_inference:
            group = f"utt{idx+1}_inference"
        else:
            group = f"utt{idx+1}_teacher_forcing"

        out = outs[idx][: out_lens[idx]]
        out_fine = outs_fine[idx][: out_lens[idx]]
        rf = model.decoder.reduction_factor
        att_w = att_ws[idx][: out_lens[idx] // rf, : in_lens[idx]]
        fig = plot_attention(att_w)
        writer.add_figure(f"{group}/attention", fig, step)
        plt.close()
        fig = plot_2d_feats(out, text)
        writer.add_figure(f"{group}/out_before_postnet", fig, step)
        plt.close()
        fig = plot_2d_feats(out_fine, text)
        writer.add_figure(f"{group}/out_after_postnet", fig, step)
        plt.close()
        if not is_inference:
            out_gt = out_feats[idx][: out_lens[idx]]
            fig = plot_2d_feats(out_gt, text)
            writer.add_figure(f"{group}/out_ground_truth", fig, step)
            plt.close()


def train_step(
    model,
    optimizer,
    lr_scheduler,
    train,
    criterions,
    in_feats,
    in_lens,
    out_feats,
    out_lens,
    stop_flags,
    spk_ids,
    spk_vector,
):
    optimizer.zero_grad()

    # Run forwaard
    outs, outs_fine, logits, _, spk_pred, asr_pred = model(in_feats, in_lens, out_feats, spk_ids)

    # Mask (B x T x 1)
    mask = make_non_pad_mask(out_lens).unsqueeze(-1).to(out_feats.device)
    out_feats = out_feats.masked_select(mask)
    outs = outs.masked_select(mask)
    outs_fine = outs_fine.masked_select(mask)
    stop_flags = stop_flags.masked_select(mask.squeeze(-1))
    logits = logits.masked_select(mask.squeeze(-1))

    # Loss
    decoder_out_loss = criterions["out_loss"](outs, out_feats)
    postnet_out_loss = criterions["out_loss"](outs_fine, out_feats)
    stop_token_loss = criterions["stop_token_loss"](logits, stop_flags)
    spk_pred_loss = 0 if spk_pred is None else criterions["spk_loss"](spk_pred, spk_vector)
    # TODO: asr_pred_lossの設定
    asr_pred_loss = 0 if asr_pred is None else criterions["asr_loss"](asr_pred, in_feats)
    # if spk_pred is None:
    #     print("""
    #           #########################################
    #           ########   spk_pred is None!!!   ########
    #           #########################################
    #           """)
    #     print(f"train is {train}, spk_ids: {len(spk_ids)}")
    voice_loss_rate = 0.90
    pred_loss_rate = 1.0 - voice_loss_rate
    if spk_pred is not None or asr_pred is not None:
        # loss = stop_token_loss + spk_pred_loss + asr_pred_loss
        voice_loss = (decoder_out_loss + postnet_out_loss) * voice_loss_rate
        pred_loss = (spk_pred_loss + asr_pred_loss) * pred_loss_rate
        loss = voice_loss + pred_loss + stop_token_loss
    else:
        loss = decoder_out_loss + postnet_out_loss + stop_token_loss

    loss_values = {
        "DecoderOutLoss": decoder_out_loss.item(),
        "PostnetOutLoss": postnet_out_loss.item(),
        "StopTokenLoss": stop_token_loss.item(),
        # "SpkPredLoss": spk_pred_loss.item(),
        # "AsrPredLoss": asr_pred_loss.item(),
        "Loss": loss.item(),
    }

    # Update
    if train:
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            logger.info("grad norm is NaN. Skip updating")
        else:
            optimizer.step()
        lr_scheduler.step()

    return loss_values


def _update_running_losses_(running_losses, loss_values):
    for key, val in loss_values.items():
        try:
            running_losses[key] += val
        except KeyError:
            running_losses[key] = val


def train_loop(config, device, model, optimizer, lr_scheduler, data_loaders, writer, wandb_run):
    criterions = {  # TODO: asr_pred_loss
        "out_loss": nn.MSELoss(),
        "stop_token_loss": nn.BCEWithLogitsLoss(),
        "spk_loss": nn.CrossEntropyLoss(),
        # "asr_loss": "",
    }

    out_dir = Path(to_absolute_path(config.train.out_dir))
    best_loss = torch.finfo(torch.float32).max
    train_iter = 1
    nepochs = config.train.nepochs
    num_spks = config.model.spk_rec_model.netG.num_classes
    # spk2id = make_spk2id(config)
    target_spk_probability = 1.0 / num_spks

    for epoch in get_epochs_with_optional_tqdm(config.tqdm, nepochs):
        for phase in data_loaders.keys():
            is_train = phase.startswith("train")
            model.train() if is_train else model.eval()
            running_losses = {}
            total = len(data_loaders[phase])
            for idx, (
                in_feats,
                in_lens,
                out_feats,
                out_lens,
                stop_flags,
                spk_ids,
            ) in tqdm(
                enumerate(data_loaders[phase]),
                desc=f"{phase} iter",
                leave=False,
                total=total,
            ):
                # ミニバッチのソート
                in_lens, indices = torch.sort(in_lens, dim=0, descending=True)
                in_feats, out_feats, out_lens = (
                    in_feats[indices].to(device),
                    out_feats[indices].to(device),
                    out_lens[indices].to(device),
                )
                stop_flags = stop_flags[indices].to(device)
                spk_ids = spk_ids[indices].to(device)

                # TODO: spk_idsからspk_vectorを生成する
                # どの話者を使うのか指定する必要があるので、もう少し設定含め構成を練る必要がある
                # と思ったけど、ひとまず中間話者を作りたいだけなので、人数で均等割りしたベクトルがあれば良い
                # 一旦の実装なので、本当は話者毎に確率を変えられるようにしたい
                spk_vector = torch.Tensor([
                    [target_spk_probability for _ in range(num_spks)]  # range(2)
                    for spk_id in spk_ids
                ]).to(device)

                loss_values = train_step(
                    model,
                    optimizer,
                    lr_scheduler,
                    is_train,
                    criterions,
                    in_feats,
                    in_lens,
                    out_feats,
                    out_lens,
                    stop_flags,
                    spk_ids,
                    spk_vector,
                )
                wandb_log_dict = {}
                wandb_log_dict["epoch"] = epoch
                for key, val in loss_values.items():
                    wandb_log_dict[f"{phase}/{key}"] = val
                if is_train:
                    for key, val in loss_values.items():
                        writer.add_scalar(f"{key}ByStep/train", val, train_iter)
                    writer.add_scalar(
                        "LearningRate", lr_scheduler.get_last_lr()[0], train_iter
                    )
                    train_iter += 1
                    wandb_log_dict[f"{phase}/LearningRate"] = lr_scheduler.get_last_lr()[0]
                _update_running_losses_(running_losses, loss_values)

                # 最初の検証用データに対して、中間結果の可視化
                if (
                    not is_train
                    and idx == 0
                    and epoch % config.train.eval_epoch_interval == 0
                ):
                    for is_inference in [False, True]:
                        eval_model(
                            train_iter,
                            model,
                            writer,
                            in_feats,
                            in_lens,
                            out_feats,
                            out_lens,
                            spk_ids,
                            is_inference,
                        )

                if idx+1 != total:
                    wandb_run.log(wandb_log_dict)

            # Epoch ごとのロスを出力
            for key, val in running_losses.items():
                ave_loss = val / len(data_loaders[phase])
                writer.add_scalar(f"{key}/{phase}", ave_loss, epoch)
                wandb_log_dict[f"{phase}/{key}_epoch"] = ave_loss

            ave_loss = running_losses["Loss"] / len(data_loaders[phase])
            wandb_log_dict[f"{phase}/ave_loss_epoch"] = ave_loss
            if not is_train and ave_loss < best_loss:
                best_loss = ave_loss
                save_checkpoint(logger, out_dir, model, optimizer, epoch, True)

            wandb_run.log(wandb_log_dict)

        if epoch % config.train.checkpoint_epoch_interval == 0:
            save_checkpoint(logger, out_dir, model, optimizer, epoch, False)

    # save at last epoch
    save_checkpoint(logger, out_dir, model, optimizer, nepochs)
    logger.info(f"The best loss was {best_loss}")

    return model


def make_spk2id(config):
    """ configから、spk2id(dict)を生成する
    """
    spk2id = dict()
    return spk2id


@hydra.main(config_path="conf/train_tacotron", config_name="config")
def my_app(config: DictConfig) -> None:
    global logger
    with wandb.init(project="ttslearn-multispk-spkrec", config=dict(config)) as wandb_run:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        collate_fn = partial(
            collate_fn_ms_tacotron, reduction_factor=config.model.netG.reduction_factor
        )
        model, optimizer, lr_scheduler, data_loaders, writer, logger = setup(
            config, device, collate_fn
        )
        train_loop(config, device, model, optimizer, lr_scheduler, data_loaders, writer, wandb_run)


if __name__ == "__main__":
    my_app()
