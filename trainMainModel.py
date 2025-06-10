import argparse
import math
import os
import random
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

try:
    import comet_ml
except ImportError:
    comet_ml = None

from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    check_amp,
    check_dataset,
    check_file,
    check_git_info,
    check_git_status,
    check_img_size,
    check_requirements,
    check_suffix,
    check_yaml,
    colorstr,
    get_latest_run,
    increment_path,
    init_seeds,
    intersect_dicts,
    labels_to_class_weights,
    labels_to_image_weights,
    methods,
    one_cycle,
    print_args,
    print_mutation,
    strip_optimizer,
    yaml_save,
    non_max_suppression
)
from utils.loggers import LOGGERS, Loggers
from utils.loggers.comet.comet_utils import check_comet_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve
from utils.torch_utils import (
    EarlyStopping,
    ModelEMA,
    de_parallel,
    select_device,
    smart_DDP,
    smart_optimizer,
    smart_resume,
    torch_distributed_zero_first,
)

current_script = Path(__file__).resolve()
base_folder = current_script.parents[0]
if str(base_folder) not in sys.path:
    sys.path.append(str(base_folder))
base_folder = Path(os.path.relpath(base_folder, Path.cwd()))

import val as val_module
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download, is_url


def plot_separate_metrics(metrics_history, save_dir):
    for metric, values in metrics_history.items():
        if metric == 'epoch':
            continue
        plt.figure()
        plt.plot(metrics_history['epoch'], values, marker='o')
        plt.title(f"{metric} over epochs")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f"{metric}_plot.png"))
        plt.close()


def plot_bbox_size_distribution(dataset, save_dir):
    widths = []
    heights = []
    for labels in dataset.labels:
        for box in labels:
            widths.append(box[3])
            heights.append(box[4])
    plt.figure()
    plt.hist(widths, bins=20, alpha=0.5, label='Width')
    plt.hist(heights, bins=20, alpha=0.5, label='Height')
    plt.title("Bounding Box Size Distribution (Normalized)")
    plt.xlabel("Normalized size")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "bbox_size_distribution.png"))
    plt.close()


def save_individual_validation_images(model, dataloader, save_dir, device, use_amp, conf_thres=0.25, iou_thres=0.45):
    individual_dir = Path(save_dir) / "val_individual"
    individual_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for batch_i, (images, labels, paths, _) in enumerate(tqdm(dataloader, desc="Saving individual validation images")):
            images = images.to(device, non_blocking=True).float() / 255
            with torch.cuda.amp.autocast(enabled=use_amp):
                preds = model(images)
                detections = non_max_suppression(preds, conf_thres, iou_thres)
            for i, det in enumerate(detections):
                img = images[i].cpu().numpy().transpose(1, 2, 0) * 255
                img = img.astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                if det is not None and len(det):
                    for *xyxy, conf, cls in det:
                        label_text = f"{model.names[int(cls)] if hasattr(model, 'names') else int(cls)}: {conf:.2f}"
                        cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                        cv2.putText(img, label_text, (int(xyxy[0]), int(xyxy[1]) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                filename = Path(paths[i]).stem + ".jpg"
                cv2.imwrite(str(individual_dir / filename), img)


local_rank = int(os.getenv("LOCAL_RANK", -1))
global_rank = int(os.getenv("RANK", -1))
world_size = int(os.getenv("WORLD_SIZE", 1))
git_info = check_git_info()


def run_training(hype_config, train_args, device, callbacks):
    save_folder = Path(train_args.save_dir)
    max_epochs = train_args.epochs
    batch_size = train_args.batch_size
    init_wts = train_args.weights
    single_class_mode = train_args.single_cls
    evolving = train_args.evolve
    data_config_path = train_args.data
    model_config_path = train_args.cfg
    resume_training = train_args.resume
    skip_val = train_args.noval
    dont_save = train_args.nosave
    num_workers = train_args.workers
    freeze_layers = train_args.freeze

    callbacks.run("on_pretrain_routine_start")
    weights_folder = save_folder / "weights"
    (weights_folder.parent if evolving else weights_folder).mkdir(parents=True, exist_ok=True)
    last_model_file = weights_folder / "last.pt"
    best_model_file = weights_folder / "best.pt"

    if isinstance(hype_config, str):
        with open(hype_config, errors="ignore") as f:
            hype_config = yaml.safe_load(f)
    LOGGER.info(colorstr("hyperparameters: ") + ", ".join(f"{k}={v}" for k, v in hype_config.items()))
    train_args.hyp = hype_config.copy()
    if not evolving:
        yaml_save(save_folder / "hyp.yaml", hype_config)
        yaml_save(save_folder / "opt.yaml", vars(train_args))

    data_config = None
    if global_rank in (-1, 0):
        logger_types = list(LOGGERS)
        if getattr(train_args, "ndjson_console", False):
            logger_types.append("ndjson_console")
        if getattr(train_args, "ndjson_file", False):
            logger_types.append("ndjson_file")
        logger_manager = Loggers(save_dir=save_folder, weights=init_wts, opt=train_args, hyp=hype_config, logger=LOGGER, include=tuple(logger_types))
        for method in methods(logger_manager):
            callbacks.register_action(method, callback=getattr(logger_manager, method))
        data_config = logger_manager.remote_dataset
        if resume_training:
            init_wts, max_epochs, hype_config, batch_size = train_args.weights, train_args.epochs, train_args.hyp, train_args.batch_size
    data_config = data_config or check_dataset(data_config_path)
    train_data_path, val_data_path = data_config["train"], data_config["val"]
    class_names = {k: v for k, v in data_config["names"].items()}
    num_classes = len(class_names)
    is_coco = isinstance(val_data_path, str) and val_data_path.endswith("coco/val2017.txt")
    check_suffix(init_wts, ".pt")
    has_pretrained = init_wts.endswith(".pt")
    if has_pretrained:
        with torch_distributed_zero_first(local_rank):
            init_wts = attempt_download(init_wts)
        pretrained_ckpt = torch.load(init_wts, map_location="cpu")
        yolo_model = Model(model_config_path or pretrained_ckpt["model"].yaml, ch=3, nc=num_classes, anchors=hype_config.get("anchors")).to(device)
        skip_keys = ["anchor"] if (model_config_path or hype_config.get("anchors")) and not resume_training else []
        loaded_state = pretrained_ckpt["model"].float().state_dict()
        loaded_state = intersect_dicts(loaded_state, yolo_model.state_dict(), exclude=skip_keys)
        yolo_model.load_state_dict(loaded_state, strict=False)
        LOGGER.info(f"Yo, loaded {len(loaded_state)}/{len(yolo_model.state_dict())} parameters from {init_wts}")
    else:
        yolo_model = Model(model_config_path, ch=3, nc=num_classes, anchors=hype_config.get("anchors")).to(device)
    use_amp = check_amp(yolo_model)
    freeze_prefixes = [f"model.{layer_idx}." for layer_idx in (freeze_layers if len(freeze_layers) > 1 else range(freeze_layers[0]))]
    for param_name, param in yolo_model.named_parameters():
        param.requires_grad = True
        if any(prefix in param_name for prefix in freeze_prefixes):
            LOGGER.info(f"Freezing {param_name}")
            param.requires_grad = False

    grid_size = max(int(yolo_model.stride.max()), 32)
    img_size = check_img_size(train_args.imgsz, grid_size, floor=grid_size * 2)
    if global_rank == -1 and batch_size == -1:
        batch_size = check_train_batch_size(yolo_model, img_size, use_amp)
        logger_manager.on_params_update({"batch_size": batch_size})
    nominal_bs = 64
    grad_accum_steps = max(round(nominal_bs / batch_size), 1)
    hype_config["weight_decay"] *= batch_size * grad_accum_steps / nominal_bs
    optimizer = smart_optimizer(yolo_model, train_args.optimizer, hype_config["lr0"], hype_config["momentum"], hype_config["weight_decay"])
    if train_args.cos_lr:
        lr_lambda = one_cycle(1, hype_config["lrf"], max_epochs)
    else:
        def lr_lambda(epoch_frac):
            return (1 - epoch_frac / max_epochs) * (1.0 - hype_config["lrf"]) + hype_config["lrf"]
    lr_scheduler_obj = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    ema_model = ModelEMA(yolo_model) if global_rank in (-1, 0) else None
    best_fitness = 0.0
    start_epoch = 0
    if has_pretrained:
        if resume_training:
            best_fitness, start_epoch, max_epochs = smart_resume(pretrained_ckpt, optimizer, ema_model, init_wts, max_epochs, resume_training)
        del pretrained_ckpt, loaded_state
    if device.type != "cpu" and global_rank == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning("WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\nSee Multi-GPU Tutorial at [Multi-GPU Tutorial](site:docs.ultralytics.com/yolov5/tutorials/multi_gpu_training) to get started.")
        yolo_model = torch.nn.DataParallel(yolo_model)
    if train_args.sync_bn and device.type != "cpu" and global_rank != -1:
        yolo_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(yolo_model).to(device)
        LOGGER.info("Using SyncBatchNorm()")
    train_loader, dataset_obj = create_dataloader(train_data_path, img_size, batch_size // world_size, grid_size, single_class_mode, hyp=hype_config, augment=True, cache=None if train_args.cache == "val" else train_args.cache, rect=train_args.rect, rank=local_rank, workers=num_workers, image_weights=train_args.image_weights, quad=train_args.quad, prefix=colorstr("train: "), shuffle=True, seed=train_args.seed)
    all_labels = np.concatenate(dataset_obj.labels, 0)
    max_label = int(all_labels[:, 0].max())
    assert max_label < num_classes, f"Label {max_label} exceeds num_classes={num_classes} in {data_config_path}. Possible labels are 0-{num_classes - 1}"
    if global_rank in (-1, 0):
        val_loader = create_dataloader(val_data_path, img_size, batch_size // world_size * 2, grid_size, single_class_mode, hyp=hype_config, cache=None if skip_val else train_args.cache, rect=True, rank=-1, workers=num_workers * 2, pad=0.5, prefix=colorstr("val: "))[0]
        if not resume_training:
            if not train_args.noautoanchor:
                check_anchors(dataset_obj, model=yolo_model, thr=hype_config["anchor_t"], imgsz=img_size)
            yolo_model.half().float()
        callbacks.run("on_pretrain_routine_end", all_labels, class_names)
    if device.type != "cpu" and global_rank != -1:
        yolo_model = smart_DDP(yolo_model)
    num_last_layers = de_parallel(yolo_model).model[-1].nl
    hype_config["box"] *= 3 / num_last_layers
    hype_config["cls"] *= num_classes / 80 * 3 / num_last_layers
    hype_config["obj"] *= (img_size / 640) ** 2 * 3 / num_last_layers
    hype_config["label_smoothing"] = train_args.label_smoothing
    yolo_model.nc = num_classes
    yolo_model.hyp = hype_config
    yolo_model.class_weights = labels_to_class_weights(dataset_obj.labels, num_classes).to(device) * num_classes
    yolo_model.names = class_names

    start_time = time.time()
    total_batches = len(train_loader)
    warmup_iters = max(round(hype_config["warmup_epochs"] * total_batches), 100)
    last_opt_step = -1
    map_scores = np.zeros(num_classes)
    train_results = (0, 0, 0, 0, 0, 0, 0)
    lr_scheduler_obj.last_epoch = start_epoch - 1
    grad_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    early_stopper = EarlyStopping(patience=train_args.patience)
    stop_training = False
    loss_calculator = ComputeLoss(yolo_model)
    callbacks.run("on_train_start")
    LOGGER.info(f"Image sizes {img_size} train, {img_size} val\nUsing {train_loader.num_workers * world_size} dataloader workers\nLogging to {colorstr('bold', save_folder)}\nStarting training for {max_epochs} epochs...")
    if global_rank in (-1, 0):
        metrics_history = {
            'epoch': [],
            'precision': [],
            'recall': [],
            'mAP50': [],
            'mAP50-95': [],
            'box_loss': [],
            'obj_loss': [],
            'cls_loss': [],
            'lr': [],
        }
    for epoch in range(start_epoch, max_epochs):
        callbacks.run("on_train_epoch_start")
        yolo_model.train()
        if train_args.image_weights:
            cls_weights = yolo_model.class_weights.cpu().numpy() * (1 - map_scores) ** 2 / num_classes
            img_weights = labels_to_image_weights(dataset_obj.labels, nc=num_classes, class_weights=cls_weights)
            dataset_obj.indices = random.choices(range(dataset_obj.n), weights=img_weights, k=dataset_obj.n)
        running_loss = torch.zeros(3, device=device)
        if global_rank != -1:
            train_loader.sampler.set_epoch(epoch)
        batch_iter = enumerate(train_loader)
        LOGGER.info(("%11s" * 7) % ("Epoch", "GPU_mem", "box_loss", "obj_loss", "cls_loss", "Inst", "Size"))
        if global_rank in (-1, 0):
            batch_iter = tqdm(batch_iter, total=total_batches, bar_format=TQDM_BAR_FORMAT)
        optimizer.zero_grad()
        for i, (images, labels, paths, _) in batch_iter:
            callbacks.run("on_train_batch_start")
            global_batch_idx = i + total_batches * epoch
            images = images.to(device, non_blocking=True).float() / 255
            if global_batch_idx <= warmup_iters:
                warmup_range = [0, warmup_iters]
                grad_accum_steps = max(1, np.interp(global_batch_idx, warmup_range, [1, nominal_bs / batch_size]).round())
                for j, group in enumerate(optimizer.param_groups):
                    group["lr"] = np.interp(global_batch_idx, warmup_range, [hype_config["warmup_bias_lr"] if j == 0 else 0.0, group["initial_lr"] * lr_lambda(epoch)])
                    if "momentum" in group:
                        group["momentum"] = np.interp(global_batch_idx, warmup_range, [hype_config["warmup_momentum"], hype_config["momentum"]])
            if train_args.multi_scale:
                new_img_size = random.randrange(int(img_size * 0.5), int(img_size * 1.5) + grid_size) // grid_size * grid_size
                scale_factor = new_img_size / max(images.shape[2:])
                if scale_factor != 1:
                    new_shape = [math.ceil(dim * scale_factor / grid_size) * grid_size for dim in images.shape[2:]]
                    images = nn.functional.interpolate(images, size=new_shape, mode="bilinear", align_corners=False)
            with torch.cuda.amp.autocast(enabled=use_amp):
                preds = yolo_model(images)
                loss_val, loss_components = loss_calculator(preds, labels.to(device))
                if global_rank != -1:
                    loss_val *= world_size
                if train_args.quad:
                    loss_val *= 4.0
            grad_scaler.scale(loss_val).backward()
            if global_batch_idx - last_opt_step >= grad_accum_steps:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(yolo_model.parameters(), max_norm=10.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                optimizer.zero_grad()
                if ema_model:
                    ema_model.update(yolo_model)
                last_opt_step = global_batch_idx
            if global_rank in (-1, 0):
                running_loss = (running_loss * i + loss_components) / (i + 1)
                gpu_mem = f"{torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0:.3g}G"
                desc = (("%11s" * 2) + ("%11.4g" * 5)) % (f"{epoch}/{max_epochs - 1}", gpu_mem, *running_loss, labels.shape[0], images.shape[-1])
                batch_iter.set_description(desc)
                callbacks.run("on_train_batch_end", yolo_model, global_batch_idx, images, labels, paths, list(running_loss))
                if callbacks.stop_training:
                    return
        lr_scheduler_obj.step()
        if global_rank in (-1, 0):
            callbacks.run("on_train_epoch_end", epoch=epoch)
            ema_model.update_attr(yolo_model, include=["yaml", "nc", "hyp", "names", "stride", "class_weights"])
            final_epoch = (epoch + 1 == max_epochs) or early_stopper.possible_stop
            if not skip_val or final_epoch:
                train_results, map_scores, _ = val_module.run(data_config, batch_size=batch_size // world_size * 2, imgsz=img_size, half=use_amp, model=ema_model.ema, single_cls=single_class_mode, dataloader=val_loader, save_dir=save_folder, plots=False, callbacks=callbacks, compute_loss=loss_calculator)
            current_fitness = fitness(np.array(train_results).reshape(1, -1))
            metrics_history['epoch'].append(epoch)
            metrics_history['precision'].append(train_results[0])
            metrics_history['recall'].append(train_results[1])
            metrics_history['mAP50'].append(train_results[2])
            metrics_history['mAP50-95'].append(train_results[3])
            metrics_history['box_loss'].append(running_loss[0].item())
            metrics_history['obj_loss'].append(running_loss[1].item())
            metrics_history['cls_loss'].append(running_loss[2].item())
            lr_values = [group["lr"] for group in optimizer.param_groups]
            metrics_history['lr'].append(np.mean(lr_values))
            stop_training = early_stopper(epoch=epoch, fitness=current_fitness)
            if current_fitness > best_fitness:
                best_fitness = current_fitness
            log_vals = list(running_loss) + list(train_results) + lr_values
            callbacks.run("on_fit_epoch_end", log_vals, epoch, best_fitness, current_fitness)
            if (not dont_save) or (final_epoch and not evolving):
                checkpoint = {
                    "epoch": epoch,
                    "best_fitness": best_fitness,
                    "model": deepcopy(de_parallel(yolo_model)).half(),
                    "ema": deepcopy(ema_model.ema).half(),
                    "updates": ema_model.updates,
                    "optimizer": optimizer.state_dict(),
                    "opt": vars(train_args),
                    "git": git_info,
                    "date": datetime.now().isoformat(),
                }
                torch.save(checkpoint, last_model_file)
                if best_fitness == current_fitness:
                    torch.save(checkpoint, best_model_file)
                if train_args.save_period > 0 and epoch % train_args.save_period == 0:
                    torch.save(checkpoint, weights_folder / f"epoch{epoch}.pt")
                del checkpoint
                callbacks.run("on_model_save", last_model_file, epoch, final_epoch, best_fitness, current_fitness)
        if global_rank != -1:
            broadcast_val = [stop_training if global_rank == 0 else None]
            dist.broadcast_object_list(broadcast_val, 0)
            if global_rank != 0:
                stop_training = broadcast_val[0]
        if stop_training:
            break
    if global_rank in (-1, 0):
        elapsed = (time.time() - start_time) / 3600
        LOGGER.info(f"\nDone! {epoch - start_epoch + 1} epochs completed in {elapsed:.3f} hours.")
        for model_file in (last_model_file, best_model_file):
            if model_file.exists():
                strip_optimizer(model_file)
                if model_file is best_model_file:
                    LOGGER.info(f"\nValidating {model_file}...")
                    val_results, _, _ = val_module.run(data_config, batch_size=batch_size // world_size * 2, imgsz=img_size, model=attempt_load(model_file, device).half(), iou_thres=0.65 if is_coco else 0.60, single_cls=single_class_mode, dataloader=val_loader, save_dir=save_folder, save_json=is_coco, verbose=True, plots=(not evolving and not train_args.noplots), callbacks=callbacks, compute_loss=loss_calculator)
                    if is_coco:
                        callbacks.run("on_fit_epoch_end", list(running_loss) + list(val_results) + lr_values, epoch, best_fitness, current_fitness)
        plot_separate_metrics(metrics_history, str(save_folder))
        plot_bbox_size_distribution(dataset_obj, str(save_folder))
        if not skip_val and 'val_loader' in locals():
            model_to_save = ema_model.ema if ema_model else yolo_model
            save_individual_validation_images(model_to_save, val_loader, str(save_folder), device, use_amp)
        callbacks.run("on_train_end", last_model_file, best_model_file, epoch, train_results)
    torch.cuda.empty_cache()
    return train_results


def parse_options(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=base_folder / "yolov5s.pt", help="Path to initial weights")
    parser.add_argument("--cfg", type=str, default="", help="Path to model.yaml")
    parser.add_argument("--data", type=str, default=base_folder / "data/coco128.yaml", help="Path to dataset.yaml")
    parser.add_argument("--hyp", type=str, default=base_folder / "data/hyps/hyp.scratch-low.yaml", help="Path to hyperparameters file")
    parser.add_argument("--epochs", type=int, default=100, help="Total training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Total batch size for all GPUs, -1 for autobatch")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="Train/val image size in pixels")
    parser.add_argument("--rect", action="store_true", help="Rectangular training")
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="Resume most recent training")
    parser.add_argument("--nosave", action="store_true", help="Only save final checkpoint")
    parser.add_argument("--noval", action="store_true", help="Only validate final epoch")
    parser.add_argument("--noautoanchor", action="store_true", help="Disable AutoAnchor")
    parser.add_argument("--noplots", action="store_true", help="Disable saving plot files")
    parser.add_argument("--evolve", type=int, nargs="?", const=300, help="Evolve hyperparameters for given generations")
    parser.add_argument("--evolve_population", type=str, default=base_folder / "data/hyps", help="Location for population files")
    parser.add_argument("--resume_evolve", type=str, default=None, help="Resume evolution from last generation")
    parser.add_argument("--bucket", type=str, default="", help="gsutil bucket")
    parser.add_argument("--cache", type=str, nargs="?", const="ram", help="Cache images in ram/disk")
    parser.add_argument("--image-weights", action="store_true", help="Use weighted image selection")
    parser.add_argument("--device", default="", help="CUDA device, e.g. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--multi-scale", action="store_true", help="Vary image size by +/- 50%")
    parser.add_argument("--single-cls", action="store_true", help="Train multi-class data as single-class")
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW"], default="SGD", help="Optimizer type")
    parser.add_argument("--sync-bn", action="store_true", help="Use SyncBatchNorm (only in DDP mode)")
    parser.add_argument("--workers", type=int, default=8, help="Max dataloader workers per RANK in DDP mode")
    parser.add_argument("--project", default=base_folder / "runs/train", help="Save to project/name")
    parser.add_argument("--name", default="exp", help="Save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="Existing project/name ok, do not increment")
    parser.add_argument("--quad", action="store_true", help="Quad dataloader")
    parser.add_argument("--cos-lr", action="store_true", help="Use cosine LR scheduler")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing epsilon")
    parser.add_argument("--patience", type=int, default=100, help="EarlyStopping patience (epochs without improvement)")
    parser.add_argument("--freeze", nargs="+", type=int, default=[0], help="Freeze layers: e.g. backbone=10 or first3=0 1 2")
    parser.add_argument("--save-period", type=int, default=-1, help="Save checkpoint every x epochs (disabled if < 1)")
    parser.add_argument("--seed", type=int, default=0, help="Global training seed")
    parser.add_argument("--local_rank", type=int, default=-1, help="DDP multi-GPU argument, do not modify")
    parser.add_argument("--entity", default=None, help="Entity")
    parser.add_argument("--upload_dataset", nargs="?", const=True, default=False, help='Upload data, "val" option')
    parser.add_argument("--bbox_interval", type=int, default=-1, help="Set bounding-box image logging interval")
    parser.add_argument("--artifact_alias", type=str, default="latest", help="Dataset artifact version to use")
    parser.add_argument("--ndjson-console", action="store_true", help="Log ndjson to console")
    parser.add_argument("--ndjson-file", action="store_true", help="Log ndjson to file")
    return parser.parse_known_args()[0] if known else parser.parse_args()


def main_run(train_args, callbacks=Callbacks()):
    if global_rank in (-1, 0):
        print_args(vars(train_args))
        check_git_status()
        check_requirements(base_folder / "requirements.txt")
    if train_args.resume and not check_comet_resume(train_args) and not train_args.evolve:
        latest_ckpt = Path(check_file(train_args.resume) if isinstance(train_args.resume, str) else get_latest_run())
        opt_yaml = latest_ckpt.parent.parent / "opt.yaml"
        if opt_yaml.is_file():
            with open(opt_yaml, errors="ignore") as f:
                opt_dict = yaml.safe_load(f)
        else:
            opt_dict = torch.load(latest_ckpt, map_location="cpu")["opt"]
        train_args = argparse.Namespace(**opt_dict)
        train_args.cfg, train_args.weights, train_args.resume = "", str(latest_ckpt), True
        if is_url(train_args.data):
            train_args.data = check_file(train_args.data)
    else:
        train_args.data, train_args.cfg, train_args.hyp, train_args.weights, train_args.project = (
            check_file(train_args.data),
            check_yaml(train_args.cfg),
            check_yaml(train_args.hyp),
            str(train_args.weights),
            str(train_args.project),
        )
        assert len(train_args.cfg) or len(train_args.weights), "Either --cfg or --weights must be specified"
        if train_args.evolve:
            if train_args.project == str(base_folder / "runs/train"):
                train_args.project = str(base_folder / "runs/evolve")
            train_args.exist_ok, train_args.resume = train_args.resume, False
        if train_args.name == "cfg":
            train_args.name = Path(train_args.cfg).stem
        train_args.save_dir = str(increment_path(Path(train_args.project) / train_args.name, exist_ok=train_args.exist_ok))
    device = select_device(train_args.device, batch_size=train_args.batch_size)
    if local_rank != -1:
        assert not train_args.image_weights, "--image-weights is incompatible with DDP"
        assert not train_args.evolve, "--evolve is incompatible with DDP"
        assert train_args.batch_size != -1, "AutoBatch with --batch-size -1 not allowed in DDP"
        assert train_args.batch_size % world_size == 0, "--batch-size must be a multiple of WORLD_SIZE"
        assert torch.cuda.device_count() > local_rank, "Not enough CUDA devices for DDP"
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo", timeout=timedelta(seconds=10800))
    if not train_args.evolve:
        run_training(train_args.hyp, train_args, device, callbacks)
    else:
        meta_info = {
            "lr0": (False, 1e-5, 1e-1),
            "lrf": (False, 0.01, 1.0),
            "momentum": (False, 0.6, 0.98),
            "weight_decay": (False, 0.0, 0.001),
            "warmup_epochs": (False, 0.0, 5.0),
            "warmup_momentum": (False, 0.0, 0.95),
            "warmup_bias_lr": (False, 0.0, 0.2),
            "box": (False, 0.02, 0.2),
            "cls": (False, 0.2, 4.0),
            "cls_pw": (False, 0.5, 2.0),
            "obj": (False, 0.2, 4.0),
            "obj_pw": (False, 0.5, 2.0),
            "iou_t": (False, 0.1, 0.7),
            "anchor_t": (False, 2.0, 8.0),
            "anchors": (False, 2.0, 10.0),
            "fl_gamma": (False, 0.0, 2.0),
            "hsv_h": (True, 0.0, 0.1),
            "hsv_s": (True, 0.0, 0.9),
            "hsv_v": (True, 0.0, 0.9),
            "degrees": (True, 0.0, 45.0),
            "translate": (True, 0.0, 0.9),
            "scale": (True, 0.0, 0.9),
            "shear": (True, 0.0, 10.0),
            "perspective": (True, 0.0, 0.001),
            "flipud": (True, 0.0, 1.0),
            "fliplr": (True, 0.0, 1.0),
            "mosaic": (True, 0.0, 1.0),
            "mixup": (True, 0.0, 1.0),
            "copy_paste": (True, 0.0, 1.0),
        }
        pop_size = 50
        min_mut_rate = 0.01
        max_mut_rate = 0.5
        min_cross_rate = 0.5
        max_cross_rate = 1
        min_elite = 2
        max_elite = 5
        min_tour = 2
        max_tour = 10
        with open(train_args.hyp, errors="ignore") as f:
            hype_config = yaml.safe_load(f)
            if "anchors" not in hype_config:
                hype_config["anchors"] = 3
        if train_args.noautoanchor:
            del hype_config["anchors"], meta_info["anchors"]
        train_args.noval, train_args.nosave, save_folder = True, True, Path(train_args.save_dir)
        evolve_yaml_path = save_folder / "hyp_evolve.yaml"
        evolve_csv_path = save_folder / "evolve.csv"
        if train_args.bucket:
            subprocess.run(["gsutil", "cp", f"gs://{train_args.bucket}/evolve.csv", str(evolve_csv_path)])
        delete_keys = [key for key, value in meta_info.items() if not value[0]]
        hype_ga = hype_config.copy()
        for key in delete_keys:
            del meta_info[key]
            del hype_ga[key]
        lower_limits = np.array([meta_info[k][1] for k in hype_ga.keys()])
        upper_limits = np.array([meta_info[k][2] for k in hype_ga.keys()])
        gene_ranges = [(lower_limits[i], upper_limits[i]) for i in range(len(upper_limits))]
        init_values = []
        if train_args.resume_evolve is not None:
            resume_path = Path(base_folder / train_args.resume_evolve)
            assert resume_path.is_file(), "Evolution population file path is incorrect!"
            with open(resume_path, errors="ignore") as f:
                evolve_pop = yaml.safe_load(f)
                for val in evolve_pop.values():
                    arr = np.array([val[k] for k in hype_ga.keys()])
                    init_values.append(list(arr))
        else:
            yaml_files = [f for f in os.listdir(train_args.evolve_population) if f.endswith(".yaml")]
            for fname in yaml_files:
                with open(os.path.join(train_args.evolve_population, fname)) as yf:
                    val = yaml.safe_load(yf)
                    arr = np.array([val[k] for k in hype_ga.keys()])
                    init_values.append(list(arr))
        if not init_values:
            population = [generate_individual(gene_ranges, len(hype_ga)) for _ in range(pop_size)]
        elif pop_size > 1:
            population = [generate_individual(gene_ranges, len(hype_ga)) for _ in range(pop_size - len(init_values))]
            for init_val in init_values:
                population = [init_val] + population
        gene_keys = list(hype_ga.keys())
        for gen in range(train_args.evolve):
            if gen >= 1:
                save_dict = {}
                for i in range(len(population)):
                    indiv_dict = {gene_keys[j]: float(population[i][j]) for j in range(len(population[i]))}
                    save_dict[f"gen{gen}_indiv{i}"] = indiv_dict
                with open(save_folder / "evolve_population.yaml", "w") as out_f:
                    yaml.dump(save_dict, out_f, default_flow_style=False)
            elite_count = min_elite + int((max_elite - min_elite) * (gen / train_args.evolve))
            fitness_scores = []
            for indiv in population:
                for key, val in zip(hype_ga.keys(), indiv):
                    hype_ga[key] = val
                hype_config.update(hype_ga)
                results = run_training(hype_config.copy(), train_args, device, callbacks)
                callbacks = Callbacks()
                print_mutation(("metrics/precision", "metrics/recall", "metrics/mAP_0.5", "metrics/mAP_0.5:0.95", "val/box_loss", "val/obj_loss", "val/cls_loss"), results, hype_config.copy(), save_folder, train_args.bucket)
                fitness_scores.append(results[2])
            selected_indices = []
            for _ in range(pop_size - elite_count):
                tour_size = max(max(2, min_tour), int(min(max_tour, pop_size) - (gen / (train_args.evolve / 10))))
                tour_inds = random.sample(range(pop_size), tour_size)
                tour_fit = [fitness_scores[j] for j in tour_inds]
                winner = tour_inds[tour_fit.index(max(tour_fit))]
                selected_indices.append(winner)
            elite_inds = [i for i in range(pop_size) if fitness_scores[i] in sorted(fitness_scores)[-elite_count:]]
            selected_indices.extend(elite_inds)
            next_gen = []
            for _ in range(pop_size):
                parent1 = population[selected_indices[random.randint(0, pop_size - 1)]]
                parent2 = population[selected_indices[random.randint(0, pop_size - 1)]]
                cross_rate = max(min_cross_rate, min(max_cross_rate, max_cross_rate - (gen / train_args.evolve)))
                if random.uniform(0, 1) < cross_rate:
                    cross_point = random.randint(1, len(hype_ga) - 1)
                    child = parent1[:cross_point] + parent2[cross_point:]
                else:
                    child = parent1.copy()
                mut_rate = max(min_mut_rate, min(max_mut_rate, max_mut_rate - (gen / train_args.evolve)))
                for j in range(len(hype_ga)):
                    if random.uniform(0, 1) < mut_rate:
                        child[j] += random.uniform(-0.1, 0.1)
                        child[j] = min(max(child[j], gene_ranges[j][0]), gene_ranges[j][1])
                next_gen.append(child)
            population = next_gen
        best_idx = fitness_scores.index(max(fitness_scores))
        best_indiv = population[best_idx]
        print("Best solution found:", best_indiv)
        plot_evolve(evolve_csv_path)
        LOGGER.info(f"Hyperparameter evolution finished after {train_args.evolve} generations.\nResults saved to {colorstr('bold', save_folder)}\nUsage example: $ python train.py --hyp {evolve_yaml_path}")
    return


def generate_individual(ranges, gene_count):
    return [random.uniform(ranges[i][0], ranges[i][1]) for i in range(gene_count)]


def execute(**kwargs):
    train_args = parse_options(True)
    for k, v in kwargs.items():
        setattr(train_args, k, v)
    main_run(train_args)
    return train_args


if __name__ == "__main__":
    train_args = parse_options()
    main_run(train_args) 