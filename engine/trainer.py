from pathlib import Path

import torch

from engine.evaluator import evaluate_model
from quant.sharpness import (
    AdaptiveSharpnessScheduler,
    build_case_perturbations,
    assign_gradients,
    combine_gradients,
    determine_sharpness_interval,
    quantization_kl_proxy,
    reuse_gradients,
    snapshot_gradients,
    temporary_perturbation,
)
from utils import AverageMeter, accuracy, save_checkpoint


class ISSAQTrainer:
    def __init__(
        self,
        args,
        model,
        task,
        criterion,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        logger,
        start_epoch=0,
    ):
        self.args = args
        self.model = model
        self.task = task
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.start_epoch = start_epoch
        self.best_metric = float("-inf")
        self.case_scheduler = AdaptiveSharpnessScheduler(
            low_threshold=args.scheduler_low,
            high_threshold=args.scheduler_high,
            beta=args.scheduler_beta,
            ema=args.scheduler_ema,
            alpha=args.alpha,
        )
        self.orthogonal_buffers = {}
        self.global_step = 0

    def fit(self):
        for epoch in range(self.start_epoch, self.args.epochs):
            self.model.controller.set_epoch(epoch, self.args.epochs)
            self.train_one_epoch(epoch)
            metrics = evaluate_model(self.args, self.model, self.task, self.val_loader, self.criterion, self.logger)
            if hasattr(self.model.controller, "update_validation_loss"):
                self.model.controller.update_validation_loss(metrics["loss"])
            metric = metrics["top1"] if self.task == "classification" else -metrics["loss"]
            latest_path = Path(self.args.output_dir) / "latest.pt"
            save_checkpoint(latest_path, self.model, self.optimizer, self.scheduler, epoch, metric)
            if metric > self.best_metric:
                self.best_metric = metric
                best_path = Path(self.args.output_dir) / "best.pt"
                save_checkpoint(best_path, self.model, self.optimizer, self.scheduler, epoch, metric)
            self.scheduler.step()

    def train_one_epoch(self, epoch):
        self.model.train()
        loss_meter = AverageMeter()
        top1_meter = AverageMeter()
        top5_meter = AverageMeter()
        bitops_meter = AverageMeter()
        sharp_meter = AverageMeter()
        kl_meter = AverageMeter()
        for step, batch in enumerate(self.train_loader):
            if self.args.max_train_steps > 0 and step >= self.args.max_train_steps:
                break
            images, targets = self._move_batch(batch)

            self.optimizer.zero_grad(set_to_none=True)
            outputs, meta = self.model(images, targets=targets, hard=self.args.controller_hard)
            task_loss = self.criterion.task_loss(outputs, targets)
            bitops_loss = self.criterion.bitops_loss(meta.bitops)
            vanilla_total = task_loss + self.criterion.lambda_bitops * bitops_loss
            vanilla_total.backward()
            vanilla_grads = snapshot_gradients(self.model)

            if self.task == "classification":
                with torch.no_grad():
                    fp_outputs = self.model.forward_full_precision(images)
                kl_value = quantization_kl_proxy(fp_outputs, outputs)
            else:
                kl_value = torch.tensor(0.0, device=images.device)
            case_weights = self.case_scheduler.step(kl_value)
            kl_meter.update(kl_value.detach().item() if isinstance(kl_value, torch.Tensor) else float(kl_value), images.size(0))

            sharpness_interval = determine_sharpness_interval(
                meta,
                base_cycle=self.args.sharpness_cycle,
                high_bit_cycle=self.args.high_bit_cycle,
                high_bit_threshold=self.args.high_bit_threshold,
            )
            needs_sharpness = self.global_step % sharpness_interval == 0

            if needs_sharpness:
                case_perturbations, rho = build_case_perturbations(
                    self.model,
                    meta,
                    rho_base=self.args.rho_base,
                    rho_gamma=self.args.rho_gamma,
                    sign_flip_prob=self.args.sign_flip_prob,
                )
                self.optimizer.zero_grad(set_to_none=True)
                sharpness_losses = []
                for mode in ["independent", "pre_quant", "post_quant"]:
                    with temporary_perturbation(self.model, case_perturbations[mode], mode):
                        perturbed_outputs, _ = self.model(images, targets=targets, hard=self.args.controller_hard)
                        sharpness_losses.append(self.criterion.task_loss(perturbed_outputs, targets))
                sharpness_loss = self.criterion.aggregate_sharpness(sharpness_losses, case_weights)
                sharpness_loss.backward()
                sharp_grads = snapshot_gradients(self.model)
                combined_grads, new_buffers, grad_stats = combine_gradients(
                    vanilla_grads,
                    sharp_grads,
                    self.orthogonal_buffers,
                    self.args.sharpness_lambda,
                    self.args.reuse_gamma,
                )
                self.orthogonal_buffers.update(new_buffers)
                sharp_meter.update(sharpness_loss.item(), images.size(0))
            else:
                combined_grads, grad_stats = reuse_gradients(
                    vanilla_grads,
                    self.orthogonal_buffers,
                    self.args.sharpness_lambda,
                    self.args.reuse_gamma,
                )
                sharpness_loss = task_loss.new_zeros(())
                rho = 0.0
                sharp_meter.update(0.0, images.size(0))

            assign_gradients(self.model, combined_grads)
            if self.args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()

            total_loss = self.criterion.combine(task_loss, bitops_loss, sharpness_loss)
            loss_meter.update(total_loss.item(), images.size(0))
            bitops_meter.update(meta.bitops.detach().item(), images.size(0))
            if self.task == "classification":
                top1, top5 = accuracy(outputs, targets, topk=(1, 5))
                top1_meter.update(top1.item(), images.size(0))
                top5_meter.update(top5.item(), images.size(0))

            if step % max(1, self.args.print_freq) == 0:
                bit_distribution = self._bit_distribution(meta)
                progress = self._format_progress(step, len(self.train_loader))
                case_display = self._format_case_weights(case_weights)
                if self.task == "classification":
                    self.logger.info(
                        "Epoch: [%d]  %s  Loss %.4f  Sharp %.4f  ACC1 %.2f%%  ACC5 %.2f%%  Bits %.2f  KL %.4f  Rho %.4f  Case %s  BitDist %s  Cos %.3f",
                        epoch,
                        progress,
                        loss_meter.avg,
                        sharp_meter.avg,
                        top1_meter.avg,
                        top5_meter.avg,
                        meta.avg_bits.mean().detach().item(),
                        kl_value.detach().item() if isinstance(kl_value, torch.Tensor) else float(kl_value),
                        rho,
                        case_display,
                        bit_distribution,
                        grad_stats.get("cosine", 0.0),
                    )
                else:
                    self.logger.info(
                        "Epoch: [%d]  %s  Loss %.4f  Sharp %.4f  Bits %.2f  KL %.4f  Rho %.4f  Case %s  BitDist %s  Cos %.3f",
                        epoch,
                        progress,
                        loss_meter.avg,
                        sharp_meter.avg,
                        meta.avg_bits.mean().detach().item(),
                        kl_value.detach().item() if isinstance(kl_value, torch.Tensor) else float(kl_value),
                        rho,
                        case_display,
                        bit_distribution,
                        grad_stats.get("cosine", 0.0),
                    )

            self.global_step += 1

    def _move_batch(self, batch):
        images, targets = batch
        images = images.to(self.args.device)
        if self.task == "classification":
            return images, targets.to(self.args.device)
        targets["yolo_targets"] = targets["yolo_targets"].to(self.args.device)
        return images, targets

    def _bit_distribution(self, meta):
        if meta.bit_usage.numel() == 0:
            return {}
        mean_usage = meta.bit_usage.mean(dim=0)
        distribution = {}
        for index, bit in enumerate(self.model.candidate_bits):
            distribution[int(bit)] = round(float(mean_usage[index].item()), 3)
        return distribution

    def _format_case_weights(self, case_weights):
        values = [round(v.detach().item(), 2) for v in case_weights]
        return f"[{', '.join(f'{value:.2f}' for value in values)}]"

    def _format_progress(self, step, total_steps):
        return f"[{step:4d}/{total_steps}]"
