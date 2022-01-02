#  from deeplearning.reporter import TensorboardHparamReporter
from tinydl.metric import DummyMetric
from torch.utils.tensorboard import SummaryWriter

LOGDIR = "runs/"

hparam = {
    "batchsize": [128, 1024],
    "lr": [2e-2, 2e-4]
}


def train(writer, lr, bs, tb_step):

    dummy_metric = DummyMetric()
    dummy_metric.calculate(1, 1)
    metric_value = dummy_metric.value * lr * bs / (tb_step+1)
    writer.add_scalar(
        "dummy-metric", scalar_value=metric_value, global_step=tb_step)
    writer.add_scalar("dummy-metric/2",
                      scalar_value=metric_value/2, global_step=tb_step)

    return metric_value, metric_value/2


def search(hparam):

    for lr in hparam["lr"]:

        for bs in hparam["batchsize"]:

            writer = SummaryWriter(log_dir=f"{LOGDIR}/dummy_bs-{bs}_lr-{lr}")
            tb_step = 0

            for _ in range(200):
                metric_value, metric_value_half = train(
                    writer, lr, bs, tb_step)

                writer.add_hparams(
                    hparam_dict={"learningrate": lr, "batchsize": bs},
                    metric_dict={"metric": metric_value,
                                 "metric/2": metric_value_half}
                )
                tb_step += 1
                print(f"Epoch [{_ + 1}/200 - metric: {metric_value}")
