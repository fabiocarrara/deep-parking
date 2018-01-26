import pyffe
from pyffe.models import mAlexNet

PKLot = pyffe.Dataset('splits/PKLot')

input_format = pyffe.InputFormat(
    new_width=256,
    new_height=256,
    crop_size=224,
    scale=1. / 256,
    mirror=True
)

model = mAlexNet(input_format, num_output=2, batch_sizes=[64, 100])

solver = pyffe.Solver(
    base_lr=0.01,
    train_epochs=18,
    lr_policy="step",
    gamma=0.5,
    stepsize_epochs=6,
    val_interval_epochs=0.5,
    val_epochs=0.05,
    display_per_epoch=30,
    snapshot_interval_epochs=0.5
)
exps = [
    # original PKLot experiments, single camera training
    pyffe.Experiment(model, solver, PKLot.UFPR05_train, val=[PKLot.UFPR04_train, PKLot.UFPR05_train, PKLot.PUC_train], test=[PKLot.UFPR04_test, PKLot.UFPR05_test, PKLot.PUC_test]),
    pyffe.Experiment(model, solver, PKLot.UFPR04_train, val=[PKLot.UFPR04_train, PKLot.UFPR05_train, PKLot.PUC_train], test=[PKLot.UFPR04_test, PKLot.UFPR05_test, PKLot.PUC_test]),
    pyffe.Experiment(model, solver, PKLot.PUC_train, val=[PKLot.UFPR04_train, PKLot.UFPR05_train, PKLot.PUC_train], test=[PKLot.UFPR04_test, PKLot.UFPR05_test, PKLot.PUC_test]),
]

for exp in exps:
    exp.setup('runs_pklot/')

for exp in exps:
    exp.run(plot=False)

pyffe.summarize(exps).to_csv('pklot_results.csv')


