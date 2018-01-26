import pyffe
from pyffe.models import mAlexNet, AlexNet

dataset_prefix = 'splits'

PKLot = pyffe.Dataset(dataset_prefix + '/PKLot')
CNRParkAB = pyffe.Dataset(dataset_prefix + '/CNRParkAB')
CNRExt = pyffe.Dataset(dataset_prefix + '/CNRPark-EXT')
Combined = pyffe.Dataset(dataset_prefix + '/Combined')

input_format = pyffe.InputFormat(
    new_width=256,
    new_height=256,
    crop_size=224,
    scale=1. / 256,
    mirror=True
)

model = mAlexNet(input_format, num_output=2, batch_sizes=[64, 100])
bigmodel = AlexNet(input_format, num_output=2, batch_sizes=[64, 50])

solver = pyffe.Solver(
    base_lr=0.0008,
    train_epochs=6,
    lr_policy="step",
    gamma=0.75,
    stepsize_epochs=2,
    val_interval_epochs=0.15,
    val_epochs=0.1,
    display_per_epoch=30,
    snapshot_interval_epochs=0.15,
)
exps = [
    # exp 1.1 and 1.2
    pyffe.Experiment(model, solver, PKLot.train, val=[PKLot.val, CNRExt.val], test=[PKLot.test, CNRExt.test]),
    pyffe.Experiment(bigmodel, solver, PKLot.train, val=[PKLot.val, CNRExt.val], test=[PKLot.test, CNRExt.test]),
    
    # exp 2.1 and 2.2
    pyffe.Experiment(model, solver, CNRParkAB.all, val=[CNRExt.val, PKLot.val], test=[CNRExt.test, PKLot.test]),
    pyffe.Experiment(bigmodel, solver, CNRParkAB.all, val=[CNRExt.val, PKLot.val], test=[CNRExt.test, PKLot.test]),

    # exp 3.1 and 3.2
    pyffe.Experiment(model, solver, Combined.CNRParkAB_Ext_train, val=[CNRExt.val, PKLot.val], test=[CNRExt.test, PKLot.test] ),
    pyffe.Experiment(bigmodel, solver, Combined.CNRParkAB_Ext_train, val=[CNRExt.val, PKLot.val], test=[CNRExt.test, PKLot.test] ),

    # exp 4.1 and 4.2
    pyffe.Experiment(model, solver, Combined.CNRParkAB_Ext_train_C1C8, val=[CNRExt.val, PKLot.val], test=[CNRExt.test, PKLot.test] ),
    pyffe.Experiment(bigmodel, solver, Combined.CNRParkAB_Ext_train_C1C8, val=[CNRExt.val, PKLot.val], test=[CNRExt.test, PKLot.test] ),
        
    # Inter-weather experiments
    pyffe.Experiment(model, solver, CNRExt.sunny, val=[CNRExt.overcast, CNRExt.rainy, PKLot.val], test=[CNRExt.overcast, CNRExt.rainy, PKLot.test]),
    pyffe.Experiment(model, solver, CNRExt.overcast, val=[CNRExt.sunny, CNRExt.rainy, PKLot.val], test=[CNRExt.sunny, CNRExt.rainy, PKLot.test]),
    pyffe.Experiment(model, solver, CNRExt.rainy, val=[CNRExt.sunny, CNRExt.overcast, PKLot.val], test=[CNRExt.sunny, CNRExt.overcast, PKLot.test]),

    # Inter-camera experiments
    pyffe.Experiment(model, solver, CNRExt.camera8, 
        val=[CNRExt.camera1, 
             CNRExt.camera2,
             CNRExt.camera3,
             CNRExt.camera4,
             CNRExt.camera5,
             CNRExt.camera6,
             CNRExt.camera7,
             PKLot.val,
             CNRExt.camera9],
        test=[CNRExt.camera1, 
             CNRExt.camera2,
             CNRExt.camera3,
             CNRExt.camera4,
             CNRExt.camera5,
             CNRExt.camera6,
             CNRExt.camera7,
             PKLot.test,
             CNRExt.camera9]),
             
    pyffe.Experiment(model, solver, CNRExt.camera1,
        val=[PKLot.val,
             CNRExt.camera2, 
             CNRExt.camera3,
             CNRExt.camera4,
             CNRExt.camera5,
             CNRExt.camera6,
             CNRExt.camera7,
             CNRExt.camera8,
             CNRExt.camera9],
        test=[PKLot.test,
             CNRExt.camera2, 
             CNRExt.camera3,
             CNRExt.camera4,
             CNRExt.camera5,
             CNRExt.camera6,
             CNRExt.camera7,
             CNRExt.camera8,
             CNRExt.camera9]),
]

for exp in exps:
    exp.setup('runs/')
    exp.run(plot=False) # run without live plot

pyffe.summarize(exps).to_csv('results.csv')


