from vizualization import Visualizer
from datasets import get_datamodule
from configs import get_config

if __name__ == "__main__":


    DATASET = "waymo"
    ITER_OVER = "test"

    cfg = get_config("../configs/models/slim.yaml", dataset=DATASET)
    data_module = get_datamodule(name=DATASET, data_path=None, cfg=cfg)
    data_module.setup()

    if ITER_OVER == "train":
        dl = data_module.train_dataloader()
    elif ITER_OVER == "test":
        dl = data_module.test_dataloader()
    else:
        raise ValueError()

    dl.num_workers = 0

    # Model
    # Loading config
    #cfg = get_config("../configs/slim.yaml", dataset="waymo")
    #model = SLIM(config=cfg, dataset="waymo")
    #model = model.load_from_checkpoint("../models/waymo12k.ckpt")
    # Wrap the dataloader into visualizer
    dl = Visualizer(dl, visualize="seg3d", model=None)

    for idx, (x, flow, T_gt) in enumerate(dl):
        continue

