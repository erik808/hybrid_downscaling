import importlib
import data_manager_cmems
import data_generator_cmems
import time

importlib.reload(data_manager_cmems)
importlib.reload(data_generator_cmems)


def test_data_generator():
    dmgr_cmems = data_manager_cmems.DataManagerCMEMS()
    dmgr_cmems.create_training_data(force_rebuild=False)

    dgen_cmems = data_generator_cmems.DataGeneratorCMEMS(
        dm=dmgr_cmems,
        batch_size=4,
        lookback=4,
        mode='train',
        shuffle=True,
        # use_multiprocessing=True,
        # workers=4,
        # max_queue_size=10,
    )

    tic = time.time()
    bx, by = dgen_cmems.__getitem__(0)
    bx, by = dgen_cmems.__getitem__(1)
    bx, by = dgen_cmems.__getitem__(2)
    toc = time.time()
    elapsed = toc - tic
    print(elapsed)
