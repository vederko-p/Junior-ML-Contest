
import random
import torch


def test_classif_model(model, ds, n_rows=3):
    n_cols = 3
    n_imgs = n_rows*n_cols
    lbls_out = {v: k for k, v in ds.labels_code.items()}
    images, labels, batch_indxs = prepare_batch(ds, n_rows=n_rows)
    indxs = [batch_indxs[i:i+n_cols] for i in range(0, n_imgs, n_cols)]
    pred = model.forward(images).argmax(axis=1).tolist()
    pred_labels = list(map(lambda x: f'{lbls_out[x]} ({x})', pred))
    lbls = [pred_labels[i:i+n_cols] for i in range(0, n_imgs, n_cols)]
    return indxs, lbls


def prepare_batch(ds, n_rows=3):
    n_cols = 3
    img_size = 128
    channels = 1
    init_batch_size = (0, channels, img_size, img_size)
    data_batch = torch.empty(init_batch_size, dtype=torch.float)
    labels_batch = torch.empty(0, dtype=torch.long)
    ds_indexes = range(len(ds))
    batch_indexes = random.sample(ds_indexes, n_rows*n_cols)
    for indx in batch_indexes:
        sample = ds[indx]
        data_batch = torch.cat([data_batch, sample[0].unsqueeze(0)])
        labels_batch = torch.cat([labels_batch, sample[1].unsqueeze(0)])
    return data_batch, labels_batch, batch_indexes
