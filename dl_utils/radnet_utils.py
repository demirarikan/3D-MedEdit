import torch
from generative.metrics import FIDMetric
from numpy import std


def subtract_mean(x: torch.Tensor) -> torch.Tensor:
    mean = [0.406, 0.456, 0.485]
    x[:, 0, :, :] -= mean[0]
    x[:, 1, :, :] -= mean[1]
    x[:, 2, :, :] -= mean[2]
    return x


def spatial_average(x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    return x.mean([2, 3], keepdim=keepdim)


def get_features(image, model):
    # If input has just 1 channel, repeat channel to have 3 channels
    if image.shape[1]:
        image = image.repeat(1, 3, 1, 1)

    # Change order from 'RGB' to 'BGR'
    image = image[:, [2, 1, 0], ...]

    # Subtract mean used during training
    image = subtract_mean(image)

    # Get model outputs
    with torch.no_grad():
        feature_image = model.forward(image)
        # flattens the image spatially
        feature_image = spatial_average(feature_image, keepdim=False)

    return feature_image


def compute_fid(model, subset1, subset2, bootstrap=False, indices_=None):

    real_features = []

    real_eval_feats = get_features(subset1, model)
    real_features.append(real_eval_feats)
    real_features = torch.vstack(real_features)

    if bootstrap:
        b = subset2.shape[0]
        sum_ = 0
        fids = []

        for i in range(10):
            synth_features = []

            if indices_ is None:
                # Generate random indices
                indices = torch.randint(b, (b,))
            else:
                indices = indices_[i]

            print(indices)

            # Use the indices to index into the tensor
            sampled_tensor = subset2[indices]

            print(sampled_tensor.shape)  # Prints: torch.Size([b, c, 128, 18])

            # Get the features for the synthetic data
            synth_eval_feats = get_features(sampled_tensor, model)
            synth_features.append(synth_eval_feats)
            synth_features = torch.vstack(synth_features)

            fid = FIDMetric()
            fid_res = fid(synth_features, real_features)
            fid_res = round(fid_res.item(), 2)

            sum_ += fid_res
            fids.append(fid_res)

        mean_fid = sum_ / 10
        std_fid = std(fids)
        return mean_fid, std_fid

    else:
        real_features = []
        synth_features = []

        real_eval_feats = get_features(subset1, model)
        real_features.append(real_eval_feats)

        synth_eval_feats = get_features(subset2, model)
        synth_features.append(synth_eval_feats)

        synth_features = torch.vstack(synth_features)
        real_features = torch.vstack(real_features)

        fid = FIDMetric()
        fid_res = fid(synth_features, real_features)

    return fid_res.item()
