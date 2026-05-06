import torch

def fgsm_attack(data, epsilon, gradient):
    sign = gradient.sign()
    perturbed = data + epsilon * sign
    return torch.clamp(perturbed, 0, 1)