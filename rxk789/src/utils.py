import torch
import torch.nn.functional as F


def entropy(probs):
    """
    Function to calculate the entropy of a probability distribution.
    H(p) = - sum(p * log(p))
    """
    e=1e-12
    log_probs = torch.log(probs + e)  # for avoiding Avoid log(0)
    return -torch.sum(probs * log_probs, dim=1).mean()

def mutual_info_ent(logits, x, device):
    """
    using Monte Carlo estimation it Estimates the mutual information I(y; x) and  .

    Parameters:
        logits: Logits output from the encoder model for unlabeled data.
        x: Input unlabeled data (to approximate q(x)).
        device: The device (CPU/GPU) for computation.

    Returns:
        mi: Estimated mutual information.
        conditional_entropy : Estimated average conditional entropy
    """
    q_y_x = F.softmax(logits, dim=1)  # to calculate probability  q(y|x)
    
    # Conditional entropy: H(q(y|x))
    log_q_y_x = torch.log(q_y_x + 1e-12)
    cond_ent=-torch.sum(q_y_x * log_q_y_x, dim=1)
    conditional_entropy = cond_ent.mean() # calculating average conditional entropy

    # calculating Marginal push forward distribution: q(y) = E_x[q(y|x)]
    q_y = q_y_x.mean(dim=0)  # Average across the batch for marginal probabilities

    # Marginal entropy: H(q(y))
    log_q_y = torch.log(q_y + 1e-12)
    marginal_entropy = -torch.sum(q_y * log_q_y)

    # Mutual Information: I(y; x) = H(q(y)) - E_x[H(q(y|x))]
    mi = marginal_entropy - conditional_entropy

    return mi, conditional_entropy


def m2_loss_labeled(recon_x, x, mean, log_var, y_pred, y_true, alpha=1):
    """
    Computing the M2 loss for labeled data:
    - Reconstruction loss
    - KL divergence for continuous latent variables
    - Cross-entropy loss for label prediction
    """
    batch_size = x.size(0)

    # Estimating the KL Divergence for continuous latent variables
    kl_cont_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / batch_size

    # Estimating the Reconstruction Loss
    recon_loss = F.mse_loss(recon_x, x, reduction="sum") / batch_size


    # calculating the Cross-Entropy Loss for labels
    ce_loss = F.cross_entropy(y_pred, y_true)

    # computing the Combined Loss
    return recon_loss + kl_cont_loss + alpha * ce_loss

def m2_loss_unlabeled(recon_x, x, mean, log_var, y_pred, alpha=1.0):
    """
    Computes the M2 loss for unlabeled data:
    - Reconstruction loss
    - KL divergence for continuous latent variables
    - KL divergence for discrete latent variables
    """
    batch_size = x.size(0)

    # computing Reconstruction Loss
    recon_loss = F.mse_loss(recon_x, x, reduction="sum") / batch_size

    # close form solution KL Divergence for continuous latent variables
    kl_cont_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / batch_size

    # KL Divergence for discrete latent variable y
    num_classes = y_pred.size(1)
    prior_y = torch.full_like(y_pred, 1.0 / num_classes)  # Uniform prior
    kl_disc_loss = F.kl_div(F.log_softmax(y_pred, dim=1), prior_y, reduction="batchmean")

    # Combined Loss
    return recon_loss + kl_cont_loss + kl_disc_loss

def optimized_ELBO_labeled(recon_x, x, mean, log_var, y_pred, y_true, alpha=1.0, optimized_label_weight=0.001):
    """
    Computes the optimized ELBO for labeled data:
    - Reconstruction loss
    - KL divergence for continuous latent variables
    - KL divergence for discrete latent variables
    - KL divergence between  emperical and q(y/x) 
    - Cross-entropy loss
    """
    batch_size = x.size(0)

    # Reconstruction Loss
    recon_loss = F.mse_loss(recon_x, x, reduction="sum") / batch_size

    # KL Divergence for continuous latent variables
    kl_cont_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / batch_size

    # KL Divergence for discrete latent variable y
    num_classes = y_pred.size(1)
    prior_y = torch.full_like(y_pred, 1.0 / num_classes)  # Uniform prior
    kl_disc_loss = F.kl_div(F.log_softmax(y_pred, dim=1), prior_y, reduction="batchmean")

    # KL Divergence for optimizeded labels
    optimized_labels = torch.ones_like(y_pred) * optimized_label_weight / num_classes
    # desining  the emperical distribution 
    optimized_labels.scatter_(1, y_true.unsqueeze(1), 1 - optimized_label_weight)
    # coputing KL Div between the emperical distribution and predicted distribution
    kl_optimized_loss = F.kl_div(F.log_softmax(y_pred, dim=1), optimized_labels, reduction="batchmean")
    ce_loss = F.cross_entropy(y_pred, y_true)
    # Combined Loss
    return recon_loss + kl_cont_loss  + kl_optimized_loss + alpha * ce_loss + kl_disc_loss
'''
def optimized_ELBO_unlabeled(recon_x, x, mean, log_var, y_pred, alpha=1.0):
    """
    Computes the optimized ELBO for unlabeled data:
    - Reconstruction loss
    - KL divergence for continuous latent variables
    - KL divergence for discrete latent variables
    """
    batch_size = x.size(0)

    # Reconstruction Loss
    recon_loss = F.mse_loss(recon_x, x, reduction="sum") / batch_size

    # KL Divergence for continuous latent variables
    kl_cont_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / batch_size

    # KL Divergence for discrete latent variable y
    num_classes = y_pred.size(1)
    prior_y = torch.full_like(y_pred, 1.0 / num_classes)  # Uniform prior
    kl_disc_loss = F.kl_div(F.log_softmax(y_pred, dim=1), prior_y, reduction="batchmean")

    # Combined Loss
    return recon_loss + kl_cont_loss #+ kl_disc_loss
'''
def optimized_ELBO_unlabeled(recon_x, x, mean, log_var, logits, alpha=1.0, beta_mi=2, beta_entropy=5, device=None):
    """
    Computes the optimized ELBO for unlabeled data with mutual information and entropy regularization.
    
    Parameters:
        recon_x: Reconstructed input.
        x: Original input.
        mean: Mean of the latent variable z.
        log_var: Log variance of the latent variable z.
        logits: Logits for discrete latent variable y.
        alpha: Weight for the KL divergence term.
        beta_mi: Weight for the mutual information term.
        beta_entropy: Weight for the entropy regularization term.
        device: The device (CPU/GPU) for computation.
    """
    batch_size = x.size(0)

    # Reconstruction Loss
    recon_loss = F.mse_loss(recon_x, x, reduction="sum") / batch_size

    # KL Divergence for continuous latent variables
    kl_cont_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / batch_size

    # KL Divergence for discrete latent variable y
    num_classes = logits.size(1)
    prior_y = torch.full_like(logits, 1.0 / num_classes)  # Uniform prior
    kl_disc_loss = F.kl_div(F.log_softmax(logits, dim=1), prior_y, reduction="batchmean")

    # compute Mutual Information and Entropy
    mi, conditional_entropy = mutual_info_ent(logits, x, device)

    # Combined Loss
    loss = (recon_loss + kl_cont_loss  + alpha * kl_disc_loss - beta_mi * mi  + beta_entropy * conditional_entropy)

    return loss


def validate(model, test_loader, device):
    """
    Validates the model on the test dataset.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y =  y.to(device)
            _, _, _, logits = model(x)
            y_pred = F.softmax(logits, dim=1)
            _, predicted = torch.max(y_pred, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

    accuracy = 100 * correct / total
    return accuracy

def ce_loss_labeled(y_pred, y_true):
    """
    Computes only the cross-entropy loss for labeled data.
    """
    return F.cross_entropy(y_pred, y_true)
