import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import log_softmax

use_cuda = True

def cross_entropy_for_onehot(logits, y):
    # Prediction should be logits instead of probs
    return torch.mean(torch.sum(-y * log_softmax(logits, dim=-1), 1))
def cross_entropy_loss(output, y):
    log_softmax_output = torch.log_softmax(output, dim=1)
    loss = torch.mean(torch.sum(log_softmax_output * y, dim=1))
    return loss

### Do not modif the following codes
class ctx_noparamgrad(object):
    def __init__(self, module):
        self.prev_grad_state = get_param_grad_state(module)
        self.module = module
        set_param_grad_off(module)

    def __enter__(self):
        pass

    def __exit__(self, *args):
        set_param_grad_state(self.module, self.prev_grad_state)
        return False
        
def get_param_grad_state(module):
    return {param: param.requires_grad for param in module.parameters()}

def set_param_grad_off(module):
    for param in module.parameters():
        param.requires_grad = False

def set_param_grad_state(module, grad_state):
    for param in module.parameters():
        param.requires_grad = grad_state[param]
### Ends


### PGD Attack
class PGDAttack():
    def __init__(self, attack_step=10, eps=8 / 255, alpha=2 / 255, loss_type='ce', targeted=True, attack_target=1, kappa=0,
                 num_classes=10):
        '''
        attack_step: number of PGD iterations
        eps: attack budget
        alpha: PGD attack step size
        '''
        ### Your code here
        self.attack_step = attack_step
        self.eps = eps
        self.alpha = alpha
        self.targeted = targeted
        self.num_classes = num_classes
        self.attack_target = attack_target
        self.device = torch.device("cuda" if use_cuda else "cpu")
        # confident threshold kappa
        self.kappa = kappa
        if loss_type == 'ce':
            self.loss = self.ce_loss
        elif loss_type == 'cw':
            self.loss = self.cw_loss
        pass
        ### Your code ends

    def ce_loss(self, logits, y):
        ### Your code here
        # print(f"logits size: {logits.shape[0]}")
        # print(logits)
        # print("-----------------------")
        # print(f"y size: {y.shape[0]}")
        # print(y)
        max_logits = torch.max(logits, dim=1, keepdim=True)[0]
        stable_logits = logits - max_logits
        exp_logits = torch.exp(stable_logits)
        log_sum_exp = torch.log(torch.sum(exp_logits, dim=1, keepdim=True))
        # print("log_sum_exp")
        log_probs = stable_logits - log_sum_exp
        # print("log_probs")
        nll_loss = -log_probs.gather(1, y.view(-1, 1))
        return nll_loss.mean()
        ### Your code ends

    def cw_loss(self, logits, y):
        ### Your code here
        # print(f"logits size: {logits.shape[0]}")
        # print(logits)
        # print("-----------------------")
        # print(f"y size: {y.shape[0]}")
        # print(y)
        y_onehot = F.one_hot(y, self.num_classes).float()
        correct_logits = torch.sum(y_onehot * logits, dim=1)
        other_logits = torch.max((1 - y_onehot) * logits - y_onehot * 1e8 * 1.0, dim=1)[0]
        if self.targeted:
            # print("target:")
            # print(self.attack_target)
            loss = torch.clamp(other_logits - correct_logits + self.kappa, min=0.)
        else:
            loss = torch.clamp(correct_logits - other_logits + self.kappa, min=0.)
        loss = -torch.mean(loss)
        return loss
        ### Your code ends

    def perturb(self, model: nn.Module, X, y):
        delta = torch.zeros_like(X).to(self.device)
        ori_input = X.data.to(self.device)
        adv_input = X.clone().requires_grad_(True).to(self.device)
        model.to(self.device)
        ### Your code here
        model.eval()
        for i in range(self.attack_step):
            adv_input.requires_grad = True
            output = model(adv_input)
            model.zero_grad()
            loss = self.loss(output, y.to(self.device))
            loss.backward()

            grad = adv_input.grad.data
            adv_input = adv_input + self.alpha * torch.sign(grad)
            if self.attack_step == 1:
                delta = self.eps * torch.sign(grad)
            else:
                delta = torch.clamp(adv_input - ori_input, min=-self.eps, max=self.eps)
            adv_input = torch.clamp(ori_input + delta, min=0, max=1).detach_()
        ### Your code ends
        return delta.to(self.device)


### FGSMAttack
'''
Technically you can transform your PGDAttack to FGSM Attack by controling parameters like `attack_step`. 
If you do that, you do not need to implement FGSM in this class.
'''
class FGSMAttack():
    def __init__(self, eps=8 / 255, loss_type='ce', targeted=True, num_classes=10):
        pass

    def perturb(self, model: nn.Module, X, y):
        delta = torch.ones_like(X)
        ### Your code here
        # here I just implement FGSM by changeing the PGD iteration step

        ### Your code ends
        return delta
