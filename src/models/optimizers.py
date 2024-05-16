from copy import copy
import functools
import math
import operator
import torch
from torch.optim import Optimizer

CBP = torch.optim.SGD


class AdaFactor(Optimizer):
    def __init__(
        self,
        params,
        lr=None,
        beta1=0.9,
        beta2=0.999,
        eps1=1e-30,
        eps2=1e-3,
        cliping_threshold=1,
        non_constant_decay=True,
        enable_factorization=True,
        ams_grad=True,
        weight_decay=0,
    ):

        enable_momentum = beta1 != 0
        self.beta1_glob = copy(beta1)
        self.beta2_glob = copy(beta2)
        self.lr_glob = copy(lr)

        beta1 = (
            self.beta1_glob if hasattr(beta1, "__call__") else lambda x: self.beta1_glob
        )
        beta2 = (
            self.beta2_glob if hasattr(beta2, "__call__") else lambda x: self.beta2_glob
        )

        if non_constant_decay:
            ams_grad = False
            if isinstance(self.beta1_glob, float):
                beta1 = (
                    lambda t: self.beta1_glob
                    * (1 - self.beta1_glob ** (t - 1))
                    / (1 - self.beta1_glob**t)
                )
            if isinstance(self.beta2_glob, float):
                beta2 = (
                    lambda t: self.beta2_glob
                    * (1 - self.beta2_glob ** (t - 1))
                    / (1 - self.beta2_glob**t)
                )

        relative_step_size = True

        if lr is None:
            # default value from article
            lr = lambda t: min(1e-2, 1 / math.sqrt(t))

        if isinstance(self.lr_glob, float):
            lr = lambda x: self.lr_glob
            relative_step_size = False

        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps1=eps1,
            eps2=eps2,
            cliping_threshold=cliping_threshold,
            weight_decay=weight_decay,
            ams_grad=ams_grad,
            enable_factorization=enable_factorization,
            enable_momentum=enable_momentum,
            relative_step_size=relative_step_size,
        )

        super(AdaFactor, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdaFactor, self).__setstate__(state)

    def _experimental_reshape(self, shape):
        temp_shape = shape[2:]
        if len(temp_shape) == 1:
            new_shape = (shape[0], shape[1] * shape[2])
        else:
            tmp_div = len(temp_shape) // 2 + len(temp_shape) % 2
            new_shape = (
                shape[0] * functools.reduce(operator.mul, temp_shape[tmp_div:], 1),
                shape[1] * functools.reduce(operator.mul, temp_shape[:tmp_div], 1),
            )
        return new_shape, copy(shape)

    def _check_shape(self, shape):
        """
        output1 - True - algorithm for matrix, False - vector;
        output2 - need reshape
        """
        if len(shape) > 2:
            return True, True
        elif len(shape) == 2:
            return True, False
        elif len(shape) == 2 and (shape[0] == 1 or shape[1] == 1):
            return False, False
        else:
            return False, False

    def _rms(self, x):
        return math.sqrt(torch.mean(x.pow(2)))

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                data_backup = p.data.clone().detach()

                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                is_matrix, is_need_reshape = self._check_shape(grad.size())
                new_shape = p.data.size()
                if is_need_reshape and group["enable_factorization"]:
                    new_shape, old_shape = self._experimental_reshape(p.data.size())
                    grad = grad.view(new_shape)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    if group["enable_momentum"]:
                        state["exp_avg"] = torch.zeros(
                            new_shape, dtype=torch.float32, device=p.grad.device
                        )

                    if is_matrix and group["enable_factorization"]:
                        state["exp_avg_sq_R"] = torch.zeros(
                            (1, new_shape[1]), dtype=torch.float32, device=p.grad.device
                        )
                        state["exp_avg_sq_C"] = torch.zeros(
                            (new_shape[0], 1), dtype=torch.float32, device=p.grad.device
                        )
                    else:
                        state["exp_avg_sq"] = torch.zeros(
                            new_shape, dtype=torch.float32, device=p.grad.device
                        )
                    if group["ams_grad"]:
                        state["exp_avg_sq_hat"] = torch.zeros(
                            new_shape, dtype=torch.float32, device=p.grad.device
                        )

                if group["enable_momentum"]:
                    exp_avg = state["exp_avg"]

                if is_matrix and group["enable_factorization"]:
                    exp_avg_sq_R = state["exp_avg_sq_R"]
                    exp_avg_sq_C = state["exp_avg_sq_C"]
                else:
                    exp_avg_sq = state["exp_avg_sq"]

                if group["ams_grad"]:
                    exp_avg_sq_hat = state["exp_avg_sq_hat"]

                state["step"] += 1
                lr_t = group["lr"](state["step"])
                if group["relative_step_size"]:
                    lr_t *= max(group["eps2"], self._rms(p.data))

                if group["enable_momentum"]:
                    beta1_t = group["beta1"](state["step"])
                    exp_avg.mul_(beta1_t).add_(1 - beta1_t, grad)

                beta2_t = group["beta2"](state["step"])

                if is_matrix and group["enable_factorization"]:
                    exp_avg_sq_R.mul_(beta2_t).add_(
                        1 - beta2_t,
                        torch.sum(
                            torch.mul(grad, grad).add_(group["eps1"]),
                            dim=0,
                            keepdim=True,
                        ),
                    )
                    exp_avg_sq_C.mul_(beta2_t).add_(
                        1 - beta2_t,
                        torch.sum(
                            torch.mul(grad, grad).add_(group["eps1"]),
                            dim=1,
                            keepdim=True,
                        ),
                    )
                    v = torch.mul(exp_avg_sq_C, exp_avg_sq_R).div_(
                        torch.sum(exp_avg_sq_R)
                    )
                else:
                    exp_avg_sq.mul_(beta2_t).addcmul_(1 - beta2_t, grad, grad).add_(
                        (1 - beta2_t) * group["eps1"]
                    )
                    v = exp_avg_sq

                g = grad
                if group["enable_momentum"]:
                    g = torch.div(exp_avg, 1 - beta1_t ** state["step"])

                if group["ams_grad"]:
                    torch.max(exp_avg_sq_hat, v, out=exp_avg_sq_hat)
                    v = exp_avg_sq_hat
                    u = torch.div(
                        g,
                        (torch.div(v, 1 - beta2_t ** state["step"]))
                        .sqrt()
                        .add_(group["eps1"]),
                    )
                else:
                    u = torch.div(g, v.sqrt())

                u.div_(max(1, self._rms(u) / group["cliping_threshold"]))
                p.data.add_(
                    -lr_t
                    * (
                        u.view(old_shape)
                        if is_need_reshape and group["enable_factorization"]
                        else u
                    )
                )

                if group["weight_decay"] != 0:
                    p.data.add_(-group["weight_decay"] * lr_t, data_backup)

        return loss


class RAdam(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        degenerated_to_sgd=False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if (
            isinstance(params, (list, tuple))
            and len(params) > 0
            and isinstance(params[0], dict)
        ):
            for param in params:
                if "betas" in param and (
                    param["betas"][0] != betas[0] or param["betas"][1] != betas[1]
                ):
                    param["buffer"] = [[None, None, None] for _ in range(10)]
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            buffer=[[None, None, None] for _ in range(10)],
        )
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError("RAdam does not support sparse gradients")

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].type_as(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state["step"] += 1
                buffered = group["buffer"][int(state["step"] % 10)]
                if state["step"] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state["step"]
                    beta2_t = beta2 ** state["step"]
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state["step"] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t)
                            * (N_sma - 4)
                            / (N_sma_max - 4)
                            * (N_sma - 2)
                            / N_sma
                            * N_sma_max
                            / (N_sma_max - 2)
                        ) / (1 - beta1 ** state["step"])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state["step"])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group["weight_decay"] != 0:
                        p_data_fp32.add_(
                            -group["weight_decay"] * group["lr"], p_data_fp32
                        )
                    denom = exp_avg_sq.sqrt().add_(group["eps"])
                    p_data_fp32.addcdiv_(-step_size * group["lr"], exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group["weight_decay"] != 0:
                        p_data_fp32.add_(
                            -group["weight_decay"] * group["lr"], p_data_fp32
                        )
                    p_data_fp32.add_(-step_size * group["lr"], exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss


class SRSGD(Optimizer):
    """
    Stochastic gradient descent with Adaptively restarting (200 iters) Nesterov momentum.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): learning rate.
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        iter_count (integer): count the iterations mod 200
    Example:
         >>> optimizer = torch.optim.SRSGD(model.parameters(), lr=0.1, weight_decay=5e-4, iter_count=1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
        >>> iter_count = optimizer.update_iter()
    Formula:
        v_{t+1} = p_t - lr*g_t
        p_{t+1} = v_{t+1} + (iter_count)/(iter_count+3)*(v_{t+1} - v_t)
    """

    def __init__(self, params, lr, weight_decay=0.0, iter_count=1, restarting_iter=100):
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if iter_count < 1:
            raise ValueError("Invalid iter count: {}".format(iter_count))
        if restarting_iter < 1:
            raise ValueError("Invalid iter total: {}".format(restarting_iter))

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            iter_count=iter_count,
            restarting_iter=restarting_iter,
        )
        super(SRSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SRSGD, self).__setstate__(state)

    def update_iter(self):
        idx = 1
        for group in self.param_groups:
            if idx == 1:
                group["iter_count"] += 1
                if group["iter_count"] >= group["restarting_iter"]:
                    group["iter_count"] = 1
            idx += 1
        return group["iter_count"], group["restarting_iter"]

    def step(self, closure=None):
        """
        Perform a single optimization step.
        Arguments: closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = (group["iter_count"] - 1.0) / (group["iter_count"] + 2.0)
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                param_state = self.state[p]

                if "momentum_buffer" not in param_state:
                    buf0 = param_state["momentum_buffer"] = torch.clone(p.data).detach()
                else:
                    buf0 = param_state["momentum_buffer"]

                buf1 = p.data - group["lr"] * d_p
                p.data = buf1 + momentum * (buf1 - buf0)
                param_state["momentum_buffer"] = buf1
        return loss


class SGD_GC(Optimizer):

    def __init__(
        self,
        params,
        lr,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
    ):
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD_GC, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_GC, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                # GC operation for Conv layers and FC layers
                if len(list(d_p.size())) > 1:
                    d_p.add_(
                        -d_p.mean(
                            dim=tuple(range(1, len(list(d_p.size())))), keepdim=True
                        )
                    )

                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group["lr"], d_p)

        return loss


class Ranger(Optimizer):

    def __init__(
        self,
        params,
        lr=1e-3,  # lr
        alpha=0.5,
        k=6,
        N_sma_threshhold=5,  # Ranger options
        betas=(0.95, 0.999),
        eps=1e-5,
        weight_decay=0,  # Adam options
        # Gradient centralization on or off, applied to conv layers only or conv + fc layers
        use_gc=True,
        gc_conv_only=False,
    ):

        # parameter checks
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid slow update rate: {alpha}")
        if not 1 <= k:
            raise ValueError(f"Invalid lookahead steps: {k}")
        if not lr > 0:
            raise ValueError(f"Invalid Learning Rate: {lr}")
        if not eps > 0:
            raise ValueError(f"Invalid eps: {eps}")

        # parameter comments:
        # beta1 (momentum) of .95 seems to work better than .90...
        # N_sma_threshold of 5 seems better in testing than 4.
        # In both cases, worth testing on your dataset (.90 vs .95, 4 vs 5) to make sure which works best for you.

        # prep defaults and init torch.optim base
        defaults = dict(
            lr=lr,
            alpha=alpha,
            k=k,
            step_counter=0,
            betas=betas,
            N_sma_threshhold=N_sma_threshhold,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

        # adjustable threshold
        self.N_sma_threshhold = N_sma_threshhold

        # look ahead params

        self.alpha = alpha
        self.k = k

        # radam buffer for state
        self.radam_buffer = [[None, None, None] for ind in range(10)]

        # gc on or off
        self.use_gc = use_gc

        # level of gradient centralization
        self.gc_gradient_threshold = 3 if gc_conv_only else 1

        print(
            f"Ranger optimizer loaded. \nGradient Centralization usage = {self.use_gc}"
        )
        if self.use_gc and self.gc_gradient_threshold == 1:
            print(f"GC applied to both conv and fc layers")
        elif self.use_gc and self.gc_gradient_threshold == 3:
            print(f"GC applied to conv layers only")

    def __setstate__(self, state):
        print("set state called")
        super(Ranger, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        # note - below is commented out b/c I have other work that passes back the loss as a float, and thus not a callable closure.
        # Uncomment if you need to use the actual closure...

        # if closure is not None:
        # loss = closure()

        # Evaluate averages and grad, update param tensors
        for group in self.param_groups:

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()

                if grad.is_sparse:
                    raise RuntimeError(
                        "Ranger optimizer does not support sparse gradients"
                    )

                p_data_fp32 = p.data.float()

                state = self.state[p]  # get state dict for this param

                if (
                    len(state) == 0
                ):  # if first time to run...init dictionary with our desired entries
                    # if self.first_run_check==0:
                    # self.first_run_check=1
                    # print("Initializing slow buffer...should not see this at load from saved model!")
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)

                    # look ahead weight storage now in state dict
                    state["slow_buffer"] = torch.empty_like(p.data)
                    state["slow_buffer"].copy_(p.data)

                else:
                    state["exp_avg"] = state["exp_avg"].type_as(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(p_data_fp32)

                # begin computations
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                # GC operation for Conv layers and FC layers
                if grad.dim() > self.gc_gradient_threshold:
                    grad.add_(-grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True))

                state["step"] += 1

                # compute variance mov avg
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                # compute mean moving avg
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                buffered = self.radam_buffer[int(state["step"] % 10)]

                if state["step"] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state["step"]
                    beta2_t = beta2 ** state["step"]
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state["step"] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma
                    if N_sma > self.N_sma_threshhold:
                        step_size = math.sqrt(
                            (1 - beta2_t)
                            * (N_sma - 4)
                            / (N_sma_max - 4)
                            * (N_sma - 2)
                            / N_sma
                            * N_sma_max
                            / (N_sma_max - 2)
                        ) / (1 - beta1 ** state["step"])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state["step"])
                    buffered[2] = step_size

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(-group["weight_decay"] * group["lr"], p_data_fp32)

                # apply lr
                if N_sma > self.N_sma_threshhold:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])
                    p_data_fp32.addcdiv_(-step_size * group["lr"], exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size * group["lr"], exp_avg)

                p.data.copy_(p_data_fp32)

                # integrated look ahead...
                # we do it at the param level instead of group level
                if state["step"] % group["k"] == 0:
                    # get access to slow param tensor
                    slow_p = state["slow_buffer"]
                    # (fast weights - slow weights) * alpha
                    slow_p.add_(self.alpha, p.data - slow_p)
                    # copy interpolated weights to RAdam param tensor
                    p.data.copy_(slow_p)

        return loss


class SMD(Optimizer):
    def __init__(self, params, lr=0.01):
        super(SMD, self).__init__(params, {})
        self.lr = lr

    def step(self, closure=None):
        loss = closure()

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                # Compute Hessian-vector product
                def hvp_product(vector):
                    hvp = torch.autograd.grad(grad, p, vector, retain_graph=True)
                    return hvp[0]

                # Example: Update parameter using Hessian-vector product
                p.data.add_(-self.lr, hvp_product(grad))

        return loss


class DoWG(Optimizer):
    """Implements DoWG optimization algorithm.

    Args:
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-4). Also used as the default squared distance estimate.
    """

    def __init__(self, params, lr=None, eps=1e-4):
        defaults = dict(eps=eps)
        self.eps = eps
        super(DoWG, self).__init__(params, defaults)

    def step(self):
        """Performs a single optimization step."""
        state = self.state

        with torch.no_grad():
            device = self.param_groups[0]["params"][0].device

            # Initialize state variables if needed
            if "rt2" not in state:
                state["rt2"] = torch.Tensor([self.eps]).to(device)
            if "vt" not in state:
                state["vt"] = torch.Tensor([0]).to(device)

            grad_sq_norm = torch.Tensor([0]).to(device)
            curr_d2 = torch.Tensor([0]).to(device)

            for idx, group in enumerate(self.param_groups):
                group_state = state[idx]
                if "x0" not in group_state:
                    group_state["x0"] = [torch.clone(p) for p in group["params"]]

                grad_sq_norm += torch.stack(
                    [(p.grad**2).sum() for p in group["params"]]
                ).sum()
                curr_d2 += torch.stack(
                    [
                        ((p - p0) ** 2).sum()
                        for p, p0 in zip(group["params"], group_state["x0"])
                    ]
                ).sum()

            state["rt2"] = torch.max(state["rt2"], curr_d2)
            state["vt"] += state["rt2"] * grad_sq_norm
            rt2, vt = state["rt2"], state["vt"]

            for group in self.param_groups:
                for p in group["params"]:
                    gt_hat = rt2 * p.grad.data
                    denom = torch.sqrt(vt).add_(group["eps"])
                    p.data.addcdiv_(gt_hat, denom, value=-1.0)
        return None


class FirstOrderGlobalUPGD(Optimizer):
    """Utility-based Perturbed Gradient Descent by Elsayed & Mahmood. Implementation taken from https://github.com/mohmdelsayed/upgd/blob/main/core/optim/weight/upgd/first_order.py#L74."""

    def __init__(self, params, lr=1e-5, weight_decay=0.0, beta_utility=0.0, sigma=1.0):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            beta_utility=beta_utility,
            sigma=sigma,
        )
        super(FirstOrderGlobalUPGD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        global_max_util = torch.tensor(-torch.inf)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(p.data)
                state["step"] += 1
                avg_utility = state["avg_utility"]
                avg_utility.mul_(group["beta_utility"]).add_(
                    -p.grad.data * p.data, alpha=1 - group["beta_utility"]
                )
                current_util_max = avg_utility.max()
                if current_util_max > global_max_util:
                    global_max_util = current_util_max

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                bias_correction = 1 - group["beta_utility"] ** state["step"]
                noise = torch.randn_like(p.grad) * group["sigma"]
                scaled_utility = torch.sigmoid_(
                    (state["avg_utility"] / bias_correction) / global_max_util
                )
                p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
                    (p.grad.data + noise) * (1 - scaled_utility),
                    alpha=-group["lr"],
                )
        return loss


class WNGrad(Optimizer):
    def __init__(self, params, lr=0.05) -> None:
        self.scale = lr
        self.scale_sq = lr**2
        self._step_count = 1
        defaults = dict(scale=lr)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        self._step_count += 1

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad.data
                state = self.state[param]
                if len(state) == 0:
                    state = self.initial_state(state, param)

                state["b"] = (
                    state["b"]
                    + self.scale_sq
                    * torch.dot(grad.view(-1), grad.view(-1))
                    / state["b"]
                )
                lr = self.scale / state["b"]
                param.data.sub_(lr * grad)
            group["lr"] = lr
        return loss

    def initial_state(self, state, param):
        assert len(state) == 0
        state["b"] = param.new_ones(1)
        return state


class COCOB(Optimizer):
    def __init__(self, params, lr: float = 100.0, eps: float = 1e-8):
        self.alpha = lr
        self.eps = eps
        defaults = dict(alpha=lr, eps=eps)
        super(COCOB, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # optimize for each parameter group
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = -param.grad.data
                state = self.state[param]

                if len(state) == 0:
                    state = self.initial_state(state, param)

                initial_weight = state["initial_weight"]
                reward = state["reward"]
                bet = state["bet"]
                grads_sum = state["grads_sum"]
                absolute_grads_sum = state["absolute_grads_sum"]
                max_observed_scale = state["max_observed_scale"]

                # update parameters
                abs_grad = torch.abs(grad)
                max_observed_scale = torch.max(max_observed_scale, abs_grad)
                absolute_grads_sum += abs_grad
                grads_sum += grad

                win_amount = bet * grad
                reward = torch.max(reward + win_amount, torch.zeros_like(reward))
                bet_fraction = grads_sum / (
                    max_observed_scale
                    * (
                        torch.max(
                            absolute_grads_sum + max_observed_scale,
                            self.alpha * max_observed_scale,
                        )
                    )
                )
                bet = bet_fraction * (max_observed_scale + reward)

                # set new state
                param.data = initial_weight + bet
                state["grads_sum"] = grads_sum
                state["absolute_grads_sum"] = absolute_grads_sum
                state["max_observed_scale"] = max_observed_scale
                state["reward"] = reward
                state["bet"] = bet

        return loss

    def initial_state(self, state, param):
        assert len(state) == 0

        state["initial_weight"] = param.data
        state["reward"] = param.new_zeros(param.shape)
        state["bet"] = param.new_zeros(param.shape)
        state["grads_sum"] = param.new_zeros(param.shape)
        state["absolute_grads_sum"] = param.new_zeros(param.shape)
        state["max_observed_scale"] = self.eps * param.new_ones(param.shape)

        return state
