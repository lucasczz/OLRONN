import torch



# UPGD: Utilited-based Perturbed Gradient Descent: variation 2 (utility controls gradient)
class FirstOrderGlobalUPGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, weight_decay=0.0, beta_utility=0.0, sigma=1.0):
        names, params = zip(*params)
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            beta_utility=beta_utility,
            sigma=sigma,
            names=names,
        )
        super(FirstOrderGlobalUPGD, self).__init__(params, defaults)

    def step(self, loss):
        global_max_util = torch.tensor(-torch.inf)
        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if "gate" in name:
                    continue
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
            for name, p in zip(group["names"], group["params"]):
                if "gate" in name:
                    continue
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


class WNGrad(torch.optim.Optimizer):
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


class COCOB(torch.optim.Optimizer):
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
