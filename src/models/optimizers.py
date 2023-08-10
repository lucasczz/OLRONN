import torch


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

        # optimize for each parameter groups
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


class ECOCOB(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 100.0, eps: float = 1e-8):
        self.alpha = lr
        self.eps = eps
        defaults = dict(alpha=lr, eps=eps)
        super(ECOCOB, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # optimize for each parameter groups
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = -param.grad.data
                state = self.state[param]

                if len(state) == 0:
                    state = self.initial_state(state, param)

                initial_weight = state["initial_weight"]
                reward = state["reward"] * 0.999
                bet = state["bet"]
                grads_sum = state["grads_sum"] * 0.999
                absolute_grads_sum = state["absolute_grads_sum"] * 0.999
                max_observed_scale = state["max_observed_scale"] * 0.999

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


class GCOCOB(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 100.0, eps: float = 1e-8):
        self.alpha = lr
        self.eps = eps
        defaults = dict(alpha=lr, eps=eps)
        super(GCOCOB, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # optimize for each parameter groups
        for group in self.param_groups:
            if "max_observed_scale" not in group:
                group["max_observed_scale"] = torch.tensor(self.eps)
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
                max_observed_scale = group["max_observed_scale"]

                # update parameters
                abs_grad = torch.abs(grad)
                max_observed_scale = torch.max(max_observed_scale, abs_grad.max())
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

        return state
