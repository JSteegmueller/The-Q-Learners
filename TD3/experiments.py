from HyperHyper import HyperParams


def hy_update_every(i: int) -> HyperParams:
    hyper = HyperParams()
    i = i * 2 - 1
    hyper.agent.update_target_every = 2 ** i
    return hyper


def hy_learning_rate(i: int) -> HyperParams:
    hyper = HyperParams()
    lr = 1 * (10 ** -i)
    hyper.agent.learning_rate_actor = lr
    hyper.agent.learning_rate_critic = lr
    return hyper


def hy_hidden_size(i: int) -> HyperParams:
    hyper = HyperParams()
    i = i + 4
    hyper.agent.hidden_sizes = 2 ** i
    return hyper


def hy_tau(i: int) -> HyperParams:
    hyper = HyperParams()
    hyper.agent.tau = 1 * (10 ** -i)
    return hyper


def hy_discount(i: int) -> HyperParams:
    hyper = HyperParams()
    hyper.agent.discount = 1 * (10 ** (-i + 1))
    return hyper


def hy_noise(i: int) -> HyperParams:
    hyper = HyperParams()
    hyper.agent.explore_noise = 0.2 * (10 ** (-i + 2))
    return hyper


def hy_random_timesteps(i: int) -> HyperParams:
    hyper = HyperParams()
    hyper.agent.random_timesteps = 10 ** (i + 3)
    return hyper


def hy_buffer_threshold(i: int) -> HyperParams:
    hyper = HyperParams()
    hyper.agent.buffer_threshold = 10 ** (i + 3)
    return hyper
