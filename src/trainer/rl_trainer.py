import numpy as np
import torch


def rl_trainer(recommend_handler, agent, device, bs, epoch):
    for step in np.arange(epoch):
        (
            state,
            action,
            reward,
            next_state,
            prob_behavior,
        ) = recommend_handler.buffer.get_batch(bs)
        prob_target = agent.get_target_probs(
            state.to(device), action.to(torch.long)
        )
        agent.update(
            state.to(device),
            reward.to(device),
            next_state.to(device),
            prob_behavior.to(device),
            prob_target.to(device),
        )

    return (agent, recommend_handler)
