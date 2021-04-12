import plotly
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line
from typing import Optional, List, Iterable
import torch
from torch import jit, nn, optim
from torch.nn import functional as F, Module
import torch.distributions
from torch.distributions.normal import Normal
from torch.distributions.transforms import Transform, TanhTransform
from torch.distributions.transformed_distribution import TransformedDistribution
import numpy as np
import cv2
import argparse
import os
import numpy as np
from torch.distributions.kl import kl_divergence
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from tensorboardX import SummaryWriter


# Plots min, max and mean + standard deviation bars of a population over time
def lineplot(xs, ys_population, title, path='', xaxis='episode'):
    max_colour, mean_colour, std_colour, transparent = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)', 'rgba(0, 0, 0, 0)'

    if isinstance(ys_population[0], list) or isinstance(ys_population[0], tuple):
        ys = np.asarray(ys_population, dtype=np.float32)
        ys_min, ys_max, ys_mean, ys_std, ys_median = ys.min(1), ys.max(1), ys.mean(1), ys.std(1), np.median(ys, 1)
        ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

        trace_max = Scatter(x=xs, y=ys_max, line=Line(color=max_colour, dash='dash'), name='Max')
        trace_upper = Scatter(x=xs, y=ys_upper, line=Line(color=transparent), name='+1 Std. Dev.', showlegend=False)
        trace_mean = Scatter(x=xs, y=ys_mean, fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour),
                             name='Mean')
        trace_lower = Scatter(x=xs, y=ys_lower, fill='tonexty', fillcolor=std_colour, line=Line(color=transparent),
                              name='-1 Std. Dev.', showlegend=False)
        trace_min = Scatter(x=xs, y=ys_min, line=Line(color=max_colour, dash='dash'), name='Min')
        trace_median = Scatter(x=xs, y=ys_median, line=Line(color=max_colour), name='Median')
        data = [trace_upper, trace_mean, trace_lower, trace_min, trace_max, trace_median]
    else:
        data = [Scatter(x=xs, y=ys_population, line=Line(color=mean_colour))]
    plotly.offline.plot({
        'data': data,
        'layout': dict(title=title, xaxis={'title': xaxis}, yaxis={'title': title})
    }, filename=os.path.join(path, title + '.html'), auto_open=False)


def write_video(frames, title, path=''):
    frames = np.multiply(np.stack(frames, axis=0).transpose(0, 2, 3, 1), 255).clip(0, 255).astype(np.uint8)[:, :, :,
             ::-1]  # VideoWrite expects H x W x C in BGR
    _, H, W, _ = frames.shape
    writer = cv2.VideoWriter(os.path.join(path, '%s.mp4' % title), cv2.VideoWriter_fourcc(*'mp4v'), 30., (W, H), True)
    for frame in frames:
        writer.write(frame)
    writer.release()


def imagine_ahead(prev_state, prev_belief, policy, transition_model, planning_horizon=12):
    '''
    imagine_ahead is the function to draw the imaginary tracjectory using the dynamics model, actor, critic.
    Input: current state (posterior), current belief (hidden), policy, transition_model  # torch.Size([50, 30]) torch.Size([50, 200])
    Output: generated trajectory of features includes beliefs, prior_states, prior_means, prior_std_devs
            torch.Size([49, 50, 200]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30])
    '''
    flatten = lambda x: x.view([-1] + list(x.size()[2:]))
    prev_belief = flatten(prev_belief)
    prev_state = flatten(prev_state)

    # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
    T = planning_horizon
    beliefs, prior_states, prior_means, prior_std_devs = [torch.empty(0)] * T, [torch.empty(0)] * T, [
        torch.empty(0)] * T, [torch.empty(0)] * T
    beliefs[0], prior_states[0] = prev_belief, prev_state

    # Loop over time sequence
    for t in range(T - 1):
        _state = prior_states[t]
        actions = policy.get_action(beliefs[t].detach(), _state.detach())
        # Compute belief (deterministic hidden state)
        hidden = transition_model.act_fn(transition_model.fc_embed_state_action(torch.cat([_state, actions], dim=1)))
        beliefs[t + 1] = transition_model.rnn(hidden, beliefs[t])
        # Compute state prior by applying transition dynamics
        hidden = transition_model.act_fn(transition_model.fc_embed_belief_prior(beliefs[t + 1]))
        prior_means[t + 1], _prior_std_dev = torch.chunk(transition_model.fc_state_prior(hidden), 2, dim=1)
        prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + transition_model.min_std_dev
        prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])
        # Return new hidden states
    # imagined_traj = [beliefs, prior_states, prior_means, prior_std_devs]
    imagined_traj = [torch.stack(beliefs[1:], dim=0), torch.stack(prior_states[1:], dim=0),
                     torch.stack(prior_means[1:], dim=0), torch.stack(prior_std_devs[1:], dim=0)]
    return imagined_traj


def lambda_return(imged_reward, value_pred, bootstrap, discount=0.99, lambda_=0.95):
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    next_values = torch.cat([value_pred[1:], bootstrap[None]], 0)
    discount_tensor = discount * torch.ones_like(imged_reward)  # pcont
    inputs = imged_reward + discount_tensor * next_values * (1 - lambda_)
    last = bootstrap
    indices = reversed(range(len(inputs)))
    outputs = []
    for index in indices:
        inp, disc = inputs[index], discount_tensor[index]
        last = inp + disc * lambda_ * last
        outputs.append(last)
    outputs = list(reversed(outputs))
    outputs = torch.stack(outputs, 0)
    returns = outputs
    return returns


class ActivateParameters:
    def __init__(self, modules: Iterable[Module]):
        """
        Context manager to locally Activate the gradients.
        example:
        ```
        with ActivateParameters([module]):
            output_tensor = module(input_tensor)
        ```
        :param modules: iterable of modules. used to call .parameters() to freeze gradients.
        """
        self.modules = modules
        self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

    def __enter__(self):
        for param in get_parameters(self.modules):
            # print(param.requires_grad)
            param.requires_grad = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(get_parameters(self.modules)):
            param.requires_grad = self.param_states[i]


# "get_parameters" and "FreezeParameters" are from the following repo
# https://github.com/juliusfrost/dreamer-pytorch
def get_parameters(modules: Iterable[Module]):
    """
    Given a list of torch modules, returns a list of their parameters.
    :param modules: iterable of modules
    :returns: a list of parameters
    """
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters


class FreezeParameters:
    def __init__(self, modules: Iterable[Module]):
        """
        Context manager to locally freeze gradients.
        In some cases with can speed up computation because gradients aren't calculated for these listed modules.
        example:
        ```
        with FreezeParameters([module]):
            output_tensor = module(input_tensor)
        ```
        :param modules: iterable of modules. used to call .parameters() to freeze gradients.
        """
        self.modules = modules
        self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

    def __enter__(self):
        for param in get_parameters(self.modules):
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(get_parameters(self.modules)):
            param.requires_grad = self.param_states[i]


# Wraps the input tuple for a function to process a time x batch x features sequence in batch x features (assumes one output)
def bottle(f, x_tuple):
    x_sizes = tuple(map(lambda x: x.size(), x_tuple))
    y = f(*map(lambda x: x[0].view(x[1][0] * x[1][1], *x[1][2:]), zip(x_tuple, x_sizes)))
    y_size = y.size()
    output = y.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:])
    return output


class TransitionModel(jit.ScriptModule):
    __constants__ = ['min_std_dev']

    def __init__(self, belief_size, state_size, action_size, hidden_size, embedding_size, activation_function='relu',
                 min_std_dev=0.1):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.min_std_dev = min_std_dev
        self.fc_embed_state_action = nn.Linear(state_size + action_size, belief_size)
        self.rnn = nn.GRUCell(belief_size, belief_size)
        self.fc_embed_belief_prior = nn.Linear(belief_size, hidden_size)
        self.fc_state_prior = nn.Linear(hidden_size, 2 * state_size)
        self.fc_embed_belief_posterior = nn.Linear(belief_size + embedding_size, hidden_size)
        self.fc_state_posterior = nn.Linear(hidden_size, 2 * state_size)
        self.modules = [self.fc_embed_state_action, self.fc_embed_belief_prior, self.fc_state_prior,
                        self.fc_embed_belief_posterior, self.fc_state_posterior]

    # Operates over (previous) state, (previous) actions, (previous) belief, (previous) nonterminals (mask), and (current) observations
    # Diagram of expected inputs and outputs for T = 5 (-x- signifying beginning of output belief/state that gets sliced off):
    # t :  0  1  2  3  4  5
    # o :    -X--X--X--X--X-
    # a : -X--X--X--X--X-
    # n : -X--X--X--X--X-
    # pb: -X-
    # ps: -X-
    # b : -x--X--X--X--X--X-
    # s : -x--X--X--X--X--X-
    @jit.script_method
    def forward(self, prev_state: torch.Tensor, actions: torch.Tensor, prev_belief: torch.Tensor,
                observations: Optional[torch.Tensor] = None, nonterminals: Optional[torch.Tensor] = None) -> List[
        torch.Tensor]:
        '''
        Input: init_belief, init_state:  torch.Size([50, 200]) torch.Size([50, 30])
        Output: beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs
                torch.Size([49, 50, 200]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30])
        '''
        # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
        T = actions.size(0) + 1
        posterior_std_devs = [torch.empty(0)] * T
        posterior_means = [torch.empty(0)] * T
        posterior_states = [torch.empty(0)] * T
        prior_std_devs = [torch.empty(0)] * T
        prior_means = [torch.empty(0)] * T
        prior_states = [torch.empty(0)] * T
        beliefs = [torch.empty(0)] * T
        beliefs[0], prior_states[0], posterior_states[0] = prev_belief, prev_state, prev_state
        # Loop over time sequence
        for t in range(T - 1):
            _state = prior_states[t] if observations is None else posterior_states[
                t]  # Select appropriate previous state
            _state = _state if nonterminals is None else _state * nonterminals[
                t]  # Mask if previous transition was terminal
            # Compute belief (deterministic hidden state)
            hidden = self.act_fn(self.fc_embed_state_action(torch.cat([_state, actions[t]], dim=1)))
            beliefs[t + 1] = self.rnn(hidden, beliefs[t])
            # Compute state prior by applying transition dynamics
            hidden = self.act_fn(self.fc_embed_belief_prior(beliefs[t + 1]))
            prior_means[t + 1], _prior_std_dev = torch.chunk(self.fc_state_prior(hidden), 2, dim=1)
            prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + self.min_std_dev
            prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])
            if observations is not None:
                # Compute state posterior by applying transition dynamics and using current observation
                t_ = t - 1  # Use t_ to deal with different time indexing for observations
                hidden = self.act_fn(
                    self.fc_embed_belief_posterior(torch.cat([beliefs[t + 1], observations[t_ + 1]], dim=1)))
                posterior_means[t + 1], _posterior_std_dev = torch.chunk(self.fc_state_posterior(hidden), 2, dim=1)
                posterior_std_devs[t + 1] = F.softplus(_posterior_std_dev) + self.min_std_dev
                posterior_states[t + 1] = posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.randn_like(
                    posterior_means[t + 1])
        # Return new hidden states
        hidden = [torch.stack(beliefs[1:], dim=0), torch.stack(prior_states[1:], dim=0),
                  torch.stack(prior_means[1:], dim=0), torch.stack(prior_std_devs[1:], dim=0)]
        if observations is not None:
            hidden += [torch.stack(posterior_states[1:], dim=0), torch.stack(posterior_means[1:], dim=0),
                       torch.stack(posterior_std_devs[1:], dim=0)]
        return hidden


class SymbolicObservationModel(jit.ScriptModule):
    def __init__(self, observation_size, belief_size, state_size, embedding_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, embedding_size)
        self.fc3 = nn.Linear(embedding_size, observation_size)
        self.modules = [self.fc1, self.fc2, self.fc3]

    @jit.script_method
    def forward(self, belief, state):
        hidden = self.act_fn(self.fc1(torch.cat([belief, state], dim=1)))
        hidden = self.act_fn(self.fc2(hidden))
        observation = self.fc3(hidden)
        return observation


class VisualObservationModel(jit.ScriptModule):
    __constants__ = ['embedding_size']

    def __init__(self, belief_size, state_size, embedding_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
        self.conv1 = nn.ConvTranspose2d(embedding_size, 128, 5, stride=2)
        self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)
        self.modules = [self.fc1, self.conv1, self.conv2, self.conv3, self.conv4]

    @jit.script_method
    def forward(self, belief, state):
        hidden = self.fc1(torch.cat([belief, state], dim=1))  # No nonlinearity here
        hidden = hidden.view(-1, self.embedding_size, 1, 1)
        hidden = self.act_fn(self.conv1(hidden))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        observation = self.conv4(hidden)
        return observation


def ObservationModel(symbolic, observation_size, belief_size, state_size, embedding_size, activation_function='relu'):
    if symbolic:
        return SymbolicObservationModel(observation_size, belief_size, state_size, embedding_size, activation_function)
    else:
        return VisualObservationModel(belief_size, state_size, embedding_size, activation_function)


class RewardModel(jit.ScriptModule):
    def __init__(self, belief_size, state_size, hidden_size, activation_function='relu'):
        # [--belief-size: 200, --hidden-size: 200, --state-size: 30]
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.modules = [self.fc1, self.fc2, self.fc3]

    @jit.script_method
    def forward(self, belief, state):
        x = torch.cat([belief, state], dim=1)
        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        reward = self.fc3(hidden).squeeze(dim=1)
        return reward


class ValueModel(jit.ScriptModule):
    def __init__(self, belief_size, state_size, hidden_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)
        self.modules = [self.fc1, self.fc2, self.fc3, self.fc4]

    @jit.script_method
    def forward(self, belief, state):
        x = torch.cat([belief, state], dim=1)
        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.act_fn(self.fc3(hidden))
        reward = self.fc4(hidden).squeeze(dim=1)
        return reward


class ActorModel(jit.ScriptModule):
    def __init__(self, belief_size, state_size, hidden_size, action_size, dist='tanh_normal',
                 activation_function='elu', min_std=1e-4, init_std=5, mean_scale=5):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, 2 * action_size)
        self.modules = [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]

        self._dist = dist
        self._min_std = min_std
        self._init_std = init_std
        self._mean_scale = mean_scale

    @jit.script_method
    def forward(self, belief, state):
        raw_init_std = torch.log(torch.exp(self._init_std) - 1)
        x = torch.cat([belief, state], dim=1)
        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.act_fn(self.fc3(hidden))
        hidden = self.act_fn(self.fc4(hidden))
        action = self.fc5(hidden).squeeze(dim=1)

        action_mean, action_std_dev = torch.chunk(action, 2, dim=1)
        action_mean = self._mean_scale * torch.tanh(action_mean / self._mean_scale)
        action_std = F.softplus(action_std_dev + raw_init_std) + self._min_std
        return action_mean, action_std

    def get_action(self, belief, state, det=False):
        action_mean, action_std = self.forward(belief, state)
        dist = Normal(action_mean, action_std)
        dist = TransformedDistribution(dist, TanhBijector())
        dist = torch.distributions.Independent(dist, 1)
        dist = SampleDist(dist)
        if det:
            return dist.mode()
        else:
            return dist.rsample()


class SymbolicEncoder(jit.ScriptModule):
    def __init__(self, observation_size, embedding_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(observation_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, embedding_size)
        self.fc3 = nn.Linear(embedding_size, embedding_size)
        self.modules = [self.fc1, self.fc2, self.fc3]

    @jit.script_method
    def forward(self, observation):
        hidden = self.act_fn(self.fc1(observation))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.fc3(hidden)
        return hidden


class VisualEncoder(jit.ScriptModule):
    __constants__ = ['embedding_size']

    def __init__(self, embedding_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        self.fc = nn.Identity() if embedding_size == 1024 else nn.Linear(1024, embedding_size)
        self.modules = [self.conv1, self.conv2, self.conv3, self.conv4]

    @jit.script_method
    def forward(self, observation):
        hidden = self.act_fn(self.conv1(observation))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        hidden = self.act_fn(self.conv4(hidden))
        hidden = hidden.view(-1, 1024)
        hidden = self.fc(hidden)  # Identity if embedding size is 1024 else linear projection
        return hidden


def Encoder(symbolic, observation_size, embedding_size, activation_function='relu'):
    if symbolic:
        return SymbolicEncoder(observation_size, embedding_size, activation_function)
    else:
        return VisualEncoder(embedding_size, activation_function)


# "atanh", "TanhBijector" and "SampleDist" are from the following repo
# https://github.com/juliusfrost/dreamer-pytorch
def atanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))


class TanhBijector(torch.distributions.Transform):
    def __init__(self):
        super().__init__()
        self.bijective = True

    @property
    def sign(self): return 1.

    def _call(self, x): return torch.tanh(x)

    def _inverse(self, y: torch.Tensor):
        y = torch.where(
            (torch.abs(y) <= 1.),
            torch.clamp(y, -0.99999997, 0.99999997),
            y)
        y = atanh(y)
        return y

    def log_abs_det_jacobian(self, x, y):
        return 2. * (np.log(2) - x - F.softplus(-2. * x))


class SampleDist:
    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return 'SampleDist'

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        sample = self._dist.rsample()
        return torch.mean(sample, 0)

    def mode(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        batch_size = sample.size(1)
        feature_size = sample.size(2)
        indices = torch.argmax(logprob, dim=0).reshape(1, batch_size, 1).expand(1, batch_size, feature_size)
        return torch.gather(sample, 0, indices).squeeze(0)

    def entropy(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        return -torch.mean(logprob, 0)

    def sample(self):
        return self._dist.sample()


class ExperienceReplay():
    def __init__(self, size, symbolic_env, observation_size, action_size, bit_depth, device):
        self.device = device
        self.symbolic_env = symbolic_env
        self.size = size
        self.observations = np.empty((size, observation_size) if symbolic_env else (size, 3, 64, 64),
                                     dtype=np.float32 if symbolic_env else np.uint8)
        self.actions = np.empty((size, action_size), dtype=np.float32)
        self.rewards = np.empty((size,), dtype=np.float32)
        self.nonterminals = np.empty((size, 1), dtype=np.float32)
        self.idx = 0
        self.full = False  # Tracks if memory has been filled/all slots are valid
        self.steps, self.episodes = 0, 0  # Tracks how much experience has been used in total
        self.bit_depth = bit_depth

    def append(self, observation, action, reward, done):
        if self.symbolic_env:
            self.observations[self.idx] = observation.numpy()
        else:
            self.observations[self.idx] = postprocess_observation(observation.numpy(),
                                                                  self.bit_depth)  # Decentre and discretise visual observations (to save memory)
        self.actions[self.idx] = action.numpy()
        self.rewards[self.idx] = reward
        self.nonterminals[self.idx] = not done
        self.idx = (self.idx + 1) % self.size
        self.full = self.full or self.idx == 0
        self.steps, self.episodes = self.steps + 1, self.episodes + (1 if done else 0)

    # Returns an index for a valid single sequence chunk uniformly sampled from the memory
    def _sample_idx(self, L):
        valid_idx = False
        while not valid_idx:
            idx = np.random.randint(0, self.size if self.full else self.idx - L)
            idxs = np.arange(idx, idx + L) % self.size
            valid_idx = not self.idx in idxs[1:]  # Make sure data does not cross the memory index
        return idxs

    def _retrieve_batch(self, idxs, n, L):
        vec_idxs = idxs.transpose().reshape(-1)  # Unroll indices
        observations = torch.as_tensor(self.observations[vec_idxs].astype(np.float32))
        if not self.symbolic_env:
            preprocess_observation_(observations, self.bit_depth)  # Undo discretisation for visual observations
        return observations.reshape(L, n, *observations.shape[1:]), self.actions[vec_idxs].reshape(L, n, -1), \
               self.rewards[vec_idxs].reshape(L, n), self.nonterminals[vec_idxs].reshape(L, n, 1)

    # Returns a batch of sequence chunks uniformly sampled from the memory
    def sample(self, n, L):
        batch = self._retrieve_batch(np.asarray([self._sample_idx(L) for _ in range(n)]), n, L)
        # print(np.asarray([self._sample_idx(L) for _ in range(n)]))
        # [1578 1579 1580 ... 1625 1626 1627]                                                                                                                                        | 0/100 [00:00<?, ?it/s]
        # [1049 1050 1051 ... 1096 1097 1098]
        # [1236 1237 1238 ... 1283 1284 1285]
        # ...
        # [2199 2200 2201 ... 2246 2247 2248]
        # [ 686  687  688 ...  733  734  735]
        # [1377 1378 1379 ... 1424 1425 1426]]
        return [torch.as_tensor(item).to(device=self.device) for item in batch]


GYM_ENVS = ['Pendulum-v0', 'MountainCarContinuous-v0', 'Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2',
            'HumanoidStandup-v2', 'InvertedDoublePendulum-v2', 'InvertedPendulum-v2', 'Reacher-v2', 'Swimmer-v2',
            'Walker2d-v2']
CONTROL_SUITE_ENVS = ['cartpole-balance', 'cartpole-swingup', 'reacher-easy', 'finger-spin', 'cheetah-run',
                      'ball_in_cup-catch', 'walker-walk', 'reacher-hard', 'walker-run', 'humanoid-stand',
                      'humanoid-walk', 'fish-swim', 'acrobot-swingup']
CONTROL_SUITE_ACTION_REPEATS = {'cartpole': 8, 'reacher': 4, 'finger': 2, 'cheetah': 4, 'ball_in_cup': 6, 'walker': 2,
                                'humanoid': 2, 'fish': 2, 'acrobot': 4}


# Preprocesses an observation inplace (from float32 Tensor [0, 255] to [-0.5, 0.5])
def preprocess_observation_(observation, bit_depth):
    observation.div_(2 ** (8 - bit_depth)).floor_().div_(2 ** bit_depth).sub_(
        0.5)  # Quantise to given bit depth and centre
    observation.add_(torch.rand_like(observation).div_(
        2 ** bit_depth))  # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)


# Postprocess an observation for storage (from float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
def postprocess_observation(observation, bit_depth):
    return np.clip(np.floor((observation + 0.5) * 2 ** bit_depth) * 2 ** (8 - bit_depth), 0, 2 ** 8 - 1).astype(
        np.uint8)


def _images_to_observation(images, bit_depth):
    images = torch.tensor(cv2.resize(images, (64, 64), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1),
                          dtype=torch.float32)  # Resize and put channel first
    preprocess_observation_(images, bit_depth)  # Quantise, centre and dequantise inplace
    return images.unsqueeze(dim=0)  # Add batch dimension


class ControlSuiteEnv():
    def __init__(self, env, symbolic, seed, max_episode_length, action_repeat, bit_depth):
        from dm_control import suite
        from dm_control.suite.wrappers import pixels
        domain, task = env.split('-')
        self.symbolic = symbolic
        self._env = suite.load(domain_name=domain, task_name=task, task_kwargs={'random': seed})
        if not symbolic:
            self._env = pixels.Wrapper(self._env)
        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat
        if action_repeat != CONTROL_SUITE_ACTION_REPEATS[domain]:
            print('Using action repeat %d; recommended action repeat for domain is %d' % (
                action_repeat, CONTROL_SUITE_ACTION_REPEATS[domain]))
        self.bit_depth = bit_depth

    def reset(self):
        self.t = 0  # Reset internal timer
        state = self._env.reset()
        if self.symbolic:
            return torch.tensor(np.concatenate(
                [np.asarray([obs]) if isinstance(obs, float) else obs for obs in state.observation.values()], axis=0),
                dtype=torch.float32).unsqueeze(dim=0)
        else:
            return _images_to_observation(self._env.physics.render(camera_id=0), self.bit_depth)

    def step(self, action):
        action = action.detach().numpy()
        reward = 0
        for k in range(self.action_repeat):
            state = self._env.step(action)
            reward += state.reward
            self.t += 1  # Increment internal timer
            done = state.last() or self.t == self.max_episode_length
            if done:
                break
        if self.symbolic:
            observation = torch.tensor(np.concatenate(
                [np.asarray([obs]) if isinstance(obs, float) else obs for obs in state.observation.values()], axis=0),
                dtype=torch.float32).unsqueeze(dim=0)
        else:
            observation = _images_to_observation(self._env.physics.render(camera_id=0), self.bit_depth)
        return observation, reward, done

    def render(self):
        cv2.imshow('screen', self._env.physics.render(camera_id=0)[:, :, ::-1])
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()
        self._env.close()

    @property
    def observation_size(self):
        return sum([(1 if len(obs.shape) == 0 else obs.shape[0]) for obs in
                    self._env.observation_spec().values()]) if self.symbolic else (3, 64, 64)

    @property
    def action_size(self):
        return self._env.action_spec().shape[0]

    # Sample an action randomly from a uniform distribution over all valid actions
    def sample_random_action(self):
        spec = self._env.action_spec()
        return torch.from_numpy(np.random.uniform(spec.minimum, spec.maximum, spec.shape))


class GymEnv():
    def __init__(self, env, symbolic, seed, max_episode_length, action_repeat, bit_depth):
        import gym
        self.symbolic = symbolic
        self._env = gym.make(env)
        self._env.seed(seed)
        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat
        self.bit_depth = bit_depth

    def reset(self):
        self.t = 0  # Reset internal timer
        state = self._env.reset()
        if self.symbolic:
            return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)
        else:
            return _images_to_observation(self._env.render(mode='rgb_array'), self.bit_depth)

    def step(self, action):
        action = action.detach().numpy()
        reward = 0
        for k in range(self.action_repeat):
            state, reward_k, done, _ = self._env.step(action)
            reward += reward_k
            self.t += 1  # Increment internal timer
            done = done or self.t == self.max_episode_length
            if done:
                break
        if self.symbolic:
            observation = torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)
        else:
            observation = _images_to_observation(self._env.render(mode='rgb_array'), self.bit_depth)
        return observation, reward, done

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    @property
    def observation_size(self):
        return self._env.observation_space.shape[0] if self.symbolic else (3, 64, 64)

    @property
    def action_size(self):
        return self._env.action_space.shape[0]

    # Sample an action randomly from a uniform distribution over all valid actions
    def sample_random_action(self):
        return torch.from_numpy(self._env.action_space.sample())


def Env(env, symbolic, seed, max_episode_length, action_repeat, bit_depth):
    if env in GYM_ENVS:
        return GymEnv(env, symbolic, seed, max_episode_length, action_repeat, bit_depth)
    elif env in CONTROL_SUITE_ENVS:
        return ControlSuiteEnv(env, symbolic, seed, max_episode_length, action_repeat, bit_depth)


# Wrapper for batching environments together
class EnvBatcher():
    def __init__(self, env_class, env_args, env_kwargs, n):
        self.n = n
        self.envs = [env_class(*env_args, **env_kwargs) for _ in range(n)]
        self.dones = [True] * n

    # Resets every environment and returns observation
    def reset(self):
        observations = [env.reset() for env in self.envs]
        self.dones = [False] * self.n
        return torch.cat(observations)

    # Steps/resets every environment and returns (observation, reward, done)
    def step(self, actions):
        done_mask = torch.nonzero(torch.tensor(self.dones))[:,
                    0]  # Done mask to blank out observations and zero rewards for previously terminated environments
        observations, rewards, dones = zip(*[env.step(action) for env, action in zip(self.envs, actions)])
        dones = [d or prev_d for d, prev_d in
                 zip(dones, self.dones)]  # Env should remain terminated if previously terminated
        self.dones = dones
        observations, rewards, dones = torch.cat(observations), torch.tensor(rewards,
                                                                             dtype=torch.float32), torch.tensor(dones,
                                                                                                                dtype=torch.uint8)
        observations[done_mask] = 0
        rewards[done_mask] = 0
        return observations, rewards, dones

    def close(self):
        [env.close() for env in self.envs]


parser = argparse.ArgumentParser(description='PlaNet or Dreamer')
parser.add_argument('--algo', type=str, default='dreamer', help='planet or dreamer')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--env', type=str, default='Pendulum-v0', choices=GYM_ENVS + CONTROL_SUITE_ENVS,
                    help='Gym/Control Suite environment')
parser.add_argument('--symbolic-env', action='store_true', help='Symbolic features')
parser.add_argument('--max-episode-length', type=int, default=1000, metavar='T', help='Max episode length')
parser.add_argument('--experience-size', type=int, default=1000000, metavar='D',
                    help='Experience replay size')  # Original implementation has an unlimited buffer size, but 1 million is the max experience collected anyway
parser.add_argument('--cnn-activation-function', type=str, default='relu', choices=dir(F),
                    help='Model activation function for a convolution layer')
parser.add_argument('--dense-activation-function', type=str, default='elu', choices=dir(F),
                    help='Model activation function a dense layer')
parser.add_argument('--embedding-size', type=int, default=1024, metavar='E',
                    help='Observation embedding size')  # Note that the default encoder for visual observations outputs a 1024D vector; for other embedding sizes an additional fully-connected layer is used
parser.add_argument('--hidden-size', type=int, default=200, metavar='H', help='Hidden size')
parser.add_argument('--belief-size', type=int, default=200, metavar='H', help='Belief/hidden size')
parser.add_argument('--state-size', type=int, default=30, metavar='Z', help='State/latent size')
parser.add_argument('--action-repeat', type=int, default=2, metavar='R', help='Action repeat')
parser.add_argument('--action-noise', type=float, default=0.3, metavar='ε', help='Action noise')
parser.add_argument('--episodes', type=int, default=1000, metavar='E', help='Total number of episodes')
parser.add_argument('--seed-episodes', type=int, default=5, metavar='S', help='Seed episodes')
parser.add_argument('--collect-interval', type=int, default=100, metavar='C', help='Collect interval')
parser.add_argument('--batch-size', type=int, default=50, metavar='B', help='Batch size')
parser.add_argument('--chunk-size', type=int, default=50, metavar='L', help='Chunk size')
parser.add_argument('--worldmodel-LogProbLoss', action='store_true',
                    help='use LogProb loss for observation_model and reward_model training')
parser.add_argument('--overshooting-distance', type=int, default=50, metavar='D',
                    help='Latent overshooting distance/latent overshooting weight for t = 1')
parser.add_argument('--overshooting-kl-beta', type=float, default=0, metavar='β>1',
                    help='Latent overshooting KL weight for t > 1 (0 to disable)')
parser.add_argument('--overshooting-reward-scale', type=float, default=0, metavar='R>1',
                    help='Latent overshooting reward prediction weight for t > 1 (0 to disable)')
parser.add_argument('--global-kl-beta', type=float, default=0, metavar='βg', help='Global KL weight (0 to disable)')
parser.add_argument('--free-nats', type=float, default=3, metavar='F', help='Free nats')
parser.add_argument('--bit-depth', type=int, default=5, metavar='B', help='Image bit depth (quantisation)')
parser.add_argument('--model_learning-rate', type=float, default=1e-3, metavar='α', help='Learning rate')
parser.add_argument('--actor_learning-rate', type=float, default=8e-5, metavar='α', help='Learning rate')
parser.add_argument('--value_learning-rate', type=float, default=8e-5, metavar='α', help='Learning rate')
parser.add_argument('--learning-rate-schedule', type=int, default=0, metavar='αS',
                    help='Linear learning rate schedule (optimisation steps from 0 to final learning rate; 0 to disable)')
parser.add_argument('--adam-epsilon', type=float, default=1e-7, metavar='ε', help='Adam optimizer epsilon value')
# Note that original has a linear learning rate decay, but it seems unlikely that this makes a significant difference
parser.add_argument('--grad-clip-norm', type=float, default=100.0, metavar='C', help='Gradient clipping norm')
parser.add_argument('--planning-horizon', type=int, default=15, metavar='H', help='Planning horizon distance')
parser.add_argument('--discount', type=float, default=0.99, metavar='H', help='Planning horizon distance')
parser.add_argument('--disclam', type=float, default=0.95, metavar='H', help='discount rate to compute return')
parser.add_argument('--optimisation-iters', type=int, default=10, metavar='I', help='Planning optimisation iterations')
parser.add_argument('--candidates', type=int, default=1000, metavar='J', help='Candidate samples per iteration')
parser.add_argument('--top-candidates', type=int, default=100, metavar='K', help='Number of top candidates to fit')
parser.add_argument('--test', action='store_true', help='Test only')
parser.add_argument('--test-interval', type=int, default=25, metavar='I', help='Test interval (episodes)')
parser.add_argument('--test-episodes', type=int, default=10, metavar='E', help='Number of test episodes')
parser.add_argument('--checkpoint-interval', type=int, default=50, metavar='I', help='Checkpoint interval (episodes)')
parser.add_argument('--checkpoint-experience', action='store_true', help='Checkpoint experience replay')
parser.add_argument('--models', type=str, default='', metavar='M', help='Load model checkpoint')
parser.add_argument('--experience-replay', type=str, default='', metavar='ER', help='Load experience replay')
parser.add_argument('--render', action='store_true', help='Render environment')
args = parser.parse_args()
args.overshooting_distance = min(args.chunk_size,
                                 args.overshooting_distance)  # Overshooting distance cannot be greater than chunk size
print(' ' * 26 + 'Options')
for k, v in vars(args).items():
    print(' ' * 26 + k + ': ' + str(v))

# Setup
results_dir = os.path.join('results', '{}_{}'.format(args.env, args.id))
os.makedirs(results_dir, exist_ok=True)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available() and not args.disable_cuda:
    print("using CUDA")
    args.device = torch.device('cuda')
    torch.cuda.manual_seed(args.seed)
else:
    print("using CPU")
    args.device = torch.device('cpu')
metrics = {'steps': [], 'episodes': [], 'train_rewards': [], 'test_episodes': [], 'test_rewards': [],
           'observation_loss': [], 'reward_loss': [], 'kl_loss': [], 'actor_loss': [], 'value_loss': []}

summary_name = results_dir + "/{}_{}_log"
writer = SummaryWriter(summary_name.format(args.env, args.id))

# Initialise training environment and experience replay memory
env = Env(args.env, args.symbolic_env, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth)
if args.experience_replay != '' and os.path.exists(args.experience_replay):
    D = torch.load(args.experience_replay)
    metrics['steps'], metrics['episodes'] = [D.steps] * D.episodes, list(range(1, D.episodes + 1))
elif not args.test:
    D = ExperienceReplay(args.experience_size, args.symbolic_env, env.observation_size, env.action_size, args.bit_depth,
                         args.device)
    # Initialise dataset D with S random seed episodes
    for s in range(1, args.seed_episodes + 1):
        observation, done, t = env.reset(), False, 0
        while not done:
            action = env.sample_random_action()
            next_observation, reward, done = env.step(action)
            D.append(observation, action, reward, done)
            observation = next_observation
            t += 1
        metrics['steps'].append(t * args.action_repeat + (0 if len(metrics['steps']) == 0 else metrics['steps'][-1]))
        metrics['episodes'].append(s)

# Initialise model parameters randomly
transition_model = TransitionModel(args.belief_size, args.state_size, env.action_size, args.hidden_size,
                                   args.embedding_size, args.dense_activation_function).to(device=args.device)
observation_model = ObservationModel(args.symbolic_env, env.observation_size, args.belief_size, args.state_size,
                                     args.embedding_size, args.cnn_activation_function).to(device=args.device)
reward_model = RewardModel(args.belief_size, args.state_size, args.hidden_size, args.dense_activation_function).to(
    device=args.device)
encoder = Encoder(args.symbolic_env, env.observation_size, args.embedding_size, args.cnn_activation_function).to(
    device=args.device)
actor_model = ActorModel(args.belief_size, args.state_size, args.hidden_size, env.action_size,
                         args.dense_activation_function).to(device=args.device)
value_model = ValueModel(args.belief_size, args.state_size, args.hidden_size, args.dense_activation_function).to(
    device=args.device)
param_list = list(transition_model.parameters()) + list(observation_model.parameters()) + list(
    reward_model.parameters()) + list(encoder.parameters())
value_actor_param_list = list(value_model.parameters()) + list(actor_model.parameters())
params_list = param_list + value_actor_param_list
model_optimizer = optim.Adam(param_list, lr=0 if args.learning_rate_schedule != 0 else args.model_learning_rate,
                             eps=args.adam_epsilon)
actor_optimizer = optim.Adam(actor_model.parameters(),
                             lr=0 if args.learning_rate_schedule != 0 else args.actor_learning_rate,
                             eps=args.adam_epsilon)
value_optimizer = optim.Adam(value_model.parameters(),
                             lr=0 if args.learning_rate_schedule != 0 else args.value_learning_rate,
                             eps=args.adam_epsilon)
if args.models != '' and os.path.exists(args.models):
    model_dicts = torch.load(args.models)
    transition_model.load_state_dict(model_dicts['transition_model'])
    observation_model.load_state_dict(model_dicts['observation_model'])
    reward_model.load_state_dict(model_dicts['reward_model'])
    encoder.load_state_dict(model_dicts['encoder'])
    actor_model.load_state_dict(model_dicts['actor_model'])
    value_model.load_state_dict(model_dicts['value_model'])
    model_optimizer.load_state_dict(model_dicts['model_optimizer'])

planner = actor_model
global_prior = Normal(torch.zeros(args.batch_size, args.state_size, device=args.device),
                      torch.ones(args.batch_size, args.state_size, device=args.device))  # Global prior N(0, I)
free_nats = torch.full((1,), args.free_nats, device=args.device)  # Allowed deviation in KL divergence


def update_belief_and_act(args, env, planner, transition_model, encoder, belief, posterior_state, action, observation,
                          explore=False):
    # Infer belief over current state q(s_t|o≤t,a<t) from the history
    # print("action size: ",action.size()) torch.Size([1, 6])
    belief, _, _, _, posterior_state, _, _ = transition_model(posterior_state, action.unsqueeze(dim=0), belief,
                                                              encoder(observation).unsqueeze(
                                                                  dim=0))  # Action and observation need extra time dimension
    belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(
        dim=0)  # Remove time dimension from belief/state
    if args.algo == "dreamer":
        action = planner.get_action(belief, posterior_state, det=not (explore))
    else:
        action = planner(belief, posterior_state)  # Get action from planner(q(s_t|o≤t,a<t), p)
    if explore:
        action = torch.clamp(Normal(action, args.action_noise).rsample(), -1,
                             1)  # Add gaussian exploration noise on top of the sampled action
        # action = action + args.action_noise * torch.randn_like(action)  # Add exploration noise ε ~ p(ε) to the action
    next_observation, reward, done = env.step(action.cpu() if isinstance(env, EnvBatcher) else action[
        0].cpu())  # Perform environment step (action repeats handled internally)
    return belief, posterior_state, action, next_observation, reward, done


# Testing only
if args.test:
    # Set models to eval mode
    transition_model.eval()
    reward_model.eval()
    encoder.eval()
    with torch.no_grad():
        total_reward = 0
        for _ in tqdm(range(args.test_episodes)):
            observation = env.reset()
            belief, posterior_state, action = torch.zeros(1, args.belief_size, device=args.device), torch.zeros(1,
                                                                                                                args.state_size,
                                                                                                                device=args.device), torch.zeros(
                1, env.action_size, device=args.device)
            pbar = tqdm(range(args.max_episode_length // args.action_repeat))
            for t in pbar:
                belief, posterior_state, action, observation, reward, done = update_belief_and_act(args, env, planner,
                                                                                                   transition_model,
                                                                                                   encoder, belief,
                                                                                                   posterior_state,
                                                                                                   action,
                                                                                                   observation.to(
                                                                                                       device=args.device))
                total_reward += reward
                if args.render:
                    env.render()
                if done:
                    pbar.close()
                    break
    print('Average Reward:', total_reward / args.test_episodes)
    env.close()
    quit()

# Training (and testing)
for episode in tqdm(range(metrics['episodes'][-1] + 1, args.episodes + 1), total=args.episodes,
                    initial=metrics['episodes'][-1] + 1):
    # Model fitting
    losses = []
    model_modules = transition_model.modules + encoder.modules + observation_model.modules + reward_model.modules

    print("training loop")
    for s in tqdm(range(args.collect_interval)):
        # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly at random from the dataset (including terminal flags)
        observations, actions, rewards, nonterminals = D.sample(args.batch_size,
                                                                args.chunk_size)  # Transitions start at time t = 0
        # Create initial belief and state for time t = 0
        init_belief, init_state = torch.zeros(args.batch_size, args.belief_size, device=args.device), torch.zeros(
            args.batch_size, args.state_size, device=args.device)
        # Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)
        beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = transition_model(
            init_state, actions[:-1], init_belief, bottle(encoder, (observations[1:],)), nonterminals[:-1])
        # Calculate observation likelihood, reward likelihood and KL losses (for t = 0 only for latent overshooting); sum over final dims, average over batch and time (original implementation, though paper seems to miss 1/T scaling?)
        if args.worldmodel_LogProbLoss:
            observation_dist = Normal(bottle(observation_model, (beliefs, posterior_states)), 1)
            observation_loss = -observation_dist.log_prob(observations[1:]).sum(
                dim=2 if args.symbolic_env else (2, 3, 4)).mean(dim=(0, 1))
        else:
            observation_loss = F.mse_loss(bottle(observation_model, (beliefs, posterior_states)), observations[1:],
                                          reduction='none').sum(dim=2 if args.symbolic_env else (2, 3, 4)).mean(
                dim=(0, 1))
        if args.worldmodel_LogProbLoss:
            reward_dist = Normal(bottle(reward_model, (beliefs, posterior_states)), 1)
            reward_loss = -reward_dist.log_prob(rewards[:-1]).mean(dim=(0, 1))
        else:
            reward_loss = F.mse_loss(bottle(reward_model, (beliefs, posterior_states)), rewards[:-1],
                                     reduction='none').mean(dim=(0, 1))
        # transition loss
        div = kl_divergence(Normal(posterior_means, posterior_std_devs), Normal(prior_means, prior_std_devs)).sum(dim=2)
        kl_loss = torch.max(div, free_nats).mean(dim=(
            0, 1))  # Note that normalisation by overshooting distance and weighting by overshooting distance cancel out
        if args.global_kl_beta != 0:
            kl_loss += args.global_kl_beta * kl_divergence(Normal(posterior_means, posterior_std_devs),
                                                           global_prior).sum(dim=2).mean(dim=(0, 1))
        # Calculate latent overshooting objective for t > 0
        if args.overshooting_kl_beta != 0:
            overshooting_vars = []  # Collect variables for overshooting to process in batch
            for t in range(1, args.chunk_size - 1):
                d = min(t + args.overshooting_distance, args.chunk_size - 1)  # Overshooting distance
                t_, d_ = t - 1, d - 1  # Use t_ and d_ to deal with different time indexing for latent states
                seq_pad = (0, 0, 0, 0, 0,
                           t - d + args.overshooting_distance)  # Calculate sequence padding so overshooting terms can be calculated in one batch
                # Store (0) actions, (1) nonterminals, (2) rewards, (3) beliefs, (4) prior states, (5) posterior means, (6) posterior standard deviations and (7) sequence masks
                overshooting_vars.append((F.pad(actions[t:d], seq_pad), F.pad(nonterminals[t:d], seq_pad),
                                          F.pad(rewards[t:d], seq_pad[2:]), beliefs[t_], prior_states[t_],
                                          F.pad(posterior_means[t_ + 1:d_ + 1].detach(), seq_pad),
                                          F.pad(posterior_std_devs[t_ + 1:d_ + 1].detach(), seq_pad, value=1),
                                          F.pad(torch.ones(d - t, args.batch_size, args.state_size, device=args.device),
                                                seq_pad)))  # Posterior standard deviations must be padded with > 0 to prevent infinite KL divergences
            overshooting_vars = tuple(zip(*overshooting_vars))
            # Update belief/state using prior from previous belief/state and previous action (over entire sequence at once)
            beliefs, prior_states, prior_means, prior_std_devs = transition_model(
                torch.cat(overshooting_vars[4], dim=0), torch.cat(overshooting_vars[0], dim=1),
                torch.cat(overshooting_vars[3], dim=0), None, torch.cat(overshooting_vars[1], dim=1))
            seq_mask = torch.cat(overshooting_vars[7], dim=1)
            # Calculate overshooting KL loss with sequence mask
            kl_loss += (1 / args.overshooting_distance) * args.overshooting_kl_beta * torch.max((kl_divergence(
                Normal(torch.cat(overshooting_vars[5], dim=1), torch.cat(overshooting_vars[6], dim=1)),
                Normal(prior_means, prior_std_devs)) * seq_mask).sum(dim=2), free_nats).mean(dim=(0, 1)) * (
                               args.chunk_size - 1)  # Update KL loss (compensating for extra average over each overshooting/open loop sequence)
            # Calculate overshooting reward prediction loss with sequence mask
            if args.overshooting_reward_scale != 0:
                reward_loss += (1 / args.overshooting_distance) * args.overshooting_reward_scale * F.mse_loss(
                    bottle(reward_model, (beliefs, prior_states)) * seq_mask[:, :, 0],
                    torch.cat(overshooting_vars[2], dim=1), reduction='none').mean(dim=(0, 1)) * (
                                       args.chunk_size - 1)  # Update reward loss (compensating for extra average over each overshooting/open loop sequence)
        # Apply linearly ramping learning rate schedule
        if args.learning_rate_schedule != 0:
            for group in model_optimizer.param_groups:
                group['lr'] = min(group['lr'] + args.model_learning_rate / args.model_learning_rate_schedule,
                                  args.model_learning_rate)
        model_loss = observation_loss + reward_loss + kl_loss
        # Update model parameters
        model_optimizer.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(param_list, args.grad_clip_norm, norm_type=2)
        model_optimizer.step()

        # Dreamer implementation: actor loss calculation and optimization
        with torch.no_grad():
            actor_states = posterior_states.detach()
            actor_beliefs = beliefs.detach()
        with FreezeParameters(model_modules):
            imagination_traj = imagine_ahead(actor_states, actor_beliefs, actor_model, transition_model,
                                             args.planning_horizon)
        imged_beliefs, imged_prior_states, imged_prior_means, imged_prior_std_devs = imagination_traj
        with FreezeParameters(model_modules + value_model.modules):
            imged_reward = bottle(reward_model, (imged_beliefs, imged_prior_states))
            value_pred = bottle(value_model, (imged_beliefs, imged_prior_states))
        returns = lambda_return(imged_reward, value_pred, bootstrap=value_pred[-1], discount=args.discount,
                                lambda_=args.disclam)
        actor_loss = -torch.mean(returns)
        # Update model parameters
        actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor_model.parameters(), args.grad_clip_norm, norm_type=2)
        actor_optimizer.step()

        # Dreamer implementation: value loss calculation and optimization
        with torch.no_grad():
            value_beliefs = imged_beliefs.detach()
            value_prior_states = imged_prior_states.detach()
            target_return = returns.detach()
        value_dist = Normal(bottle(value_model, (value_beliefs, value_prior_states)),
                            1)  # detach the input tensor from the transition network.
        value_loss = -value_dist.log_prob(target_return).mean(dim=(0, 1))
        # Update model parameters
        value_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(value_model.parameters(), args.grad_clip_norm, norm_type=2)
        value_optimizer.step()

        # # Store (0) observation loss (1) reward loss (2) KL loss (3) actor loss (4) value loss
        losses.append(
            [observation_loss.item(), reward_loss.item(), kl_loss.item(), actor_loss.item(), value_loss.item()])

    # Update and plot loss metrics
    losses = tuple(zip(*losses))
    metrics['observation_loss'].append(losses[0])
    metrics['reward_loss'].append(losses[1])
    metrics['kl_loss'].append(losses[2])
    metrics['actor_loss'].append(losses[3])
    metrics['value_loss'].append(losses[4])
    lineplot(metrics['episodes'][-len(metrics['observation_loss']):], metrics['observation_loss'], 'observation_loss',
             results_dir)
    lineplot(metrics['episodes'][-len(metrics['reward_loss']):], metrics['reward_loss'], 'reward_loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['kl_loss']):], metrics['kl_loss'], 'kl_loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['actor_loss']):], metrics['actor_loss'], 'actor_loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['value_loss']):], metrics['value_loss'], 'value_loss', results_dir)

    # Data collection
    print("Data collection")
    with torch.no_grad():
        observation, total_reward = env.reset(), 0
        belief, posterior_state, action = torch.zeros(1, args.belief_size, device=args.device), torch.zeros(1,
                                                                                                            args.state_size,
                                                                                                            device=args.device), torch.zeros(
            1, env.action_size, device=args.device)
        pbar = tqdm(range(args.max_episode_length // args.action_repeat))
        for t in pbar:
            # print("step",t)
            belief, posterior_state, action, next_observation, reward, done = update_belief_and_act(args, env, planner,
                                                                                                    transition_model,
                                                                                                    encoder, belief,
                                                                                                    posterior_state,
                                                                                                    action,
                                                                                                    observation.to(
                                                                                                        device=args.device),
                                                                                                    explore=True)
            D.append(observation, action.cpu(), reward, done)
            total_reward += reward
            observation = next_observation
            if args.render:
                env.render()
            if done:
                pbar.close()
                break

        # Update and plot train reward metrics
        metrics['steps'].append(t + metrics['steps'][-1])
        metrics['episodes'].append(episode)
        metrics['train_rewards'].append(total_reward)
        lineplot(metrics['episodes'][-len(metrics['train_rewards']):], metrics['train_rewards'], 'train_rewards',
                 results_dir)

    # Test model
    print("Test model")
    if episode % args.test_interval == 0:
        # Set models to eval mode
        transition_model.eval()
        observation_model.eval()
        reward_model.eval()
        encoder.eval()
        actor_model.eval()
        value_model.eval()
        # Initialise parallelised test environments
        test_envs = EnvBatcher(Env, (
            args.env, args.symbolic_env, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth), {},
                               args.test_episodes)

        with torch.no_grad():
            observation, total_rewards, video_frames = test_envs.reset(), np.zeros((args.test_episodes,)), []
            belief, posterior_state, action = torch.zeros(args.test_episodes, args.belief_size,
                                                          device=args.device), torch.zeros(args.test_episodes,
                                                                                           args.state_size,
                                                                                           device=args.device), torch.zeros(
                args.test_episodes, env.action_size, device=args.device)
            pbar = tqdm(range(args.max_episode_length // args.action_repeat))
            for t in pbar:
                belief, posterior_state, action, next_observation, reward, done = update_belief_and_act(args, test_envs,
                                                                                                        planner,
                                                                                                        transition_model,
                                                                                                        encoder, belief,
                                                                                                        posterior_state,
                                                                                                        action,
                                                                                                        observation.to(
                                                                                                            device=args.device))
                total_rewards += reward.numpy()
                if not args.symbolic_env:  # Collect real vs. predicted frames for video
                    video_frames.append(make_grid(
                        torch.cat([observation, observation_model(belief, posterior_state).cpu()], dim=3) + 0.5,
                        nrow=5).numpy())  # Decentre
                observation = next_observation
                if done.sum().item() == args.test_episodes:
                    pbar.close()
                    break

        # Update and plot reward metrics (and write video if applicable) and save metrics
        metrics['test_episodes'].append(episode)
        metrics['test_rewards'].append(total_rewards.tolist())
        lineplot(metrics['test_episodes'], metrics['test_rewards'], 'test_rewards', results_dir)
        lineplot(np.asarray(metrics['steps'])[np.asarray(metrics['test_episodes']) - 1], metrics['test_rewards'],
                 'test_rewards_steps', results_dir, xaxis='step')
        if not args.symbolic_env:
            episode_str = str(episode).zfill(len(str(args.episodes)))
            write_video(video_frames, 'test_episode_%s' % episode_str, results_dir)  # Lossy compression
            save_image(torch.as_tensor(video_frames[-1]),
                       os.path.join(results_dir, 'test_episode_%s.png' % episode_str))
        torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))

        # Set models to train mode
        transition_model.train()
        observation_model.train()
        reward_model.train()
        encoder.train()
        actor_model.train()
        value_model.train()
        # Close test environments
        test_envs.close()

    writer.add_scalar("train_reward", metrics['train_rewards'][-1], metrics['steps'][-1])
    writer.add_scalar("train/episode_reward", metrics['train_rewards'][-1], metrics['steps'][-1] * args.action_repeat)
    writer.add_scalar("observation_loss", metrics['observation_loss'][0][-1], metrics['steps'][-1])
    writer.add_scalar("reward_loss", metrics['reward_loss'][0][-1], metrics['steps'][-1])
    writer.add_scalar("kl_loss", metrics['kl_loss'][0][-1], metrics['steps'][-1])
    writer.add_scalar("actor_loss", metrics['actor_loss'][0][-1], metrics['steps'][-1])
    writer.add_scalar("value_loss", metrics['value_loss'][0][-1], metrics['steps'][-1])
    print("episodes: {}, total_steps: {}, train_reward: {} ".format(metrics['episodes'][-1], metrics['steps'][-1],
                                                                    metrics['train_rewards'][-1]))

    # Checkpoint models
    if episode % args.checkpoint_interval == 0:
        torch.save({'transition_model': transition_model.state_dict(),
                    'observation_model': observation_model.state_dict(),
                    'reward_model': reward_model.state_dict(),
                    'encoder': encoder.state_dict(),
                    'actor_model': actor_model.state_dict(),
                    'value_model': value_model.state_dict(),
                    'model_optimizer': model_optimizer.state_dict(),
                    'actor_optimizer': actor_optimizer.state_dict(),
                    'value_optimizer': value_optimizer.state_dict()
                    }, os.path.join(results_dir, 'models_%d.pth' % episode))
        if args.checkpoint_experience:
            torch.save(D, os.path.join(results_dir,
                                       'experience.pth'))  # Warning: will fail with MemoryError with large memory sizes

# Close training environment
env.close()
