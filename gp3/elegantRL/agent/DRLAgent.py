from elegantRL.train import Config
from elegantRL.train import train_agent
from elegantRL.agent import AgentPPO
import numpy as np
import torch


MODELS = {"ppo": AgentPPO}


class DRLAgent:
    """Implementations of DRL algorithms
    Attributes
    ----------
        env: gym env class or user-defined class

    Methods
    -------
        get_model()
            setup DRL algorithms
        train_model()
            train DRL algorithms in a train dataset
            and output the trained model
        DRL_prediction()
            make a prediction in a test dataset and get results
    """

    def __init__(self, env, price_array, tech_array, turbulence_array):
            self.env = env
            self.price_array = price_array
            self.tech_array = tech_array
            self.turbulence_array = turbulence_array

    def get_model(self, model_name, model_kwargs):
        env_config = {
            "price_array": self.price_array,
            "tech_array": self.tech_array,
            "turbulence_array": self.turbulence_array,
            "gamma": model_kwargs["gamma"],
            "reward_scaling": model_kwargs["reward_scaling"],
            "if_train": True,
        }
        env = self.env(config=env_config)
        env_args = {
            'config': env_config,
            'env_name': env.env_name,
            'num_envs': env.num_envs,
            'max_step': env.max_step,
            'state_dim': env.state_dim,
            'action_dim': env.action_dim,
            'if_build_vec_env': False,
            'if_discrete': False,
        }
        agent = MODELS[model_name]
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")
        model = Config(agent_class=agent, env_class=self.env, env_args=env_args)
        if model_kwargs is not None:
            try:
                model.learning_rate = model_kwargs["learning_rate"]
                model.batch_size = model_kwargs["batch_size"]
                model.gamma = model_kwargs["gamma"]
                model.net_dims = model_kwargs["net_dimension"]
                model.weight_decay = model_kwargs["weight_decay"]
                model.lambda_entropy = model_kwargs["lambda_entropy"]
                model.horizon_len = model_kwargs["horizon_len"]
                model.optimizer = model_kwargs["optimizer"]
                model.clip_grad_norm = model_kwargs["clip_grad_norm"]
                model.ratio_clip = model_kwargs["ratio_clip"]
                model.lambda_gae_adv = model_kwargs["lambda_gae_adv"]
                model.repeat_times = model_kwargs["repeat_times"]
                model.activation = model_kwargs["activation"]
                model.random_seed = model_kwargs["random_seed"]
                
            except BaseException:
                raise ValueError(
                    "Fail to read arguments, please check 'model_kwargs' input."
                )
        return model

    def train_model(self, model, cwd, total_timesteps=5000):
        model.cwd = cwd
        model.break_step = total_timesteps
        train_agent(model)

    @staticmethod
    def DRL_prediction(env, cwd):
        env.num_envs = 1
        
        # load agent
        try:  
            actor_path = cwd + '/actor_model_avgR.pth'
            print(f"| Loading actor from: {actor_path}")
            actor = torch.load(actor_path, map_location=lambda storage, loc: storage)
            actor.eval()  # set to evaluation mode
            act = actor
            device = next(actor.parameters()).device
        except BaseException:
            raise ValueError(f"Failed to load actor_model.pth from {cwd}!")

        # test on the testing env
        state = env.reset()
        episode_total_assets = [env.initial_total_assets]
        with torch.no_grad():
            for _ in range(env.max_step):
                tensor_state = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                tensor_action = act(tensor_state)
                action = tensor_action.detach().cpu().numpy()[0]
                state, reward, done, _ = env.step(action)
                total_assets = env.total_assets
                episode_total_assets.append(total_assets)
                if done:
                    break
        episode_return = env.cumulative_return
        running_max = np.maximum.accumulate(episode_total_assets)
        drawdowns = running_max - episode_total_assets
        max_drawdown = np.max(drawdowns)
        avg_reward = np.mean(env.rewards)
        print("\nTest Results:")
        print(f" - Episode Return: {episode_return}")
        print(f" - Total Assets: {episode_total_assets[-1]}")
        print(f" - Max Drawdown: {max_drawdown}")
        print(f" - Average Reward: {avg_reward}\n")
        return episode_total_assets[-1], max_drawdown, avg_reward