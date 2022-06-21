from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO
from stable_baselines3.common import policies


def load_expert_policy(env_name: str, venv = None):
    """Loads expert policy from huggingface hub.

    Args:
        env_name: The name of the environment, that the expert was created for.
    """
    # TODO(ernestum): use naming scheme tools from the future here
    # TODO(ernestum): use official repositories
    return PPO.load(
        load_from_hub(
            f"ernestumorga/ppo-{env_name.replace('/', '-')}", f"ppo-{env_name}.zip"
        ), env = venv
    )