"""Import continuous-action environments to easily expose them to the user"""

from probing_environments.gymnax_envs.continuous_actions.AdvantagePolicyLossPolicyUpdateEnv import (
    AdvantagePolicyLossPolicyUpdateEnv,
)
from probing_environments.gymnax_envs.continuous_actions.PolicyAndValueEnv import (
    PolicyAndValueEnv,
)
from probing_environments.gymnax_envs.continuous_actions.RewardDiscountingEnv import (
    RewardDiscountingEnv,
)
from probing_environments.gymnax_envs.continuous_actions.ValueBackpropEnv import (
    ValueBackpropEnv,
)
from probing_environments.gymnax_envs.continuous_actions.ValueLossOrOptimizerEnv import (
    ValueLossOrOptimizerEnv,
)

env_list = [
    AdvantagePolicyLossPolicyUpdateEnv,
    PolicyAndValueEnv,
    RewardDiscountingEnv,
    ValueBackpropEnv,
    ValueLossOrOptimizerEnv,
]
