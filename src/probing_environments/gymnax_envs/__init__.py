"""Import environments to easily expose them to the user"""

from probing_environments.gymnax_envs.AdvantagePolicyLossPolicyUpdateEnv import (
    AdvantagePolicyLossPolicyUpdateEnv,
)
from probing_environments.gymnax_envs.PolicyAndValueEnv import PolicyAndValueEnv
from probing_environments.gymnax_envs.RewardDiscountingEnv import RewardDiscountingEnv
from probing_environments.gymnax_envs.ValueBackpropEnv import ValueBackpropEnv
from probing_environments.gymnax_envs.ValueLossOrOptimizerEnv import (
    ValueLossOrOptimizerEnv,
)

env_list = [  # pylint: disable=F401
    AdvantagePolicyLossPolicyUpdateEnv,
    PolicyAndValueEnv,
    RewardDiscountingEnv,
    ValueBackpropEnv,
    ValueLossOrOptimizerEnv,
]
