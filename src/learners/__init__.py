from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .actor_critic_learner import ActorCriticLearner
from .actor_critic_pac_learner import PACActorCriticLearner
from .actor_critic_pac_dcg_learner import PACDCGLearner
from .maddpg_learner import MADDPGLearner
from .ppo_learner import PPOLearner
from .bc_learner import BCLearner
from .icq_learner import ICQLearner
from .offline_q_learner import OfflineQLearner
from .dynamics_learner import OfflineDynamicsLearner
from .matd3_learner import MATD3Learner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["actor_critic_learner"] = ActorCriticLearner
REGISTRY["maddpg_learner"] = MADDPGLearner
REGISTRY["ppo_learner"] = PPOLearner
REGISTRY["pac_learner"] = PACActorCriticLearner
REGISTRY["pac_dcg_learner"] = PACDCGLearner
REGISTRY["bc_learner"] = BCLearner
REGISTRY["icq_learner"] = ICQLearner
REGISTRY["offline_q_learner"] = OfflineQLearner
REGISTRY["offline_dynamics_learner"] = OfflineDynamicsLearner
REGISTRY["matd3_learner"] = MATD3Learner
