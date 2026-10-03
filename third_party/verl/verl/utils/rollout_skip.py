class RolloutSkip:
    def __init__(self, config, rollout_wg):
        self.config = config
        self._rollout_wg = rollout_wg
        self.is_enable = bool(config.actor_rollout_ref.rollout.skip.get("enable", False))
        self.is_active = self.is_enable
        self.is_dump_step = self.is_enable

    def wrap_generate_sequences(self):
        print("RolloutSkip is disabled or running in compatibility mode", flush=True)

    def record(self, *args, **kwargs):
        return None
