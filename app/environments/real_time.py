from app.environments.environment import Environment

class RealTimeEnvironment(Environment):
    """
    Defines a real-time environment.

    state consists of:
        * temperature
        * position

    action consists of:
        * voltage change
    """

    def get_initial_state(self, options):
        # TODO
        return None

    def get_next_state(self, action):
        # TODO
        return None
