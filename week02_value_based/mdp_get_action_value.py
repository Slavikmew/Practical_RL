
def get_action_value(mdp, state_values, state, action, gamma):
    """ Computes Q(s,a) as in formula above """
    result = 0
    for to_state in mdp.get_all_states():
        transition_probability = mdp.get_transition_prob(state, action, to_state)
        reward = mdp.get_reward(state, action, to_state)
        result += transition_probability * (reward + gamma * state_values[to_state])
    return result
