import numpy as np


class Base_agent:
    def __init__(self):
        pass

    def train(self):
        pass

    def get_action(self):
        pass

    def end_episode(self):
        pass

    def set_initial_values(self):
        pass


import numpy as np
from CybORG_plus_plus.mini_CAGE.minimal import HOSTS
from CybORG_plus_plus.mini_CAGE.agents import Base_agent


class B_line_minimal(Base_agent):
    _HOST_TO_SUBNET = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipeline = [2, 0, 1]  # user → enterprise → op
        self.n = len(HOSTS)
        self.reset()

    def reset(self):
        self.subnet_ptr = 0
        self.stage = 0
        self.last_escalated = None
        self.last_action = None
        self.user_target = None  # Will hold a randomly selected host from user subnet
        self.target_host = None
        self.ent_target_ptr = 0

    def end_episode(self):
        self.reset()

    def get_action(self, observation, *args, **kwargs):
        print(f"######### get action ##########")
        succ = observation.reshape(1, -1)[0, 0]
        obs = observation.reshape(1, -1)[0, 1:].reshape(self.n, 3)
        a = 0

        if self.last_action is not None and succ == 1:
            # If we have already finished the pipeline, stay in impact loop
            if self.subnet_ptr >= len(self.pipeline):
                self.stage = 4  # Enter infinite impact loop stage
                self.last_action = None
                return self.get_action(observation)

            if self.stage == 2 and self.target_host == 7:
                print("[DEBUG] Exploit success on op server after restore, moving to escalate.")
                self.stage = 3
                self.last_action = None
                return self.get_action(observation)

            # If remote discovery just succeeded, move immediately to next stage
            if self.stage == 0:
                self.stage += 1
                self.last_action = None
                return self.get_action(observation)

            print(f"self.target_host: {self.target_host}, obs.shape: {obs.shape}")
            print(f"obs[self.target_host, 1]: {obs[self.target_host, 1]}")
            print(f"stage: {self.stage}, self.last_escalated: {self.last_escalated}")

            # After escalation, either move to next target or subnet
            if self.stage == 3:
                if obs[self.target_host, 1] == 0 and obs[self.target_host, 2] == 0:
                    print("[DEBUG] Escalation failed on target host, try escalate again.")
                    self.stage = 3
                    # Keep the last escalated host for next escalation attempt
                else:
                    h = self.last_action - (4 + 2 * self.n)
                    self.last_escalated = h

                    if self.pipeline[self.subnet_ptr] == 2:
                        self.user_target = None

                    if self.pipeline[self.subnet_ptr] == 0:  # Enterprise subnet
                        self.ent_target_ptr += 1
                        if self.ent_target_ptr >= len(self.ent_targets):
                            self.subnet_ptr += 1  # Move to next subnet after both ent targets
                            self.ent_target_ptr = 0
                        self.stage = 0
                    else:
                        self.subnet_ptr += 1
                        self.stage = 0

                    self.target_host = None

            # If a remove happens when an exploit happens, the remove trumps the exploit, so try the exploit again.
            elif self.stage == 2 and self.target_host is not None and obs[self.target_host, 1] == 0:
                print("[DEBUG] Exploit failed on target host, try exploit again.")
                self.stage = 2  # Stay in exploit stage
                self.user_target = self.target_host  # Keep the user target for next exploit attempt
            else:
                self.stage += 1

            self.last_action = None

        # If last action failed, repeat it
        if self.last_action is not None and succ == 0:
            # If in impact loop and op server has been restored, go back to exploit stage
            if self.stage == 4 and obs[7, 2] == 0:
                print("[DEBUG] Op server restored during impact (via failed action)! Going back to exploit stage.")
                self.stage = 2
                self.target_host = 7

                # Plan the exploit action directly here
                a = 4 + self.n + 7  # Exploit op server again

                # IMPORTANT: Reset the subnet_ptr to point to the op server subnet
                self.subnet_ptr = len(self.pipeline) - 1  # Index of the last subnet in the pipeline
                return self.get_action(observation)
            return np.array([[self.last_action]])

        if self.subnet_ptr >= len(self.pipeline):
            if obs[7, 2] == 1:
                # If op server is still impacted, keep impacting
                a = 4 + 3 * self.n + 7
            else:
                # Op server has been restored: go straight to exploit stage
                print("[DEBUG] Op server restored! Going back to exploit stage.")
                self.stage = 2  # Set to exploit stage
                self.target_host = 7  # Set target

                # Plan the exploit action directly here
                a = 4 + self.n + 7  # Exploit op server again

                # IMPORTANT: Reset the subnet_ptr to point to the op server subnet
                self.subnet_ptr = len(self.pipeline) - 1  # Index of the last subnet in the pipeline

                self.last_action = int(a)
                return np.array([[self.last_action]])
        else:
            subnet = self.pipeline[self.subnet_ptr]
            hosts = np.where(self._HOST_TO_SUBNET == subnet)[0]

            if subnet == 2:
                hosts = hosts[hosts != 8]  # Exclude user0
                if self.user_target is None:
                    print(f'hosts: {hosts}')

                    self.user_target = np.random.choice(hosts)
                    print(f"Chosen user host: {self.user_target}")
                    # For debugging decoy purposes, hardcode the same user host
                    # self.user_target = 9  # For testing purposes, always choose user1


                    # Set ent_targets based on chosen user host
                    if self.user_target in [9, 10]:
                        self.ent_targets = [1, 3]  # ent0 then ent2
                        print(f'self.user_target: {self.user_target}, PART 1')
                    else:
                        self.ent_targets = [2, 3]  # ent1 then ent2
                        print(f'self.user_target: {self.user_target}, PART 2')

                    self.ent_target_ptr = 0  # Reset pointer to start of ent_targets

                hosts = np.array([self.user_target])  # Focus only on one random user host

            if subnet == 0:
                target_host = self.ent_targets[self.ent_target_ptr]
                hosts = np.array([target_host])
                print(f"Enterprise target host: {hosts}")

            # STAGE 0: remote_subnet
            if self.stage == 0:
                if not (subnet == 1 and self.last_escalated == 3):
                    if subnet == 0:
                        # Enterprise: must discover target host specifically
                        if obs[hosts[0], 0] == -1:
                            a = subnet + 1  # Remote discovery
                        else:
                            self.stage += 1
                            return self.get_action(observation)
                    else:
                        if np.any(obs[hosts, 0] == -1):
                            a = subnet + 1
                        else:
                            self.stage += 1
                            return self.get_action(observation)
                else:
                    # If last escalated was enterprise, skip to next stage
                    self.stage += 1
                    return self.get_action(observation)

            # STAGE 1: network_host
            elif self.stage == 1:
                mask = (obs[hosts, 0] == 0)
                print(f"obs[hosts, 0]: {obs[hosts, 0]}")
                print(f"Hosts: {hosts}, Mask: {mask}")
                if np.any(mask):
                    h = hosts[np.argmax(mask)]
                    a = 4 + h

            # STAGE 2: exploit_host
            elif self.stage == 2:
                mask = (obs[hosts, 0] == 1) & (obs[hosts, 1] == 0)
                if np.any(mask):
                    h = hosts[np.argmax(mask)]
                    a = 4 + self.n + h
                    self.target_host = h

            # STAGE 3: escalate_host
            elif self.stage == 3:
                if self.target_host is not None:
                    if obs[self.target_host, 1] == 1 and obs[self.target_host, 2] == 0:
                        a = 4 + 2 * self.n + self.target_host
                    elif obs[self.target_host, 1] == 0 and obs[self.target_host, 2] == 0:
                        # If target host is not exploited, escalate again
                        print("[DEBUG] Escalation failed on target host due to restore action.")
                        self.stage = 2
                        self.last_escalated = None
                        return self.get_action(observation)
                    else:
                        a = 0  # Sleep
                else:
                    print(f"obs: {obs}")
                    print(f"success: {succ}")
                    raise ValueError("No target host set for escalation.")

            # Infinite impact loop
            if self.stage == 4:
                if obs[7, 2] == 1:
                    # If op server is still impacted, keep impacting
                    a = 4 + 3 * self.n + 7
                else:
                    # If op server has been restored, go back to exploit stage
                    self.stage = 2  # Go back to exploit stage
                    self.target_host = 7  # Set op server as target
                    a = 4 + self.n + 7  # Exploit op server again

        self.last_action = int(a)
        return np.array([[self.last_action]])


# class B_line_minimal(Base_agent):
#     '''
#     B_line implementation that is compatible
#     with both minimal and default CAGE.
#     '''
#     def __init__(self, *args, **kwargs):
#         super(Base_agent).__init__(*args, **kwargs)
#
#     def get_action(self, observation, *args, **kwargs):
#
#         # convert to 2D
#         # decompose the state
#         if observation.ndim == 1:
#             observation = observation.reshape(1, -1)
#         host_info = observation[:, 1:].reshape(observation.shape[0], -1, 3)
#         actions = np.zeros(observation.shape[0])
#         action_selected = np.zeros_like(actions)
#
#         # scan subnet 3
#         actions, action_selected = self._scan_subnet(
#             subnet_idx=2, host_idx=12, host_info=host_info,
#              actions=actions, action_selected=action_selected)
#
#         # select action for a user host
#         # chosen_user_host = np.random.choice([9, 10, 11, 12]) # don't include user 0 (8) (foothold user)
#         actions, action_selected = self._check_host(
#             host_index=1, action_selected=action_selected,
#             actions=actions, host_info=host_info)
#
#         # select action for ent 1
#         actions, action_selected = self._check_host(
#             host_index=2, action_selected=action_selected,
#             actions=actions, host_info=host_info)
#
#         # scan subnet 1
#         actions, action_selected = self._scan_subnet(
#             subnet_idx=0, host_idx=0, host_info=host_info,
#              actions=actions, action_selected=action_selected)
#
#         # select action for ent 2
#         actions, action_selected = self._check_host(
#             host_index=3, action_selected=action_selected,
#             actions=actions, host_info=host_info)
#
#         # select action for opserver
#         actions, action_selected = self._check_host(
#             host_index=7, action_selected=action_selected,
#             actions=actions, host_info=host_info)
#
#         # impact the operational server
#         at_op_server = np.invert(action_selected.astype(bool))
#         if np.any(at_op_server):
#             actions[at_op_server] = 50
#             action_selected[at_op_server] = 1
#
#         return actions.reshape(-1, 1).astype(int)
#
#
#     def _check_host(self, host_index, action_selected, actions, host_info):
#         '''Given the index of a host select to scan, exploit or escalate. '''
#
#         # scan the host
#         num_hosts = host_info.shape[1]
#         host_scanned = host_info[:, host_index, 0] == 1
#         scan_host = np.logical_and(
#             np.invert(host_scanned), np.invert(action_selected.astype(bool)))
#         if np.any(scan_host):
#             actions[scan_host] = host_index+4
#             action_selected[scan_host] = 1
#
#         # exploit user host
#         host_exploited = host_info[:, host_index, 1] == 1
#         host_privileged = host_info[:, host_index, -1] == 1
#         host_access = np.logical_or(host_exploited, host_privileged)
#         host_exp = np.logical_and(host_scanned, np.invert(host_access))
#         exp_host = np.logical_and(
#             host_exp, np.invert(action_selected.astype(bool)))
#         if np.any(exp_host):
#             actions[exp_host] = num_hosts+4+host_index
#             action_selected[exp_host] = 1
#
#         # priv access user host
#         host_priv = np.logical_and(host_exploited, np.invert(host_privileged))
#         priv_host = np.logical_and(
#             host_priv, np.invert(action_selected.astype(bool)))
#         if np.any(host_priv):
#             actions[priv_host] = num_hosts*2+4+host_index
#             action_selected[priv_host] = 1
#
#         return actions, action_selected
#
#     def _scan_subnet(
#         self, subnet_idx, host_idx, host_info, actions, action_selected):
#
#         subnet_unknown = host_info[:, host_idx, -1] == -1
#         subnet_unknown = np.logical_and(
#             subnet_unknown, np.invert(action_selected.astype(bool)))
#         if np.any(subnet_unknown):
#             actions[subnet_unknown] = subnet_idx+1
#             action_selected[subnet_unknown] = 1
#
#         return actions, action_selected

class Meander_minimal(Base_agent):
    def __init__(self, *args, **kwargs):
        super(Base_agent).__init__(*args, **kwargs)
        self.subnet_structure = np.array([
            0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        self.num_hosts = len(self.subnet_structure)

    def get_action(self, observation, *args, **kwargs):

        # convert to 2D
        # decompose the state
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
        host_info = observation[:, 1:].reshape(observation.shape[0], -1, 3)
        actions = np.zeros(observation.shape[0])
        action_selected = np.zeros_like(actions).astype(bool)

        # impact opserver ---------------------------------------

        opserver_priv = (host_info[:, 7, -1] == 1)
        if np.any(opserver_priv):
            actions[opserver_priv] = 50
            action_selected[opserver_priv] = 1

        # subnet scan --------------------------------------------

        host_priv = (host_info[:, :, -1] == 1)[..., None]
        host_unknown = (host_info[:, :, 0] == -1)[..., None]

        # get unique subnet configuration
        unique_subnets = np.unique(self.subnet_structure)
        subnet_idx = (self.subnet_structure[:, None] == unique_subnets).astype(int)
        subnet_idx = subnet_idx[None]

        # identifty subnets with privilege and unknown hosts
        subnet_unknown = np.any(subnet_idx * host_unknown, axis=1)
        subnet_priv = np.any(subnet_idx * host_priv, axis=1)
        subnet_scan = np.logical_and(subnet_unknown, subnet_priv)
        network_scan = np.any(subnet_scan, axis=-1)
        network_scan = np.logical_and(network_scan, np.invert(action_selected))
        if np.any(network_scan):
            selected_subnet = np.argmax(subnet_scan[network_scan], axis=-1)
            actions[network_scan] = selected_subnet + 1
            action_selected[network_scan] = 1

        # network scan -------------------------------------------

        host_unscanned = (host_info[:, :, 0] == 0)
        network_unscanned = np.any(host_unscanned, axis=-1)
        network_unscanned = np.logical_and(
            network_unscanned, np.invert(action_selected))
        if np.any(network_unscanned):
            row_probs = (host_unscanned[network_unscanned] != 0) / (
                np.sum((host_unscanned[network_unscanned] != 0), axis=-1)).reshape(-1, 1)
            selected_host = (
                    np.random.rand(len(row_probs), 1) < row_probs.cumsum(axis=-1)).argmax(axis=-1)
            actions[network_unscanned] = selected_host + 4
            action_selected[network_unscanned] = 1

        # escalate exploited network -------------------------------

        host_exploited = (host_info[:, :, 1] == 1)
        network_exploits = np.any(host_exploited, axis=-1)
        network_exploits = np.logical_and(
            network_exploits, np.invert(action_selected))
        if np.any(network_exploits):
            selected_host = np.argmax(host_exploited[network_exploits], axis=-1)
            actions[network_exploits] = selected_host + 2 * self.num_hosts + 4
            action_selected[network_exploits] = 1

        # exploit host ---------------------------------------------

        # ensure you always ignore the defender with exploitation
        host_scanned = (host_info[:, :, 0] == 1)
        host_priv = (host_info[:, :, -1] == 1)
        host_scanned[:, 0] = False
        host_exploitable = np.logical_and(host_scanned, np.invert(host_priv))
        network_exploitable = np.any(host_exploitable, axis=-1)
        network_exploitable = np.logical_and(
            network_exploitable, np.invert(action_selected))
        if np.any(network_exploitable):
            row_probs = (host_exploitable[network_exploitable] != 0) / (
                np.sum((host_exploitable[network_exploitable] != 0), axis=-1)).reshape(-1, 1)
            selected_host = (
                    np.random.rand(len(row_probs), 1) < row_probs.cumsum(axis=-1)).argmax(axis=-1)
            actions[network_exploitable] = selected_host + self.num_hosts + 4
            action_selected[network_exploitable] = 1

        return actions.reshape(-1, 1)


class React_restore_minimal(Base_agent):
    '''
    React-restore agent compatible with minimal and default simulator.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()

    def reset(self, *args, **kwargs):
        self.num_hosts = 13
        self.host_list = None

    def get_action(self, observation, *args, **kwargs):

        # reformat observation
        batch_size = observation.shape[0]
        if self.host_list is None:
            self.host_list = np.zeros((batch_size, self.num_hosts))
        host_info = observation[:, :4 * self.num_hosts].reshape(-1, self.num_hosts, 4)

        # update the host list
        exploited_host = host_info[:, :, -1] == 1
        host_idxs = np.where(exploited_host == 1)
        if len(host_idxs[0]) > 0:
            self.host_list[host_idxs[0], host_idxs[1]] = 1

        # restore a host to default
        actions = np.zeros(batch_size)
        host_to_restore = np.any(self.host_list == 1, axis=-1)
        if np.any(host_to_restore):
            selected_host = np.argmax(self.host_list[host_to_restore], axis=-1)
            actions[host_to_restore] = selected_host + 40
            filtered_hosts = np.arange(len(observation))[host_to_restore]
            self.host_list[filtered_hosts, selected_host] = 0

        return actions.reshape(-1, 1).astype(int)


class Blue_sleep(Base_agent):
    '''
    Inactive agent compatible with CAGE implementations.
    '''

    def __init__(self, *args, **kwargs):
        super(Base_agent).__init__(*args, **kwargs)

    def get_action(self, observation, *args, **kwargs):
        return np.zeros((observation.shape[0], 1)).astype(int)

    def end_episode(self):
        pass


class Restore_decoys(React_restore_minimal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()

    def reset(self, *args, **kwargs):
        self.decoy_order = np.array([3, 7, 2, 1, 7, 7, 1, 7, 1, 1])
        self.obs_set = None
        super().reset(*args, **kwargs)

    def get_action(self, observation, *args, **kwargs):

        # get the order of decoys
        if self.obs_set is None:
            self.obs_set = observation.shape[0]
            self.decoy_order = np.tile(
                self.decoy_order.reshape(1, -1), (self.obs_set, 1))

        # select the action using react restore logic
        action = super().get_action(observation, *args, **kwargs)

        # if action is zero add an appropriate decoy
        # interested in decoys on opserver, ent2, ent1, ent0
        is_sleep = (action == 0).reshape(-1)
        has_decoys = np.sum(self.decoy_order != -1, axis=-1).reshape(-1)
        valid = np.logical_and(is_sleep, has_decoys)
        if np.any(valid):
            new_action_idx = np.argmax(self.decoy_order != -1, axis=-1)
            new_action = self.decoy_order[np.arange(len(self.decoy_order)), new_action_idx]
            self.decoy_order[np.arange(len(self.decoy_order))[is_sleep], new_action_idx[is_sleep]] = -1
            action[is_sleep] = new_action.reshape(-1, 1)[is_sleep] + 14

        return action.reshape(-1, 1).astype(int)


if __name__ == '__main__':

    from CybORG_plus_plus.mini_CAGE.minimal import SimplifiedCAGE
    from CybORG_plus_plus.mini_CAGE.minimal import action_mapping

    seed = 55749  # random.randint(1, 100000)
    np.random.seed(seed)

    # initialise environment
    batch_size = 1
    env = SimplifiedCAGE(num_envs=batch_size, num_nodes=13)
    s, _ = env.reset()

    # initialise the agents 
    red_agent = Meander_minimal()
    blue_agent = Restore_decoys()

    reward_log = []
    actions = []
    total_reward = np.zeros(batch_size)
    action_mapping = action_mapping()

    for i in range(10):
        print('###################')

        print(f"{i} - {s['Red']}")

        # print(f"length of red obs: {len(s['Red'][0])}")

        blue_action = blue_agent.get_action(observation=s['Blue'])
        red_action = red_agent.get_action(observation=s['Red'])

        red_action_no = red_action.reshape(-1)[0]
        blue_action_no = blue_action.reshape(-1)[0]

        print(f'Red: {red_action_no} - Blue: {blue_action_no}')
        print(f'Red: {action_mapping["Red"][int(red_action_no)]} - Blue: {action_mapping["Blue"][int(blue_action_no)]}')

        s, r, d, i = env.step(
            blue_action=blue_action, red_action=red_action)

        total_reward += r['Blue'].reshape(-1)
        reward_log.append(r['Blue'].reshape(-1))
        print('Reward ', r['Blue'])
        print(f'Blue obs space: {s["Blue"]}')
        print(f'len Blue obs space: {len(s["Blue"][0])}')

    print(actions)
    print(f'Total Reward: {total_reward}')
    print(np.stack(reward_log, axis=-1))
    print('SEED', seed)
