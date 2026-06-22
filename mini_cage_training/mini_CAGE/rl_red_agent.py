import random


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
    # ───────────────────────── constructor ────────────────────────────
    def __init__(self):
        self.action       = 0
        self.last_host    = None          # index of most-recent target host
        self.last_action  = None          # np.array([[id]]) sent last step
        self.jumps        = [0, 1, 2, 2, 2, 2, 5, 5, 5, 5, 9, 9, 9, 12, 13]
        self.first_user_host = None

        HOSTS = [
            'def', 'ent0', 'ent1', 'ent2', 'ophost0',
            'ophost1', 'ophost2', 'opserv',
            'user0', 'user1', 'user2', 'user3', 'user4'
        ]
        self.hosts = {h: i for i, h in enumerate(HOSTS)}

    # ────────────────────  USER-SUBNET HELPERS  ───────────────────────
    def DiscoverRemoteSystems_user(self):
        return 3                                                # DRS user subnet

    def DiscoverNetworkServices_user(self, host, obs):
        if host not in (9, 10, 11, 12):
            raise ValueError(f"bad host {host}")

        row = obs.reshape(1, -1)[0, 1:].reshape(13, 3)[host]
        if (row == [0, 0, 0]).all():                            # not scanned yet
            return {9: 13, 10: 14, 11: 15, 12: 16}[host]
        return None                                             # already scanned

    def ExploitRemoteService_user(self, host, obs):
        if host not in (9, 10, 11, 12):
            raise ValueError(f"bad host {host}")

        row = obs.reshape(1, -1)[0, 1:].reshape(13, 3)[host]

        # legal pre-condition: scanned but not yet exploited
        if (row == [1, 0, 0]).all():
            return {9: 26, 10: 27, 11: 28, 12: 29}[host]

        # idempotent success: already compromised or escalated
        if (row == [1, 1, 0]).all() or (row == [1, 0, 1]).all():
            return {9: 26, 10: 27, 11: 28, 12: 29}[host]

        return None                                             # can’t exploit yet

    def PrivilegeEscalate_user(self, host, obs):
        if host not in (9, 10, 11, 12):
            raise ValueError(f"bad host {host}")

        row = obs.reshape(1, -1)[0, 1:].reshape(13, 3)[host]

        # legal: exploited but not escalated
        if (row == [1, 1, 0]).all():
            return {9: 39, 10: 40, 11: 41, 12: 42}[host]

        # idempotent: already escalated
        if (row == [1, 0, 1]).all():
            return {9: 39, 10: 40, 11: 41, 12: 42}[host]

        return None                                             # can’t escalate yet

    # ───────────────── ENTERPRISE-SUBNET HELPERS (unchanged pre-checks) ─────────
    def DiscoverNetworkServices_ent(self, host, obs, last_host):
        if host not in (1, 2, 3):
            raise ValueError(f"bad host {host}")

        row = obs.reshape(1, -1)[0, 1:].reshape(13, 3)[host]

        if host in (1, 2):                                      # ent0/ent1 gated
            ok_prev = (obs.reshape(1, -1)[0, 1:].reshape(13, 3)[last_host] == [1,0,1]).all()
            if not ok_prev:
                return None

        if (row == [0, 0, 0]).all() or (row == [1, 0, 0]).all():
            return {1: 5, 2: 6, 3: 7}[host]
        return None

    def ExploitRemoteService_ent(self, host, obs):
        if host not in (1, 2, 3):
            raise ValueError(f"bad host {host}")

        row = obs.reshape(1, -1)[0, 1:].reshape(13, 3)[host]

        if (row == [1, 0, 0]).all():                            # legal first exploit
            return {1: 18, 2: 19, 3: 20}[host]
        if (row == [1, 1, 0]).all() or (row == [1, 0, 1]).all():                  # idempotent repeat
            return {1: 18, 2: 19, 3: 20}[host]
        return None

    def PrivilegeEscalate_ent(self, host, obs):
        if host not in (1, 2, 3):
            raise ValueError(f"bad host {host}")

        row = obs.reshape(1, -1)[0, 1:].reshape(13, 3)[host]

        if (row == [1, 1, 0]).all():
            return {1: 31, 2: 32, 3: 33}[host]
        if (row == [1, 0, 1]).all():                         # idempotent
            return {1: 31, 2: 32, 3: 33}[host]
        return None

    def DiscoverRemoteSystems_ent(self, host, obs):
        row = obs.reshape(1, -1)[0, 1:].reshape(13, 3)[host]
        if (row == [1, 0, 1]).all():                            # escalated host
            return 1                                            # DRS enterprise subnet
        return None

    # ───────────────── OP-SERVER HELPERS (same gating) ────────────────
    def DiscoverNetworkServices_op(self, host, obs):
        ok_prev = (obs.reshape(1, -1)[0, 1:].reshape(13, 3)[3] == [1,0,1]).all()
        if not ok_prev:
            return None
        row = obs.reshape(1, -1)[0, 1:].reshape(13, 3)[host]
        if (row == [0, 0, 0]).all():
            return 11
        return None

    def ExploitRemoteService_op(self, host, obs):
        row = obs.reshape(1, -1)[0, 1:].reshape(13, 3)[host]
        if (row == [1, 0, 0]).all() or row[0] == 1:
            return 24
        return None

    def PrivilegeEscalate_op(self, host, obs):
        row = obs.reshape(1, -1)[0, 1:].reshape(13, 3)[host]
        if (row == [1,1,0]).all() or (row[0]==1 and row[2]==1):
            return 37
        return None

    def Impact(self, host, obs):
        row = obs.reshape(1, -1)[0, 1:].reshape(13, 3)[host]
        if (row == [1,0,1]).all():
            return 50
        return None

    # ───────────────────────────  FSM  ────────────────────────────────
    def get_action(self, observation, success=None):
        succ = observation.reshape(1, -1)[0, 0]
        if success is False:
            succ = 0

        loops = 0
        while True:
            loops += 1
            if loops > 200:
                raise RuntimeError(f"stuck: state={self.action}, host={self.last_host}")

            # advance or jump
            self.action = min(self.action + 1, 14) if succ == 1 else self.jumps[self.action]

            # 0 ─ DiscoverRemoteSystems on user subnet
            if self.action == 0:
                action_id = self.DiscoverRemoteSystems_user()

            # 1 ─ DiscoverNetworkServices on a user host
            elif self.action == 1:
                if self.first_user_host is None:  # first time this episode
                    self.first_user_host = np.random.choice([9, 10, 11, 12])
                self.last_host = self.first_user_host
                action_id = self.DiscoverNetworkServices_user(self.last_host, observation)
                if action_id is None: succ = 0; continue

            # 2 ─ Exploit user host
            elif self.action == 2:
                if self.last_host not in (9, 10, 11, 12):  # guard – re-use the saved one
                    self.last_host = self.first_user_host

                action_id = self.ExploitRemoteService_user(self.last_host, observation)
                if action_id is None: succ = 0; continue
                # idempotent repeat counts as success
                if observation.reshape(1, -1)[0, 1:].reshape(13, 3)[self.last_host][0] == 1:
                    succ = 1

            # 3 ─ Privilege escalate user host
            elif self.action == 3:
                action_id = self.PrivilegeEscalate_user(self.last_host, observation)
                if action_id is None: succ = 0; continue
                if observation.reshape(1, -1)[0, 1:].reshape(13, 3)[self.last_host][2] == 1:
                    succ = 1

            # 4 – 14 : identical logic to your previous code
            # (enterprise, op-server, impact)  ──────────────
            elif self.action == 4:
                if self.last_host in (9,10): self.enterprise_host = 1
                else:                         self.enterprise_host = 2
                action_id = self.DiscoverNetworkServices_ent(self.enterprise_host, observation, self.last_host)
                if action_id is None: succ = 0; continue

            elif self.action == 5:
                self.last_host = self.enterprise_host
                action_id = self.ExploitRemoteService_ent(self.last_host, observation)
                if action_id is None: succ = 0; continue
                if observation.reshape(1, -1)[0, 1:].reshape(13, 3)[self.last_host][0] == 1:
                    succ = 1


            elif self.action == 6:
                action_id = self.PrivilegeEscalate_ent(self.last_host, observation)
                if action_id is None: succ = 0; continue
                if observation.reshape(1, -1)[0, 1:].reshape(13, 3)[self.last_host][2] == 1:
                    succ = 1

            elif self.action == 7:
                action_id = self.DiscoverRemoteSystems_ent(self.last_host, observation)
                if action_id is None: succ = 0; continue

            elif self.action == 8:
                action_id = self.DiscoverNetworkServices_ent(3, observation, self.last_host)
                if action_id is None: succ = 0; continue

            elif self.action == 9:
                self.last_host = 3
                action_id = self.ExploitRemoteService_ent(self.last_host, observation)
                if action_id is None: succ = 0; continue
                if observation.reshape(1, -1)[0, 1:].reshape(13, 3)[self.last_host][0] == 1:
                    succ = 1

            elif self.action == 10:
                action_id = self.PrivilegeEscalate_ent(self.last_host, observation)
                if action_id is None: succ = 0; continue
                if observation.reshape(1, -1)[0, 1:].reshape(13, 3)[self.last_host][2] == 1:
                    succ = 1

            elif self.action == 11:
                action_id = self.DiscoverNetworkServices_op(self.hosts['opserv'], observation)
                if action_id is None: succ = 0; continue

            elif self.action == 12:
                self.last_host = self.hosts['opserv']
                action_id = self.ExploitRemoteService_op(self.last_host, observation)
                if action_id is None: succ = 0; continue
                if observation.reshape(1, -1)[0, 1:].reshape(13, 3)[self.last_host][0] == 1:
                    succ = 1

            elif self.action == 13:
                action_id = self.PrivilegeEscalate_op(self.last_host, observation)
                if action_id is None: succ = 0; continue
                if observation.reshape(1, -1)[0, 1:].reshape(13, 3)[self.last_host][2] == 1:
                    succ = 1

            elif self.action == 14:
                action_id = self.Impact(self.last_host, observation)
                if action_id is None:
                    succ = 0
                    continue

            # ── return np.array([[id]]) like the simulator expects ─────
            self.last_action = np.array([[action_id]], dtype=np.int32)
            return self.last_action

    # ───────────────────────── housekeeping ───────────────────────────
    def reset(self):
        self.action = 0
        self.last_host = None
        self.last_action = None
        self.first_user_host = None  # first user host to scan, set on first DiscoverNetworkServices_user


# class B_line_minimal(Base_agent):
#     def __init__(self):
#         self.action = 0
#         self.last_host = None
#         self.last_action = None
#         # self.target_ip_address = None
#         # self.last_subnet = None
#         # self.last_ip_address = None
#         # self.action_history = {}
#         self.jumps = [0, 1, 2, 2, 2, 2, 5, 5, 5, 5, 9, 9, 9, 12, 13]
#
#         HOSTS = ['def', 'ent0', 'ent1', 'ent2', 'ophost0',
#                  'ophost1', 'ophost2', 'opserv', 'user0', 'user1', 'user2', 'user3', 'user4']
#         self.hosts = {host: index for index, host in enumerate(HOSTS)}
#
#     def DiscoverRemoteSystems_user(self):
#         # Action is to to DRS on user subnet
#         action = 3
#         return action
#
#     def DiscoverNetworkServices_user(self, host, observation):
#         # if host not in [9, 10, 11, 12]:
#         #     raise ValueError(f"Invalid host: {host} on DiscoverNetworkServices_user")
#         # obs = observation.reshape(1, -1)[0, 1:].reshape(13, 3)
#         # # return the correct action based on the host given
#         # # all the user hosts need to have red obs of [0,0,0] (successful scan but nothing else)
#         # if host == 9 and (obs[host, 0] == 0) & (obs[host, 1] == 0) and (obs[host, 2] == 0):
#         #     action = 13
#         # else:
#         #     return self.get_action(observation, success=False)
#         #
#         # if host == 10 and (obs[host, 0] == 0) & (obs[host, 1] == 0) and (obs[host, 2] == 0):
#         #     action = 14
#         # else:
#         #     return self.get_action(observation, success=False)
#         #
#         # if host == 11 and (obs[host, 0] == 0) & (obs[host, 1] == 0) and (obs[host, 2] == 0):
#         #     action = 15
#         # else:
#         #     return self.get_action(observation, success=False)
#         #
#         # if host == 12 and (obs[host, 0] == 0) & (obs[host, 1] == 0) and (obs[host, 2] == 0):
#         #     action = 16
#         # else:
#         #     return self.get_action(observation, success=False)
#
#         if host not in (9, 10, 11, 12):
#             raise ValueError(f"Invalid host: {host}")
#
#         obs = observation.reshape(1, -1)[0, 1:].reshape(13, 3)
#
#         # true when the three values at that row are all zero
#         is_clean = (obs[host] == 0).all()  # cleaner NumPy idiom
#
#         if not is_clean:
#             return None
#
#         # one-line lookup instead of four separate `if`s
#         return {9: 13, 10: 14, 11: 15, 12: 16}[host]
#
#         return action
#
#     def ExploitRemoteService_user(self, host, observation):
#         if host not in (9, 10, 11, 12):
#             raise ValueError(f"Invalid host: {host} in exploit_remote_service_user")
#
#         obs = observation.reshape(1, -1)[0, 1:].reshape(13, 3)
#         print(f"obs: {obs}")
#
#         print(f"obs[host]: {obs[host]} and host: {host}")
#
#         if (obs[host] == [1, 0, 0]).all():  # can still exploit
#             return {9: 26, 10: 27, 11: 28, 12: 29}[host]
#
#         if (obs[host] == [1, 1, 0]).all():  # already escalated → skip ahead
#             return {9: 40, 10: 41, 11: 42, 12: 43}[host]  # escalate action IDs
#
#         return None  # can’t act right now
#
#
#     def PrivilegeEscalate_user(self, host, observation):
#         if host not in (9, 10, 11, 12):
#             raise ValueError(f"Invalid host: {host} in exploit_remote_service_user")
#
#         obs = observation.reshape(1, -1)[0, 1:].reshape(13, 3)
#
#         # row must be exactly [1, 1, 0]
#         red_conditions = (obs[host] == [1, 1, 0]).all()
#         if not red_conditions:
#             return None
#
#         # -------- lookup table ------------ #
#         action_lookup = {9: 39, 10: 40, 11: 41, 12: 42}
#         return action_lookup[host]
#
#     def DiscoverNetworkServices_ent(self, host, observation, last_host):
#         if host not in (1, 2, 3):
#             raise ValueError(f"Invalid host: {host}")
#
#         obs = observation.reshape(1, -1)[0, 1:].reshape(13, 3)
#
#         if host in [1, 2]:
#             successful_last_escalate = (obs[last_host] == [1, 0, 1]).all()
#             if not successful_last_escalate:
#                 return None
#
#         # true when the three values at that row are all zero
#         is_clean = (obs[host] == 0).all()  # cleaner NumPy idiom
#         if not is_clean:
#             return None
#
#         self.last_host = self.enterprise_host
#         # one-line lookup instead of four separate `if`s
#         return {1: 5, 2: 6, 3: 7}[host]
#
#     def ExploitRemoteService_ent(self, host, observation):
#         # return the correct action based on the host given
#         # if host == 1:
#         #     action = 18
#         # elif host == 2:
#         #     action = 19
#         # elif host == 3:
#         #     action = 20
#         # else:
#         #     raise ValueError(f"Invalid host: {host} on ExploitRemoteService_ent")
#         #
#         # return action
#         if host not in (1, 2, 3):
#             raise ValueError(f"Invalid host: {host} in exploit_remote_service_user")
#
#         obs = observation.reshape(1, -1)[0, 1:].reshape(13, 3)
#
#         # print(f"obs[host]: {obs[host]}")
#
#         if host in [1, 2]:
#             red_conditions = (obs[host] == [1, 0, 0]).all()
#             if not red_conditions:
#                 return None
#         elif host == 3:
#             # row must be exactly [1, 0, 0]
#             red_conditions = (obs[host] == [1, 0, 0]).all()
#             if not red_conditions:
#                 return None
#
#         # -------- lookup table ------------ #
#         action_lookup = {1: 18, 2: 19, 3: 20}
#         return action_lookup[host]
#
#     def PrivilegeEscalate_ent(self, host, observation):
#         # return the correct action based on the host given
#         # if host == 1:
#         #     action = 31
#         # elif host == 2:
#         #     action = 32
#         # elif host == 3:
#         #     action = 33
#         # else:
#         #     raise ValueError(f"Invalid host: {host} on PrivilegeEscalate_ent")
#         #
#         # return action
#         if host not in (1, 2, 3):
#             raise ValueError(f"Invalid host: {host} in exploit_remote_service_user")
#
#         obs = observation.reshape(1, -1)[0, 1:].reshape(13, 3)
#
#         # row must be exactly [1, 1, 0]
#         red_conditions = (obs[host] == [1, 1, 0]).all()
#         if not red_conditions:
#             return None
#
#         # -------- lookup table ------------ #
#         action_lookup = {1: 31, 2: 32, 3: 33}
#         return action_lookup[host]
#
#     def DiscoverRemoteSystems_ent(self, host, observation):
#         if host not in (1, 2):
#             raise ValueError(f"Invalid host: {host} in exploit_remote_service_user")
#
#         obs = observation.reshape(1, -1)[0, 1:].reshape(13, 3)
#
#         # row must be exactly [1, 0, 1] indicating a previous successful escalate on the ent host
#         red_conditions = (obs[host] == [1, 0, 1]).all()
#         if not red_conditions:
#             return None
#
#         # Action is to DRS on enterprise subnet
#         action = 1
#
#         return action
#
#     def DiscoverNetworkServices_op(self, host, observation):
#         obs = observation.reshape(1, -1)[0, 1:].reshape(13, 3)
#
#         successful_last_escalate = (obs[3] == [1, 0, 1]).all()
#         if not successful_last_escalate:
#             return self.get_action(observation, success=False)
#
#         # true when the three values at that row are all zero
#         is_clean = (obs[host] == 0).all()  # cleaner NumPy idiom
#
#         if not is_clean:
#             return None
#
#         # Action is to DRS on operational subnet
#         action = 11
#         return action
#
#     def ExploitRemoteService_op(self, host, observation):
#         obs = observation.reshape(1, -1)[0, 1:].reshape(13, 3)
#
#         # row must be exactly [1, 0, 0]
#         red_conditions = (obs[host] == [1, 0, 0]).all()
#         if not red_conditions:
#             return None
#
#         # Action is to exploit on operational subnet
#         action = 24
#         return action
#
#     def PrivilegeEscalate_op(self, host, observation):
#         obs = observation.reshape(1, -1)[0, 1:].reshape(13, 3)
#
#         # row must be exactly [1, 1, 0]
#         red_conditions = (obs[host] == [1, 1, 0]).all()
#         if not red_conditions:
#             return None
#
#         # Action is to escalate on operational subnet
#         action = 37
#         return action
#
#     def Impact(self, host, observation):
#         obs = observation.reshape(1, -1)[0, 1:].reshape(13, 3)
#
#         # row must be exactly [1, 0, 1] (same as escalate)
#         red_conditions = (obs[host] == [1, 0, 1]).all()
#         if not red_conditions:
#             return None
#
#         # Action is to impact on operational server (impacts can only occur here)
#         action = 50
#         return action
#
#     def get_action(self, observation, success=None):
#         # print(self.action)
#         """gets an action from the agent that should be performed based on the agent's internal state and provided observation and action space"""
#         session = 0
#         succ = observation.reshape(1, -1)[0, 0]
#
#         if success is False:
#             succ = 0
#
#         loops = 0
#         while True:
#             loops += 1
#             if loops > 200:
#                 raise RuntimeError(f"stuck in get_action: action={self.action}, last_host={self.last_host}")
#             if succ == 1:
#                 self.action += 1 if self.action < 14 else 0
#             elif succ == 0 or succ == -1:
#                 self.action = self.jumps[self.action]
#
#             # if self.action in self.action_history:
#             # action = self.action_history[self.action]
#
#             # Discover Remote Systems
#             if self.action == 0:
#                 # self.initial_ip = observation['User0']['Interface'][0]['IP Address']
#                 # self.last_subnet = observation['User0']['Interface'][0]['Subnet']
#                 # user host 0 is the
#                 self.inital_host = self.hosts['user0']
#                 action = self.DiscoverRemoteSystems_user()
#             # Discover Network Services- new IP address found
#             elif self.action == 1:
#                 # hosts = [value for key, value in observation.items() if key != 'success']
#                 # get_ip = lambda x: x['Interface'][0]['IP Address']
#                 # interfaces = [get_ip(x) for x in hosts if get_ip(x) != self.initial_ip]
#                 # self.last_ip_address = random.choice(interfaces)
#                 user_hosts = [self.hosts['user1'], self.hosts['user2'], self.hosts['user3'], self.hosts['user4']]
#                 chosen_user = np.random.choice(user_hosts)
#
#                 # self.last_host = chosen_user
#                 self.last_host = 10
#                 action = self.DiscoverNetworkServices_user(self.last_host, observation)
#                 if action is None:
#                     succ = 0
#                     continue
#
#             # Exploit User1
#             elif self.action == 2:
#                 if self.last_host not in (9, 10, 11, 12):  # guard
#                     self.last_host = 10
#                 action = self.ExploitRemoteService_user(self.last_host, observation)
#                 if action is None:
#                     succ = 0
#                     continue
#
#                 # target already compromised?  → pretend success
#                 row = observation.reshape(1, -1)[0, 1:].reshape(13, 3)[self.last_host]
#                 if row[0] == 1:  # 1st flag = compromised
#                     succ = 1  # force success
#
#             # Privilege escalation on User Host
#             elif self.action == 3:
#                 # hostname = \
#                 # [value for key, value in observation.items() if key != 'success' and 'System info' in value][0][
#                 #     'System info']['Hostname']
#                 # action = PrivilegeEscalate(agent='Red', hostname=hostname, session=session)
#                 print(f'self.last_host: {self.last_host}')
#                 action = self.PrivilegeEscalate_user(self.last_host, observation)
#                 if action is None:
#                     succ = 0
#                     continue
#
#             # Discover Network Services- new IP address found
#             elif self.action == 4:
#                 # self.enterprise_host = [x for x in observation if 'Enterprise' in x][0]
#                 # self.last_ip_address = observation[self.enterprise_host]['Interface'][0]['IP Address']
#                 # action = DiscoverNetworkServices(session=session, agent='Red', ip_address=self.last_ip_address)
#
#                 if self.last_host in [self.hosts['user1'], self.hosts['user2']]:
#                     self.enterprise_host = self.hosts['ent0']
#                 elif self.last_host in [self.hosts['user3'], self.hosts['user4']]:
#                     self.enterprise_host = self.hosts['ent1']
#
#                 action = self.DiscoverNetworkServices_ent(self.enterprise_host, observation, self.last_host)
#                 if action is None:
#                     succ = 0
#                     continue
#
#             # Exploit- Enterprise Host
#             elif self.action == 5:
#                 # self.target_ip_address = \
#                 # [value for key, value in observation.items() if key != 'success'][0]['Interface'][0]['IP Address']
#                 # action = ExploitRemoteService(session=session, agent='Red', ip_address=self.target_ip_address)
#                 action = self.ExploitRemoteService_ent(self.last_host, observation)
#                 if action is None:
#                     succ = 0
#                     continue
#
#             # Privilege escalation on Enterprise Host
#             elif self.action == 6:
#                 # hostname = \
#                 # [value for key, value in observation.items() if key != 'success' and 'System info' in value][0][
#                 #     'System info']['Hostname']
#                 # action = PrivilegeEscalate(agent='Red', hostname=hostname, session=session)
#                 action = self.PrivilegeEscalate_ent(self.last_host, observation)
#                 if action is None:
#                     succ = 0
#                     continue
#
#             # Scanning the new subnet found.
#             elif self.action == 7:
#                 # self.last_subnet = observation[self.enterprise_host]['Interface'][0]['Subnet']
#                 # action = DiscoverRemoteSystems(subnet=self.last_subnet, agent='Red', session=session)
#                 action = self.DiscoverRemoteSystems_ent(self.last_host, observation)
#                 if action is None:
#                     succ = 0
#                     continue
#
#             # Discover Network Services- Enterprise2
#             elif self.action == 8:
#                 # self.target_ip_address = \
#                 # [value for key, value in observation.items() if key != 'success'][2]['Interface'][0]['IP Address']
#                 # action = DiscoverNetworkServices(session=session, agent='Red', ip_address=self.target_ip_address)
#                 action = self.DiscoverNetworkServices_ent(
#                     self.hosts['ent2'], observation, self.last_host)  # Discover Network Services on Enterprise2
#                 if action is None:
#                     succ = 0
#                     continue
#
#             # Exploit- Enterprise2
#             elif self.action == 9:
#                 # self.target_ip_address = \
#                 # [value for key, value in observation.items() if key != 'success'][0]['Interface'][0]['IP Address']
#                 # action = ExploitRemoteService(session=session, agent='Red', ip_address=self.target_ip_address)
#                 action = self.ExploitRemoteService_ent(self.hosts['ent2'], observation)  # Exploit Enterprise2
#                 if action is None:
#                     succ = 0
#                     continue
#
#             # Privilege escalation on Enterprise2
#             elif self.action == 10:
#                 # hostname = \
#                 # [value for key, value in observation.items() if key != 'success' and 'System info' in value][0][
#                 #     'System info']['Hostname']
#                 # action = PrivilegeEscalate(agent='Red', hostname=hostname, session=session)
#                 action = self.PrivilegeEscalate_ent(self.hosts['ent2'], observation)  # Privilege Escalate Enterprise2
#                 if action is None:
#                     succ = 0
#                     continue
#
#             # Discover Network Services- Op_Server0
#             elif self.action == 11:
#                 # action = DiscoverNetworkServices(session=session, agent='Red',
#                 #                                  ip_address=observation['Op_Server0']['Interface'][0]['IP Address'])
#                 action = self.DiscoverNetworkServices_op(self.hosts['opserv'], observation)
#                 if action is None:
#                     succ = 0
#                     continue
#
#             # Exploit- Op_Server0
#             elif self.action == 12:
#                 # info = [value for key, value in observation.items() if key != 'success']
#                 # if len(info) > 0:
#                 #     action = ExploitRemoteService(agent='Red', session=session,
#                 #                                   ip_address=info[0]['Interface'][0]['IP Address'])
#                 # else:
#                 #     self.action = 0
#                 #     continue
#                 action = self.ExploitRemoteService_op(self.hosts['opserv'], observation)
#                 if action is None:
#                     succ = 0
#                     continue
#
#             # Privilege escalation on Op_Server0
#             elif self.action == 13:
#                 action = self.PrivilegeEscalate_op(self.hosts['opserv'], observation)
#                 if action is None:
#                     succ = 0
#                     continue
#
#             # Impact on Op_server0
#             elif self.action == 14:
#                 action = self.Impact(self.hosts['opserv'], observation)
#                 if action is None:
#                     succ = 0
#                     continue
#
#             # if self.action not in self.action_history:
#             #     self.action_history[self.action] = action
#             action = np.array([[action]])
#             self.last_action = action
#             # print(f"action from B_line_minimal: {action}")
#             return action
#
#     def reset(self):
#         self.action = 0
#         self.last_host = None
#         self.last_action = None
#         # self.target_ip_address = None
#         # self.last_subnet = None
#         # self.last_ip_address = None
#         # self.action_history = {}
#
#     def set_initial_values(self, action_space, observation):
#         pass

# import numpy as np
# import random
# from CybORG_plus_plus.mini_CAGE.agents import Base_agent   # adjust import if needed


class B_line_minimal(Base_agent):
    # ───────────────────────── constructor ────────────────────────────
    def __init__(self):
        self.action       = 0
        self.last_host    = None          # index of most-recent target host
        self.last_action  = None          # np.array([[id]]) sent last step
        self.jumps        = [0, 1, 2, 2, 2, 2, 5, 5, 5, 5, 9, 9, 9, 12, 13]
        self.first_user_host = None

        HOSTS = [
            'def', 'ent0', 'ent1', 'ent2', 'ophost0',
            'ophost1', 'ophost2', 'opserv',
            'user0', 'user1', 'user2', 'user3', 'user4'
        ]
        self.hosts = {h: i for i, h in enumerate(HOSTS)}

    # ────────────────────  USER-SUBNET HELPERS  ───────────────────────
    def DiscoverRemoteSystems_user(self):
        return 3                                                # DRS user subnet

    def DiscoverNetworkServices_user(self, host, obs):
        if host not in (9, 10, 11, 12):
            raise ValueError(f"bad host {host}")

        row = obs.reshape(1, -1)[0, 1:].reshape(13, 3)[host]
        if (row == [0, 0, 0]).all():                            # not scanned yet
            return {9: 13, 10: 14, 11: 15, 12: 16}[host]
        return None                                             # already scanned

    def ExploitRemoteService_user(self, host, obs):
        if host not in (9, 10, 11, 12):
            raise ValueError(f"bad host {host}")

        row = obs.reshape(1, -1)[0, 1:].reshape(13, 3)[host]

        # legal pre-condition: scanned but not yet exploited
        if (row == [1, 0, 0]).all():
            return {9: 26, 10: 27, 11: 28, 12: 29}[host]

        # idempotent success: already compromised or escalated
        if (row == [1, 1, 0]).all() or (row == [1, 0, 1]).all():
            return {9: 26, 10: 27, 11: 28, 12: 29}[host]

        return None                                             # can’t exploit yet

    def PrivilegeEscalate_user(self, host, obs):
        if host not in (9, 10, 11, 12):
            raise ValueError(f"bad host {host}")

        row = obs.reshape(1, -1)[0, 1:].reshape(13, 3)[host]

        # legal: exploited but not escalated
        if (row == [1, 1, 0]).all():
            return {9: 39, 10: 40, 11: 41, 12: 42}[host]

        # idempotent: already escalated
        if (row == [1, 0, 1]).all():
            return {9: 39, 10: 40, 11: 41, 12: 42}[host]

        return None                                             # can’t escalate yet

    # ───────────────── ENTERPRISE-SUBNET HELPERS (unchanged pre-checks) ─────────
    def DiscoverNetworkServices_ent(self, host, obs, last_host):
        if host not in (1, 2, 3):
            raise ValueError(f"bad host {host}")

        row = obs.reshape(1, -1)[0, 1:].reshape(13, 3)[host]

        if host in (1, 2):                                      # ent0/ent1 gated
            ok_prev = (obs.reshape(1, -1)[0, 1:].reshape(13, 3)[last_host] == [1,0,1]).all()
            if not ok_prev:
                return None

        if (row == [0, 0, 0]).all() or (row == [1, 0, 0]).all():
            return {1: 5, 2: 6, 3: 7}[host]
        return None

    def ExploitRemoteService_ent(self, host, obs):
        if host not in (1, 2, 3):
            raise ValueError(f"bad host {host}")

        row = obs.reshape(1, -1)[0, 1:].reshape(13, 3)[host]

        if (row == [1, 0, 0]).all():                            # legal first exploit
            return {1: 18, 2: 19, 3: 20}[host]
        if (row == [1, 1, 0]).all() or (row == [1, 0, 1]).all():                  # idempotent repeat
            return {1: 18, 2: 19, 3: 20}[host]
        return None

    def PrivilegeEscalate_ent(self, host, obs):
        if host not in (1, 2, 3):
            raise ValueError(f"bad host {host}")

        row = obs.reshape(1, -1)[0, 1:].reshape(13, 3)[host]

        if (row == [1, 1, 0]).all():
            return {1: 31, 2: 32, 3: 33}[host]
        if (row == [1, 0, 1]).all():                         # idempotent
            return {1: 31, 2: 32, 3: 33}[host]
        return None

    def DiscoverRemoteSystems_ent(self, host, obs):
        row = obs.reshape(1, -1)[0, 1:].reshape(13, 3)[host]
        if (row == [1, 0, 1]).all():                            # escalated host
            return 1                                            # DRS enterprise subnet
        return None

    # ───────────────── OP-SERVER HELPERS (same gating) ────────────────
    def DiscoverNetworkServices_op(self, host, obs):
        ok_prev = (obs.reshape(1, -1)[0, 1:].reshape(13, 3)[3] == [1,0,1]).all()
        if not ok_prev:
            return None
        row = obs.reshape(1, -1)[0, 1:].reshape(13, 3)[host]
        if (row == [0, 0, 0]).all():
            return 11
        return None

    def ExploitRemoteService_op(self, host, obs):
        row = obs.reshape(1, -1)[0, 1:].reshape(13, 3)[host]
        if (row == [1, 0, 0]).all() or row[0] == 1:
            return 24
        return None

    def PrivilegeEscalate_op(self, host, obs):
        row = obs.reshape(1, -1)[0, 1:].reshape(13, 3)[host]
        if (row == [1,1,0]).all() or (row[0]==1 and row[2]==1):
            return 37
        return None

    def Impact(self, host, obs):
        row = obs.reshape(1, -1)[0, 1:].reshape(13, 3)[host]
        if (row == [1,0,1]).all():
            return 50
        return None

    # ───────────────────────────  FSM  ────────────────────────────────
    def get_action(self, observation, success=None):
        succ = observation.reshape(1, -1)[0, 0]
        if success is False:
            succ = 0

        loops = 0
        while True:
            loops += 1
            if loops > 200:
                raise RuntimeError(f"stuck: state={self.action}, host={self.last_host}")

            # advance or jump
            self.action = min(self.action + 1, 14) if succ == 1 else self.jumps[self.action]

            # 0 ─ DiscoverRemoteSystems on user subnet
            if self.action == 0:
                action_id = self.DiscoverRemoteSystems_user()

            # 1 ─ DiscoverNetworkServices on a user host
            elif self.action == 1:
                if self.first_user_host is None:  # first time this episode
                    self.first_user_host = np.random.choice([9, 10, 11, 12])
                self.last_host = self.first_user_host
                action_id = self.DiscoverNetworkServices_user(self.last_host, observation)
                if action_id is None: succ = 0; continue

            # 2 ─ Exploit user host
            elif self.action == 2:
                if self.last_host not in (9, 10, 11, 12):  # guard – re-use the saved one
                    self.last_host = self.first_user_host

                action_id = self.ExploitRemoteService_user(self.last_host, observation)
                if action_id is None: succ = 0; continue
                # idempotent repeat counts as success
                if observation.reshape(1, -1)[0, 1:].reshape(13, 3)[self.last_host][0] == 1:
                    succ = 1

            # 3 ─ Privilege escalate user host
            elif self.action == 3:
                action_id = self.PrivilegeEscalate_user(self.last_host, observation)
                if action_id is None: succ = 0; continue
                if observation.reshape(1, -1)[0, 1:].reshape(13, 3)[self.last_host][2] == 1:
                    succ = 1

            # 4 – 14 : identical logic to your previous code
            # (enterprise, op-server, impact)  ──────────────
            elif self.action == 4:
                if self.last_host in (9,10): self.enterprise_host = 1
                else:                         self.enterprise_host = 2
                action_id = self.DiscoverNetworkServices_ent(self.enterprise_host, observation, self.last_host)
                if action_id is None: succ = 0; continue

            elif self.action == 5:
                self.last_host = self.enterprise_host
                action_id = self.ExploitRemoteService_ent(self.last_host, observation)
                if action_id is None: succ = 0; continue
                if observation.reshape(1, -1)[0, 1:].reshape(13, 3)[self.last_host][0] == 1:
                    succ = 1


            elif self.action == 6:
                action_id = self.PrivilegeEscalate_ent(self.last_host, observation)
                if action_id is None: succ = 0; continue
                if observation.reshape(1, -1)[0, 1:].reshape(13, 3)[self.last_host][2] == 1:
                    succ = 1

            elif self.action == 7:
                action_id = self.DiscoverRemoteSystems_ent(self.last_host, observation)
                if action_id is None: succ = 0; continue

            elif self.action == 8:
                action_id = self.DiscoverNetworkServices_ent(3, observation, self.last_host)
                if action_id is None: succ = 0; continue

            elif self.action == 9:
                self.last_host = 3
                action_id = self.ExploitRemoteService_ent(self.last_host, observation)
                if action_id is None: succ = 0; continue
                if observation.reshape(1, -1)[0, 1:].reshape(13, 3)[self.last_host][0] == 1:
                    succ = 1

            elif self.action == 10:
                action_id = self.PrivilegeEscalate_ent(self.last_host, observation)
                if action_id is None: succ = 0; continue
                if observation.reshape(1, -1)[0, 1:].reshape(13, 3)[self.last_host][2] == 1:
                    succ = 1

            elif self.action == 11:
                action_id = self.DiscoverNetworkServices_op(self.hosts['opserv'], observation)
                if action_id is None: succ = 0; continue

            elif self.action == 12:
                self.last_host = self.hosts['opserv']
                action_id = self.ExploitRemoteService_op(self.last_host, observation)
                if action_id is None: succ = 0; continue
                if observation.reshape(1, -1)[0, 1:].reshape(13, 3)[self.last_host][0] == 1:
                    succ = 1

            elif self.action == 13:
                action_id = self.PrivilegeEscalate_op(self.last_host, observation)
                if action_id is None: succ = 0; continue
                if observation.reshape(1, -1)[0, 1:].reshape(13, 3)[self.last_host][2] == 1:
                    succ = 1

            elif self.action == 14:
                action_id = self.Impact(self.last_host, observation)
                if action_id is None:
                    succ = 0
                    continue

            # ── return np.array([[id]]) like the simulator expects ─────
            self.last_action = np.array([[action_id]], dtype=np.int32)
            return self.last_action

    # ───────────────────────── housekeeping ───────────────────────────
    def reset(self):
        self.action = 0
        self.last_host = None
        self.last_action = None
        self.first_user_host = None  # first user host to scan, set on first DiscoverNetworkServices_user

