import random
import numpy as np

class Base_agent:
    def __init__(self): pass
    def train(self): pass
    def get_action(self): pass
    def end_episode(self): pass
    def set_initial_values(self): pass

class Meander_minimal(Base_agent):
    """
    Mini-CAGE red 'meander' with:
      - DNS all users → PE user0
      - Random exploit pool of any [1,0,0] host (users, ent0/1/2, opserv, **defender included for EXP only**)
      - After EXP on a host, immediately PE that host (except defender: never PE)
      - Branch hints:
          * if primary is user0 → EXP+PE another user (≠ user0)
          * if primary in {user1,user2} → hint ent0
          * if primary in {user3,user4} → hint ent1
      - After DRS_ent, bulk DNS (shuffled) for ent0, ent1, ent2, def (only those still [0,0,0])
      - After ent2 is escalated → immediately DNS opserv; opserv then joins the exploit pool
      - If a PE action fails, next turn return to random exploit pool and include that same host in the pool
      - **NEW**: Defender can be selected for EXP from the pool; after EXP(def) we immediately return to the pool (no PE)
    """

    def __init__(self):
        super().__init__()
        self.last_host = None
        self.last_action = None

        hosts = [
            'def', 'ent0', 'ent1', 'ent2', 'ophost0',
            'ophost1', 'ophost2', 'opserv',
            'user0', 'user1', 'user2', 'user3', 'user4'
        ]
        self.hosts = {h: i for i, h in enumerate(hosts)}
        self.inv_hosts = {i: h for h, i in self.hosts.items()}

        self.foothold = {'DNS': False, 'PE': False, 'EXP': False}

        self.host_vector_dict = {
            "unknown": np.array([-1, -1, -1]),
            "subnet_known": np.array([0, 0, 0]),
            "host_known_no_exploit": np.array([1, 0, 0]),
            "exploited": np.array([1, 1, 0]),
            "escalated": np.array([1, 0, 1]),
            "red_user0_foothold": np.array([0, 0, 1]),
        }

        # FSM state
        self.phase = "dns_users_then_pe_user0"
        self.primary_user = None
        self.secondary_user = None
        self.branch_ent = None
        self.pending_dns_enterprise = False

        # Track a host to force-include in exploit pool after a failed PE
        self.failed_retry_host = None

        # Debug trackers
        self.debug = True
        self.last_candidate_pool = []
        self.last_candidate_pool_shuffled = []
        self.last_dns_candidates_enterprise = []

    # ---------- debug helpers ----------
    def dbg(self, *args):
        if self.debug:
            print("[Meander]", *args)

    def _name(self, h_idx):
        return self.inv_hosts.get(h_idx, f"h{h_idx}")

    # ---------- basic helpers ----------
    def _rows(self, obs):
        v = obs.reshape(1, -1)[0]
        return int(v[0]), v[1:].reshape(13, 3)

    def _row(self, h, rows):
        return rows[h]

    def _rowstr(self, h, rows):
        r = self._row(h, rows)
        return f"{self._name(h)} row={r.tolist()}"

    def _any_unknown(self, rows, group):
        return any(rows[h][0] == -1 for h in group)

    def _is_def(self, h):
        return h == self.hosts['def']

    # ---------- action encoders ----------
    def DRS_user(self): return 3   # remote_sub2
    def DRS_ent(self):  return 1   # remote_sub0
    def DRS_op(self):   return 2   # remote_sub1

    def DNS_id(self, h):
        m = self.hosts
        return {
            m['def']: 4,  m['ent0']: 5,  m['ent1']: 6,  m['ent2']: 7,
            m['ophost0']: 8, m['ophost1']: 9, m['ophost2']: 10, m['opserv']: 11,
            m['user0']: 12, m['user1']: 13, m['user2']: 14, m['user3']: 15, m['user4']: 16
        }[h]

    def EXP_id(self, h):
        m = self.hosts
        if h == m['user0']:
            self.foothold['EXP'] = True
            return 25
        return {
            m['def']: 17, m['ent0']: 18, m['ent1']: 19, m['ent2']: 20,
            m['ophost0']: 21, m['ophost1']: 22, m['ophost2']: 23, m['opserv']: 24,
            m['user1']: 26, m['user2']: 27, m['user3']: 28, m['user4']: 29
        }[h]

    def PE_id(self, h):
        m = self.hosts
        if h == m['user0']:
            self.foothold['PE'] = True
            return 38
        return {
            m['def']: 30, m['ent0']: 31, m['ent1']: 32, m['ent2']: 33,
            m['ophost0']: 34, m['ophost1']: 35, m['ophost2']: 36, m['opserv']: 37,
            m['user1']: 39, m['user2']: 40, m['user3']: 41, m['user4']: 42
        }.get(h, None)

    def IMPACT_id(self, h):
        m = self.hosts
        return {
            m['def']: 43, m['ent0']: 44, m['ent1']: 45, m['ent2']: 46,
            m['ophost0']: 47, m['ophost1']: 48, m['ophost2']: 49, m['opserv']: 50,
            m['user0']: 51, m['user1']: 52, m['user2']: 53, m['user3']: 54, m['user4']: 55
        }[h]

    # ---------- legality ----------
    def can_dns(self, h, rows):
        if h == self.hosts['user0'] and not self.foothold['DNS']:
            self.dbg("Allowing one-time DNS on user0 for parity with original.")
            self.foothold['DNS'] = True
            return True
        ok = (rows[h] == self.host_vector_dict["subnet_known"]).all()
        self.dbg(f"can_dns? {self._rowstr(h, rows)} -> {ok}")
        return ok

    def can_exploit(self, h, rows):
        # Defender now allowed for EXP (may fail, that’s fine)
        r = rows[h]
        ok = (r == self.host_vector_dict["host_known_no_exploit"]).all() \
             or (r == self.host_vector_dict["exploited"]).all() \
             or (r == self.host_vector_dict["escalated"]).all()
        self.dbg(f"can_exploit? {self._rowstr(h, rows)} -> {ok}")
        return ok

    def can_pe(self, h, rows):
        # Defender may NOT be escalated
        if self._is_def(h):
            self.dbg(f"can_pe? {self._name(h)} -> False (defender excluded)")
            return False
        r = rows[h]
        ok = (r == self.host_vector_dict["exploited"]).all() \
             or (r == self.host_vector_dict["escalated"]).all() \
             or (h == self.hosts['user0'] and (r == self.host_vector_dict["red_user0_foothold"]).all())
        self.dbg(f"can_pe? {self._rowstr(h, rows)} -> {ok}")
        return ok

    def can_impact(self, h, rows):
        ok = (rows[h] == self.host_vector_dict["escalated"]).all()
        self.dbg(f"can_impact? {self._rowstr(h, rows)} -> {ok}")
        return ok

    # ---------- policy ----------
    def get_action(self, observation, success=None):
        succ, rows = self._rows(observation)
        if success is False:
            succ = 0
        self.dbg(f"--- STEP --- phase={self.phase} succ={succ}")

        op = self.hosts['opserv']
        ent0, ent1, ent2, dfd = self.hosts['ent0'], self.hosts['ent1'], self.hosts['ent2'], self.hosts['def']
        users = [self.hosts[f'user{i}'] for i in range(5)]

        # Handle failed PE: include that host in exploit pool next
        if succ == 0 and self.last_action is not None:
            last_id = int(self.last_action[0, 0])
            was_pe = (30 <= last_id <= 42)
            if was_pe:
                self.failed_retry_host = self.last_host
                self.dbg(f"PE failed on {self._name(self.failed_retry_host)} -> switch to select_exploit "
                         f"and include this host in candidate pool.")
                self.phase = "select_exploit"

        # Impact opserv if escalated
        if self.can_impact(op, rows):
            aid = self.IMPACT_id(op)
            self.dbg("IMPACT on opserv (50).")
            self.last_host = op
            self.last_action = np.array([[aid]], dtype=np.int32)
            return self.last_action

        # After ent2 escalated, DNS opserv immediately if not DNSed
        if rows[ent2][2] == 1:
            if (rows[op] == self.host_vector_dict["subnet_known"]).all():
                aid = self.DNS_id(op)
                self.dbg("ent2 escalated AND opserv is [0,0,0] -> DNS opserv now (11).")
                self.last_host = op
                self.last_action = np.array([[aid]], dtype=np.int32)
                return self.last_action

        # DRS user subnet if needed
        if self._any_unknown(rows, users) and self.phase == "dns_users_then_pe_user0":
            self.dbg("User subnet unknown -> DRS_user (3).")
            self.last_action = np.array([[self.DRS_user()]], dtype=np.int32)
            return self.last_action

        # Phase 1: DNS all users, then PE user0
        if self.phase == "dns_users_then_pe_user0":
            random.shuffle(users)
            for h in users:
                if self.can_dns(h, rows):
                    aid = self.DNS_id(h)
                    self.dbg(f"DNS user host -> {self._name(h)} action={aid}")
                    self.last_host = h
                    self.last_action = np.array([[aid]], dtype=np.int32)
                    return self.last_action
            u0 = self.hosts['user0']
            aid = self.PE_id(u0)
            self.dbg("All users DNSed -> PE user0 immediately (38).")
            self.last_host = u0
            self.last_action = np.array([[aid]], dtype=np.int32)
            self.phase = "choose_user"
            return self.last_action

        # Pending enterprise bulk DNS
        if self.pending_dns_enterprise:
            self.dbg("Pending enterprise bulk DNS -> dns_enterprise_bulk phase.")
            self.phase = "dns_enterprise_bulk"

        if self.phase == "dns_enterprise_bulk":
            ent_targets = [ent0, ent1, ent2, dfd]
            random.shuffle(ent_targets)
            self.dbg("Enterprise bulk-DNS order (shuffled):", [self._name(e) for e in ent_targets])
            for e in ent_targets:
                if (rows[e] == self.host_vector_dict["subnet_known"]).all():
                    aid = self.DNS_id(e)
                    self.dbg(f"Bulk DNS enterprise -> {self._name(e)} action={aid}")
                    self.last_host = e
                    self.last_action = np.array([[aid]], dtype=np.int32)
                    return self.last_action
            self.dbg("Bulk DNS enterprise complete; returning to select_exploit.")
            self.pending_dns_enterprise = False
            self.phase = "select_exploit"

        # Branch: possibly DRS_ent (to reveal ent2) else DNS branch target if needed
        if self.phase == "branch":
            ent_escalated = (rows[ent0][2] == 1) or (rows[ent1][2] == 1)
            need_ent_drs = ent_escalated and (rows[ent2][0] == -1)
            self.dbg(f"Enterprise DRS gate in branch: ent_escalated={ent_escalated} "
                     f"ent2_row={rows[ent2].tolist()} need_ent_drs={need_ent_drs}")
            if need_ent_drs:
                self.dbg("-> DRS_ent (1) due to ent2 unknown & gate satisfied. Enabling bulk DNS.")
                self.pending_dns_enterprise = True
                self.last_action = np.array([[self.DRS_ent()]], dtype=np.int32)
                return self.last_action

            e = self.branch_ent
            if (rows[e] == self.host_vector_dict["subnet_known"]).all():
                aid = self.DNS_id(e)
                self.dbg(f"Branch DNS enterprise -> {self._name(e)} action={aid}")
                self.last_host = e
                self.last_action = np.array([[aid]], dtype=np.int32)
                self.phase = "select_exploit"
                return self.last_action
            else:
                self.dbg(f"Branch enterprise already DNSed -> return to pool. {self._rowstr(e, rows)}")
                self.phase = "select_exploit"

        # Phase hop
        if self.phase == "choose_user":
            self.dbg("Phase change: choose_user -> select_exploit")
            self.primary_user = None
            self.secondary_user = None
            self.phase = "select_exploit"

        # Build & use exploit pool (now includes defender)
        if self.phase == "select_exploit":
            ent_escalated = (rows[ent0][2] == 1) or (rows[ent1][2] == 1)
            need_ent_drs = ent_escalated and (rows[ent2][0] == -1)
            self.dbg(f"Enterprise DRS gate in select_exploit: ent_escalated={ent_escalated} "
                     f"ent2_row={rows[ent2].tolist()} need_ent_drs={need_ent_drs}")
            if need_ent_drs:
                self.dbg("-> DRS_ent (1) in select_exploit to reveal ent2. Enabling bulk DNS.")
                self.pending_dns_enterprise = True
                self.last_action = np.array([[self.DRS_ent()]], dtype=np.int32)
                return self.last_action

            candidates = []
            # users
            for h in users:
                if (rows[h] == self.host_vector_dict["host_known_no_exploit"]).all():
                    candidates.append(h)
            # enterprises + opserv + **defender included here**
            for e in [ent0, ent1, ent2, op, dfd]:
                if (rows[e] == self.host_vector_dict["host_known_no_exploit"]).all():
                    candidates.append(e)

            dns_candidates_ent = []
            for e in [ent0, ent1, ent2, dfd]:
                if (rows[e] == self.host_vector_dict["subnet_known"]).all():
                    dns_candidates_ent.append(e)

            if self.failed_retry_host is not None and not self._is_def(self.failed_retry_host):
                if self.failed_retry_host not in candidates:
                    self.dbg(f"Including failed-PE host in exploit pool: {self._name(self.failed_retry_host)}")
                    candidates.append(self.failed_retry_host)
                else:
                    self.dbg(f"Failed-PE host already in pool: {self._name(self.failed_retry_host)}")
                self.failed_retry_host = None

            self.last_candidate_pool = [self._name(h) for h in candidates]
            self.last_dns_candidates_enterprise = [self._name(h) for h in dns_candidates_ent]

            self.dbg("Exploit candidate pool (pre-shuffle):", self.last_candidate_pool)
            if dns_candidates_ent:
                self.dbg("Enterprise DNS candidates (still [0,0,0]):", self.last_dns_candidates_enterprise)
            self.dbg("Note: defender is INCLUDED in EXP pool but still EXCLUDED from PE.")

            if len(candidates) == 0:
                # no [1,0,0] targets → try PE any exploited-but-not-escalated (excluding defender)
                breadth = users + [ent0, ent1, ent2, op]
                for h in breadth:
                    if (rows[h] == self.host_vector_dict["exploited"]).all() and not self._is_def(h):
                        aid = self.PE_id(h)
                        self.dbg(f"No [1,0,0] candidates; PE exploited host -> {self._name(h)} action={aid}")
                        self.last_host = h
                        self.last_action = np.array([[aid]], dtype=np.int32)
                        self.phase = "choose_user"
                        return self.last_action
                raise NotImplementedError(
                    f"No exploitable hosts [1,0,0] and no exploited hosts to PE. "
                    f"phase={self.phase}, ent2_row={rows[self.hosts['ent2']].tolist()}"
                )

            random.shuffle(candidates)
            self.last_candidate_pool_shuffled = [self._name(h) for h in candidates]
            self.dbg("Exploit candidate pool (shuffled):", self.last_candidate_pool_shuffled)

            self.primary_user = candidates[0]
            self.dbg(f"Selected primary target to exploit: {self._name(self.primary_user)}")
            self.phase = "exp_user" if self.primary_user in users else "exp_ent"

        # EXP chosen user
        if self.phase == "exp_user":
            h = self.primary_user
            if (rows[h] == self.host_vector_dict["subnet_known"]).all():
                aid = self.DNS_id(h)
                self.dbg(f"Primary user is [0,0,0]; DNS first -> {self._name(h)} action={aid}")
                self.last_host = h
                self.last_action = np.array([[aid]], dtype=np.int32)
                return self.last_action

            if self.can_exploit(h, rows) and rows[h][1] == 0:
                if h == self.hosts['user0'] and self.foothold['EXP'] is True:
                    self.dbg("user0 EXP was flagged earlier; skip EXP, go to PE.")
                else:
                    aid = self.EXP_id(h)
                    self.dbg(f"EXP user -> {self._name(h)} action={aid}")
                    self.last_host = h
                    self.last_action = np.array([[aid]], dtype=np.int32)
                    self.phase = "pe_user"
                    return self.last_action

            self.dbg(f"User already exploited or cannot exploit; advancing to PE: {self._name(h)}")
            self.phase = "pe_user"

        # PE chosen user, then set branch hint
        if self.phase == "pe_user":
            h = self.primary_user
            if self.can_pe(h, rows):
                if rows[h][2] == 0:
                    aid = self.PE_id(h)
                    self.dbg(f"PE user -> {self._name(h)} action={aid}")
                    self.last_host = h
                    self.last_action = np.array([[aid]], dtype=np.int32)
                else:
                    self.dbg(f"User already escalated; idempotent PE -> continue: {self._rowstr(h, rows)}")
                    self.last_action = np.array([[self.PE_id(h)]], dtype=np.int32)

                if h == self.hosts['user0']:
                    others = [u for u in [self.hosts[f'user{i}'] for i in range(5)] if u != self.hosts['user0']]
                    random.shuffle(others)
                    self.secondary_user = others[0]
                    self.dbg(f"Primary was user0; secondary pick -> {self._name(self.secondary_user)}")
                    self.phase = "exp_user2"
                elif h in (self.hosts['user1'], self.hosts['user2']):
                    self.branch_ent = ent0
                    self.dbg("Primary in {user1,user2}; branch -> ent0.")
                    self.phase = "branch"
                elif h in (self.hosts['user3'], self.hosts['user4']):
                    self.branch_ent = ent1
                    self.dbg("Primary in {user3,user4}; branch -> ent1.")
                    self.phase = "branch"
                else:
                    self.branch_ent = ent0
                    self.dbg("Primary unknown; default branch -> ent0.")
                    self.phase = "branch"
                return self.last_action

            self.dbg(f"PE precondition not met for {self._name(h)}; fallback to select_exploit.")
            self.failed_retry_host = h
            self.phase = "select_exploit"
            return self.get_action(observation, success=succ)

        # If primary was user0, EXP+PE another random user (≠ user0)
        if self.phase == "exp_user2":
            h = self.secondary_user
            if (rows[h] == self.host_vector_dict["subnet_known"]).all():
                aid = self.DNS_id(h)
                self.dbg(f"Secondary user is [0,0,0]; DNS first -> {self._name(h)} action={aid}")
                self.last_host = h
                self.last_action = np.array([[aid]], dtype=np.int32)
                return self.last_action
            if self.can_exploit(h, rows) and rows[h][1] == 0:
                aid = self.EXP_id(h)
                self.dbg(f"EXP secondary user -> {self._name(h)} action={aid}")
                self.last_host = h
                self.last_action = np.array([[aid]], dtype=np.int32)
                self.phase = "pe_user2"
                return self.last_action
            self.dbg("Secondary already exploited or cannot exploit; go PE secondary.")
            self.phase = "pe_user2"

        if self.phase == "pe_user2":
            h = self.secondary_user
            if self.can_pe(h, rows):
                if rows[h][2] == 0:
                    aid = self.PE_id(h)
                    self.dbg(f"PE secondary user -> {self._name(h)} action={aid}")
                    self.last_host = h
                    self.last_action = np.array([[aid]], dtype=np.int32)
                else:
                    self.dbg(f"Secondary already escalated; continue: {self._rowstr(h, rows)}")
                    self.last_action = np.array([[self.PE_id(h)]], dtype=np.int32)
                self.branch_ent = ent0
                self.dbg("After secondary, branch -> ent0.")
                self.phase = "branch"
                return self.last_action

            self.dbg(f"PE secondary precondition not met for {self._name(h)}; fallback to select_exploit.")
            self.failed_retry_host = h
            self.phase = "select_exploit"
            return self.get_action(observation, success=succ)

        # EXP/PE chosen enterprise (includes defender EXP-only behavior)
        if self.phase == "exp_ent":
            e = self.primary_user
            # DNS first if needed
            if (rows[e] == self.host_vector_dict["subnet_known"]).all():
                aid = self.DNS_id(e)
                self.dbg(f"Primary enterprise is [0,0,0]; DNS first -> {self._name(e)} action={aid}")
                self.last_host = e
                self.last_action = np.array([[aid]], dtype=np.int32)
                return self.last_action

            # If defender selected → allow EXP, then immediately return to pool (never PE)
            if self._is_def(e):
                if self.can_exploit(e, rows) and rows[e][1] == 0:
                    aid = self.EXP_id(e)  # 17
                    self.dbg(f"EXP defender -> {self._name(e)} action={aid} (will fail by env).")
                    self.last_host = e
                    self.last_action = np.array([[aid]], dtype=np.int32)
                    # No PE for defender; go straight back to pool next step
                    self.phase = "select_exploit"
                    return self.last_action
                self.dbg("Defender not ready for EXP or already tried; back to pool.")
                self.phase = "select_exploit"
                return self.get_action(observation, success=succ)

            # Normal enterprise EXP
            if self.can_exploit(e, rows) and rows[e][1] == 0:
                aid = self.EXP_id(e)
                self.dbg(f"EXP enterprise -> {self._name(e)} action={aid}")
                self.last_host = e
                self.last_action = np.array([[aid]], dtype=np.int32)
                self.phase = "pe_ent"
                return self.last_action
            self.dbg("Enterprise already exploited or cannot exploit; go PE enterprise.")
            self.phase = "pe_ent"

        if self.phase == "pe_ent":
            e = self.primary_user if isinstance(self.primary_user, int) else self.branch_ent

            # Never PE defender
            if self._is_def(e):
                self.dbg("Enterprise target is defender; skipping PE, returning to select_exploit.")
                self.phase = "select_exploit"
                return self.get_action(observation, success=succ)

            if self.can_pe(e, rows):
                if rows[e][2] == 0:
                    aid = self.PE_id(e)
                    self.dbg(f"PE enterprise -> {self._name(e)} action={aid}")
                    self.last_host = e
                    self.last_action = np.array([[aid]], dtype=np.int32)
                else:
                    self.dbg(f"Enterprise already escalated; continue: {self._rowstr(e, rows)}")
                    self.last_action = np.array([[self.PE_id(e)]], dtype=np.int32)

                self.dbg("Enterprise step complete; returning to select_exploit.")
                self.phase = "select_exploit"
                self.primary_user = None
                return self.last_action

            self.dbg(f"PE enterprise precondition not met for {self._name(e)}; fallback to select_exploit.")
            self.failed_retry_host = e
            self.phase = "select_exploit"
            return self.get_action(observation, success=succ)

        raise NotImplementedError(f"Reached end of get_action with no action. phase={self.phase}")

    def reset(self):
        self.last_host = None
        self.last_action = None
        self.foothold = {'DNS': False, 'PE': False, 'EXP': False}
        self.phase = "dns_users_then_pe_user0"
        self.primary_user = None
        self.secondary_user = None
        self.branch_ent = None
        self.pending_dns_enterprise = False
        self.failed_retry_host = None
        self.last_candidate_pool = []
        self.last_candidate_pool_shuffled = []
        self.last_dns_candidates_enterprise = []
        self.dbg("Reset agent.")