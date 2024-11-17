import time
import cv2
import numpy

from gymnasium.spaces import Discrete, Box

'''
    multiple robots are gathering targets
    when target is reached, reward +1 is generated
    episode ends when all targets are reaches, or 1000 steps passed

    when n_targets and n_robots the state and actions shapes are : 
    state   : (n_targets + n_robots), 3)
    actions : (4, )

    reset and step returns states for all robots in shape : (n_robots, n_targets + n_robots), 3)
    step action is expected in shape : (n_robots, )
'''
class RobotTargetsEnv:

    def __init__(self, n_envs, render_mode = None, n_robots = 4, n_targets = 40, tau = 0.9, dt = 0.025,  max_steps = 1000):
        self.n_envs      = n_envs
        self.render_mode = render_mode
        self.n_robots    = n_robots
        self.n_targets   = n_targets
        self.tau         = tau
        self.dt          = dt

        self.max_steps   = max_steps
        self.steps       = 0    

        self.action_space      = Discrete(4)    

        self.observation_shape = ((n_targets + n_robots), 3)
        self.observation_space = Box(low = numpy.zeros(self.observation_shape), 
                                     high = numpy.ones(self.observation_shape),
                                     dtype = numpy.float32)

        self.init()
        

    def init(self):
        # intial initial robots position
        self.robots_vel = numpy.zeros((self.n_envs, self.n_robots, 2))
        self.robots_pos = numpy.zeros((self.n_envs, self.n_robots, 2))
        
        # intial targets positions
        self.targets_pos   = numpy.zeros((self.n_envs, self.n_targets, 2))
        self.targets_active= numpy.zeros((self.n_envs, self.n_targets))

        self.steps = numpy.zeros((self.n_envs, ), dtype=int)

        self.reset_all()

    def __len__(self):
        return self.n_envs

    def reset_all(self):
        states = []
        infos  = []
        for e in range(self.n_envs):
            state, info = self.reset(e)
            states.append(state)
            infos.append(info)

        return numpy.array(states), infos


    def reset(self, env_id):
        # random initial robots position
        self.robots_vel[env_id] = numpy.zeros((self.n_robots, 2))
        self.robots_pos[env_id] = numpy.random.rand(self.n_robots, 2)
        
        # random targets positions
        self.targets_pos[env_id]    = numpy.random.rand(self.n_targets, 2)*0.8 + 0.1
        self.targets_active[env_id] = numpy.ones((self.n_targets, ))

        self.steps[env_id] = 0
       
        return self._make_state(env_id), None
    

    def step(self, actions):   
        actions_ = numpy.expand_dims(actions[:, 0:self.n_robots], 2)
        
        # move robots
        d_vel = numpy.zeros((self.n_envs, self.n_robots, 2))
        d_vel+= numpy.array([[[1.0,  0.0]]])*(actions_ == 0)
        d_vel+= numpy.array([[[-1.0, 0.0]]])*(actions_ == 1)
        d_vel+= numpy.array([[[0.0,  1.0]]])*(actions_ == 2)
        d_vel+= numpy.array([[[0.0, -1.0]]])*(actions_ == 3)


        self.robots_vel = self.tau*self.robots_vel + (1.0 - self.tau)*d_vel
        self.robots_pos = numpy.clip(self.robots_pos + self.robots_vel*self.dt, 0.0, 1.0)
        
        self.steps+= 1

        rewards     = numpy.zeros((self.n_envs, ), dtype=numpy.float32)
        dones       = numpy.zeros((self.n_envs, ), dtype=bool)
        infos       = []

        for e in range(self.n_envs):
            
            reward = self._reach_targets(e)

            if self.targets_active.sum() < 10**-6:
                done = True
            else:
                done = False
            
            if self.steps[e] > self.max_steps:
                done = True

            rewards[e] = reward
            dones[e]   = done
            infos.append(None)

        if self.render_mode == "human":
            self.render()

        return self._make_states(), rewards, dones, dones


    def render(self, env_id = 0):
        size  = 512
        image = numpy.zeros((size, size, 3))

        # plot targets
        for n in range(self.n_targets):
            
            x      = int(self.targets_pos[env_id][n][0]*size)
            y      = int(self.targets_pos[env_id][n][1]*size)
            active = int(self.targets_active[env_id][n])

            if active:
                cv2.circle(image, (x, y), 4, (0, 1, 0), -1)
            else:
                cv2.circle(image, (x, y), 4, (0, 1, 0), 1)

        # plot robots
        ca = numpy.array([1.0, 0.0, 0.0])
        cb = numpy.array([0.0, 0.0, 1.0])
        for n in range(self.n_robots):
            if self.n_robots > 1:
                k  = n/(self.n_robots-1)
            else:
                k  = 0
            color = (1.0 - k)*ca + k*cb

            x = int(self.robots_pos[env_id][n][0]*size)
            y = int(self.robots_pos[env_id][n][1]*size)

            cv2.circle(image, (x, y), 8, color, -1)

        image = cv2.GaussianBlur(image,(3,3),0)

        cv2.imshow("render", image)
        cv2.waitKey(1)



    def _reach_targets(self, env_id):

        dist = (numpy.expand_dims(self.robots_pos[env_id], 1) - numpy.expand_dims(self.targets_pos[env_id], 0))
        dist = ((dist**2).sum(axis=2))**0.5

        indices = numpy.argwhere(dist < 1*self.dt)

        reward = 0.0
        for idx in indices:     
            if self.targets_active[env_id][idx[1]] != 0:
                reward+= 1.0
                self.targets_active[env_id][idx[1]] = 0

        return reward

    def _make_state(self, env_id):
        result = numpy.zeros(self.observation_shape)

        # fill robot positions, and robot flag ID is 2
        result_robots = numpy.concatenate([self.robots_pos[env_id], 2*numpy.ones((self.n_robots, 1))], axis=1)
        result[0:self.n_robots] = result_robots

        # fill targets positions, target ID is 0 or 1
        result[self.n_robots:, 0:2] = self.targets_pos[env_id]
        result[self.n_robots:, 2]   = self.targets_active[env_id]


        return result.astype(numpy.float32)


    def _make_states(self):
        result = []

        for i in range(self.n_envs):
            result.append(self._make_state(i))

        return numpy.array(result).astype(numpy.float32)


if __name__ == "__main__":

    n_envs = 32
    env = RobotTargetsEnv(n_envs, render_mode="human")

    states, _ = env.reset_all()

    print("states = ", states.shape, env.observation_shape)
    
    fps = 0.0
    while True:

        action = numpy.random.randint(0, env.action_space.n, (n_envs, env.n_robots, ))

        time_start = time.time()
        states, rewards, dones, _ = env.step(action)
        time_stop = time.time()

        fps = 0.9*fps + n_envs*1.0/(time_stop - time_start)
        
        idx = numpy.where(dones)[0]
        for i in idx:
            state, _ = env.reset(i)
        
        if dones[0] or rewards[0] != 0:
            print("fps = ", round(fps, 2), states.shape)
        


    