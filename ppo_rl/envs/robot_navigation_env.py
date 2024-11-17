import time
import cv2
import numpy

from gymnasium.spaces import Discrete, Box
from xRLAgents.training.ValuesLogger           import *


class RobotNavigationEnv:

    def __init__(self, n_envs, render_mode = None, n_robots = 100, n_obstacles = 50, grid_size = 16, max_steps = 1000, single_robot_revard=True):
        self.n_envs = n_envs
        self.n_robots = n_robots
        self.n_obstacles = n_obstacles
        self.grid_size = grid_size

        self.render_mode = render_mode

        self.max_steps   = max_steps

        # robot inertia
        self.tau = 0.7

        self.single_robot_revard = single_robot_revard

        self.action_space      = Discrete(5)    

        self.observation_shape = ((n_robots + n_obstacles), 7)
        self.observation_space = Box(low = numpy.zeros(self.observation_shape), 
                                     high = numpy.ones(self.observation_shape),
                                     dtype = numpy.float32)
        
        self.logger = ValuesLogger("env")
        self.logger.add("targets_per_episode", 0.0)
        self.logger.add("rr_collisions_per_episode", 0.0)
        self.logger.add("ro_collisions_per_episode", 0.0)

        self.targets_per_episode      = numpy.zeros((self.n_envs, ))
        self.targets_per_episode_curr = numpy.zeros((self.n_envs, ))

        self.rr_collisions_per_episode      = numpy.zeros((self.n_envs, ))
        self.rr_collisions_per_episode_curr = numpy.zeros((self.n_envs, ))

        self.ro_collisions_per_episode      = numpy.zeros((self.n_envs, ))
        self.ro_collisions_per_episode_curr = numpy.zeros((self.n_envs, ))

        self.reset_all()

        if self.render_mode == "human":
            cv2.namedWindow("render", cv2.WINDOW_NORMAL)

    def __len__(self):
        return self.n_envs
    
    def get_logs(self):
        return [self.logger]

    def reset_all(self):
        self._init()

        states = []
        infos  = []
        for i in range(self.n_envs):
            state, info = self.reset(i)
            states.append(state)
            infos.append(infos)

        return numpy.array(states), infos
    

    def reset(self, env_id):
        self.occupied_map[env_id] = 0

        for i in range(self.n_robots):
            self.occupied_map[env_id], y, x = self._new_random_pos(self.occupied_map[env_id])

            self.robots_velocities[env_id][i][0] = 0.0
            self.robots_velocities[env_id][i][1] = 0.0


            self.robots_positions[env_id][i][0] = y
            self.robots_positions[env_id][i][1] = x

        for i in range(self.n_robots):
            self.occupied_map[env_id], y, x = self._new_random_pos(self.occupied_map[env_id])

        
            self.targets_positions[env_id][i][0] = y
            self.targets_positions[env_id][i][1] = x

        for i in range(self.n_obstacles):
            self.occupied_map[env_id], y, x = self._new_random_pos(self.occupied_map[env_id])
          
            self.obstacles_positions[env_id][i][0] = y
            self.obstacles_positions[env_id][i][1] = x
        
        self.steps[env_id]           = 0

        # update statistics
        self.targets_per_episode[env_id]         = self.targets_per_episode_curr[env_id]
        self.rr_collisions_per_episode[env_id]   = self.rr_collisions_per_episode_curr[env_id]
        self.ro_collisions_per_episode[env_id]   = self.ro_collisions_per_episode_curr[env_id]

        # clear counters
        self.targets_per_episode_curr[env_id]       = 0
        self.rr_collisions_per_episode_curr[env_id] = 0
        self.ro_collisions_per_episode_curr[env_id] = 0


       
        return self._make_state(env_id), None
    
    def step(self, actions):
        collision_distance = (1.25/self.grid_size)
        
        actions_ = numpy.expand_dims(actions[:, 0:self.n_robots], 2)

    
        # move robots
        d_vel = numpy.zeros((self.n_envs, self.n_robots, 2))
        d_vel+= numpy.array([[[1.0,  0.0]]])*(actions_ == 0)
        d_vel+= numpy.array([[[-1.0, 0.0]]])*(actions_ == 1)
        d_vel+= numpy.array([[[0.0,  1.0]]])*(actions_ == 2)
        d_vel+= numpy.array([[[0.0, -1.0]]])*(actions_ == 3)

        dt = 1

        self.robots_velocities  = self.tau*self.robots_velocities + (1.0 - self.tau)*d_vel
        robots_positions   = numpy.clip(self.robots_positions + self.robots_velocities*dt, 0, self.grid_size-1)

        # compute and fill rewards
        rewards = numpy.zeros((self.n_envs, ))
        dones   = numpy.zeros((self.n_envs, ), dtype=bool)
        infos   = []

        # max episode steps limit
        self.steps+= 1

        idx = numpy.where(self.steps > self.max_steps)[0]
        for i in idx:
            dones[i] = True
        

        
        positive_rewards     = numpy.zeros((self.n_envs, self.n_robots))
        rr_collissions       = numpy.zeros((self.n_envs, self.n_robots))
        ro_collissions       = numpy.zeros((self.n_envs, self.n_robots))

        targets_collisions   = self._find_collisions(robots_positions, self.targets_positions, collision_distance)
        robots_collisions    = self._find_collisions(robots_positions, self.robots_positions, collision_distance)
        obstacles_collision  = self._find_collisions(robots_positions, self.obstacles_positions, collision_distance)

        
        # process reaching target
        for collision in targets_collisions:
            env_id   = collision[0]
            robot_id = collision[1]
            target_id= collision[2]

            # reward for reaching target, and generate new target
            if target_id == robot_id:
                positive_rewards[env_id][robot_id]+= 1.0
                self._new_target(env_id, target_id)

                self.targets_per_episode_curr[env_id]+= 1

        # process robot-robot collision
        for collision in robots_collisions:
            env_id    = collision[0]
            robot_id  = collision[1]
            robot_b_id= collision[2]

            # reward for robot-robot collision, remove self robot 
            if robot_id != robot_b_id:
                rr_collissions[env_id][robot_id]+= 1.0
                self.rr_collisions_per_episode_curr[env_id]+= 1.0

                self.robots_velocities[env_id][robot_id] = 0.0

        # process robot-obstacle collision
        for collision in obstacles_collision:
            env_id     = collision[0]
            robot_id   = collision[1]
            obstacle_id= collision[2]

            # reward for robot-obstacle collision
            ro_collissions[env_id][robot_id]+= 1.0
            self.ro_collisions_per_episode_curr[env_id]+= 1.0

            self.robots_velocities[env_id][robot_id] = 0.0

        # update robots position
        self.robots_positions   = numpy.clip(self.robots_positions + self.robots_velocities*dt, 0, self.grid_size-1)

        # if robot - robot collisions count > 1, generate +1
        rr_collissions = numpy.clip(rr_collissions - 1, 0.0, 1.0)

        # if robot - obstacle collisions count > 0, generate +1
        ro_collissions = numpy.clip(ro_collissions, 0.0, 1.0)

        rewards = positive_rewards - 0.01*rr_collissions - 0.01*ro_collissions

        if self.single_robot_revard:
            rewards = 0.5*(rewards[:, 0] + rewards.mean(axis=1))
        else:
            rewards = rewards.sum(axis=1)

        for env_id in range(self.n_envs):
            infos.append(None)
        

       
        if self.render_mode == "human":
            self.render(0)

        # update log
        self.logger.add("targets_per_episode", self.targets_per_episode.mean())
        self.logger.add("rr_collisions_per_episode", self.rr_collisions_per_episode.mean())
        self.logger.add("ro_collisions_per_episode", self.ro_collisions_per_episode.mean())


        return self._make_states(), rewards, dones, infos
    

    def _find_collisions(self, source, dest, collision_distance):
        d = ((numpy.expand_dims(source, 2) - numpy.expand_dims(dest, 1))**2).sum(axis=-1)
        
        indices = numpy.argwhere(d < collision_distance)

        return indices

    
    def render(self, env_id = 0):
        size  = 512
        image = numpy.zeros((size, size, 3))

        robots_positions     = (size*(self.robots_positions[env_id]/self.grid_size)).astype(int)
        targets_positions   = (size*(self.targets_positions[env_id]/self.grid_size)).astype(int)
        obstacles_positions = (size*(self.obstacles_positions[env_id]/self.grid_size)).astype(int)

        robots_positions+= int(size*0.5*1.0/self.grid_size)
        targets_positions+= int(size*0.5*1.0/self.grid_size)
        obstacles_positions+= int(size*0.5*1.0/self.grid_size)

        size = int(size*(1.0/self.grid_size)/6.0)
        # plot obstacles
        for n in range(self.n_obstacles):

            y = obstacles_positions[n][0]
            x = obstacles_positions[n][1]

            cv2.circle(image, (x, y), size, (0.5, 0.5, 0.5), -1)


        # plot robots and targets
        ca = numpy.array([1.0, 0.0, 0.0])
        cb = numpy.array([0.0, 0.0, 1.0])
        for n in range(self.n_robots):
            
            k  = n/(self.n_robots-1)
            color = (1.0 - k)*ca + k*cb

            yr = robots_positions[n][0]
            xr = robots_positions[n][1]

            cv2.circle(image, (xr, yr), size, color, -1)

            yt = targets_positions[n][0]
            xt = targets_positions[n][1]

            cv2.circle(image, (xt, yt), size, color, 2)

            if n%10 == 0:
                cv2.line(image, (xr, yr), (xt, yt), color, 2)

        image = cv2.GaussianBlur(image,(3,3),0)

        cv2.imshow("render", image)
        cv2.waitKey(1)
    
    def _make_states(self):
        states = []
        for i in range(self.n_envs):
            states.append(self._make_state(i))

        return numpy.array(states)

    def _make_state(self, env_id):

        items_count = self.n_robots + self.n_obstacles
        result      = numpy.zeros((items_count, 7), dtype=numpy.float32)

        # fill robots and corresponding targets, robot ID is 0
        result[0:self.n_robots, 0:2] = self.robots_positions[env_id]/self.grid_size
        result[0:self.n_robots, 2:4] = self.robots_velocities[env_id]
        result[0:self.n_robots, 4:6] = self.targets_positions[env_id]/self.grid_size
        result[0:self.n_robots, 6]   = 0

        
        # fill obstacles, obstacle ID is 1
        result[self.n_robots:, 0:2]  = self.obstacles_positions[env_id]/self.grid_size
        result[self.n_robots:, 6] = 2

        return result
    

    def _make_info(self, env_id):

        info = {}
        info["targets_per_episode"]       = self.targets_counter_episode[env_id]
        info["rr_collisions_per_episode"] = self.rr_collisions_episode[env_id]
        info["ro_collisions_per_episode"] = self.ro_collisions_episode[env_id]

        return info
    

    def _new_random_pos(self, occupied_map):

        y = numpy.random.randint(0, self.grid_size)
        x = numpy.random.randint(0, self.grid_size)

        while occupied_map[y][x] != 0:
            y = numpy.random.randint(0, self.grid_size)
            x = numpy.random.randint(0, self.grid_size)

        occupied_map[y][x] = 1

        return occupied_map, y, x

    def _init(self):
        self.robots_velocities    = numpy.zeros((self.n_envs, self.n_robots, 2), dtype=numpy.float32)
        self.robots_positions     = numpy.zeros((self.n_envs, self.n_robots, 2), dtype=numpy.float32)
        self.targets_positions   = numpy.zeros((self.n_envs, self.n_robots, 2), dtype=numpy.float32)
        self.obstacles_positions = numpy.zeros((self.n_envs, self.n_obstacles, 2), dtype=numpy.float32)

        self.steps           = numpy.zeros((self.n_envs, ))

        self.occupied_map = numpy.zeros((self.n_envs, self.grid_size, self.grid_size))


    def _new_target(self, env_id, target_idx):
        ty = int(self.targets_positions[env_id][target_idx][0])
        tx = int(self.targets_positions[env_id][target_idx][1])
        
        self.occupied_map[env_id][ty][tx] = 0

        self.occupied_map[env_id], y, x = self._new_random_pos(self.occupied_map[env_id])
          
        self.targets_positions[env_id][target_idx][0] = y
        self.targets_positions[env_id][target_idx][1] = x




if __name__ == "__main__":

    n_envs = 16
    env = RobotNavigationEnv(n_envs, render_mode="human")

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
            print("fps = ", round(fps, 2), states.shape, rewards[0])

        time.sleep(0.1)
    
        


    