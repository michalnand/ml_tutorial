import time
import cv2
import numpy

from gymnasium.spaces import Discrete, Box

'''
    two robots are gathering targets
    when target is reached, reward +1 is generated
    episode ends when all targets are reaches, or 1000 steps passed

    state   : vector of shape ((n_targets + 2), 3)
    actions : 8 discrtete actions
'''
class RobotEnv:

    def __init__(self, render_mode = None, n_targets = 10, dt = 0.025, max_steps = 1000):
        self.render_mode = render_mode
        self.n_targets   = n_targets
        self.dt          = dt
        self.max_steps   = max_steps
        self.steps       = 0

        self.action_space      = Discrete(8)

        self.observation_shape = ((n_targets + 2), 3)
        self.observation_space = Box(low = numpy.zeros(self.observation_shape), 
                                     high = numpy.ones(self.observation_shape),
                                     dtype = numpy.float32)

        self.reset()


    def reset(self):
        # random initial robot position
        self.robot_a_x = numpy.random.rand()
        self.robot_a_y = numpy.random.rand()

        self.robot_b_x = numpy.random.rand()
        self.robot_b_y = numpy.random.rand()

        # random targets positions
        self.targets_x = numpy.random.rand(self.n_targets)*0.8 + 0.1
        self.targets_y = numpy.random.rand(self.n_targets)*0.8 + 0.1
        self.targets_a = numpy.ones((self.n_targets, ))

        self.steps = 0

        if self.render_mode == "human":
            self.render()

        return self._make_state(), None
    

    def step(self, action):
        # move robot A
        if action == 0:
            self.robot_a_x+= self.dt
        elif action == 1:
            self.robot_a_x-= self.dt
        elif action == 2:
            self.robot_a_y+= self.dt
        elif action == 3:
            self.robot_a_y-= self.dt
        # move robot B
        elif action == 4:
            self.robot_b_x+= self.dt
        elif action == 5:
            self.robot_b_x-= self.dt
        elif action == 6:
            self.robot_b_y+= self.dt
        elif action == 7:
            self.robot_b_y-= self.dt


        self.robot_a_x = numpy.clip(self.robot_a_x, 0.0, 1.0)
        self.robot_a_y = numpy.clip(self.robot_a_y, 0.0, 1.0)

        self.robot_b_x = numpy.clip(self.robot_b_x, 0.0, 1.0)
        self.robot_b_y = numpy.clip(self.robot_b_y, 0.0, 1.0)


        reward = self._reach_targets()

        if self.targets_a.sum() < 10**-6:
            done = True
        else:
            done = False

        self.steps+= 1
        if self.steps > self.max_steps:
            done = True

        if self.render_mode == "human":
            self.render()

        return self._make_state(), reward, done, False, None

    def render(self):
        size  = 512
        image = numpy.zeros((size, size, 3))

        robot_a_x = int(self.robot_a_x*size)
        robot_a_y = int(self.robot_a_y*size)

        robot_b_x = int(self.robot_b_x*size)
        robot_b_y = int(self.robot_b_y*size)

        cv2.circle(image, (robot_a_x, robot_a_y), 15, (1, 0, 0.3), -1)
        cv2.circle(image, (robot_b_x, robot_b_y), 15, (0.3, 0, 1.0), -1)

        for n in range(self.n_targets):
            target_x = int(self.targets_x[n]*size)
            target_y = int(self.targets_y[n]*size)
            active   = int(self.targets_a[n])

            if active:
                cv2.circle(image, (target_x, target_y), 15, (0, 1, 0), -1)
            else:
                cv2.circle(image, (target_x, target_y), 15, (0, 1, 0), 1)


        cv2.imshow("render", image)
        cv2.waitKey(1)



    def _reach_targets(self):
        da = ((self.robot_a_x - self.targets_x)**2) + ((self.robot_a_y - self.targets_y)**2)
        da = da**0.5

        db = ((self.robot_b_x - self.targets_x)**2) + ((self.robot_b_y - self.targets_y)**2)
        db = db**0.5

        robot_a_reached = numpy.where(da < 2*self.dt)[0]
        robot_b_reached = numpy.where(db < 2*self.dt)[0]

        reward = 0.0
        for i in robot_a_reached:
            if self.targets_a[i] != 0:
                reward+= 1.0
                self.targets_a[i] = 0

        for i in robot_b_reached:
            if self.targets_a[i] != 0:
                reward+= 1.0
                self.targets_a[i] = 0

        return reward

    def _make_state(self):

        robots_x = numpy.array([self.robot_a_x, self.robot_b_x, -1])
        robots_y = numpy.array([self.robot_a_y, self.robot_b_y, -2])

        robots   = numpy.stack([robots_x, robots_y])
        targets  = numpy.array([self.targets_x, self.targets_y, self.targets_a]).T

        result = numpy.concatenate([robots, targets])
        result = result.astype(numpy.float32)

        return result
    

if __name__ == "__main__":

    env = RobotEnv(render_mode="human")

    state, _ = env.reset()

    print("state = ", state.shape)
    print(state)

    while True:

        action = numpy.random.randint(0, env.action_space.n)

        state, reward, done, _, _ = env.step(action)
        
        if done:
            state, _ = env.reset()

        if reward != 0:
            print(state)
            print("\n\n")


        print(state, "\n\n")



    