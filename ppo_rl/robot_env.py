import cv2
import numpy

class RobotEnv:


    def __init__(self, n_targets = 10, dt = 0.02):
        self.n_targets = n_targets
        self.dt        = dt

        self.reset()


    def reset(self):
        # random initial robot position
        self.robot_a_x = numpy.random.rand()
        self.robot_a_y = numpy.random.rand()

        self.robot_b_x = numpy.random.rand()
        self.robot_b_y = numpy.random.rand()

        # random targets positions
        self.targets_x = numpy.random.rand((self.n_targets, ))
        self.targets_y = numpy.random.rand((self.n_targets, ))
        self.targets_a = numpy.ones((self.n_targets, ))

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

        return self._make_state()
    
    def _reached_targets(self):
        da = ((self.robot_a_x - self.targets_x)**2) + ((self.robot_a_y - self.targets_y)**2)
        da = da**0.5

        db = ((self.robot_b_x - self.targets_x)**2) + ((self.robot_b_y - self.targets_y)**2)
        db = db**0.5

        robot_a_reached = numpy.where(da < 2*self.dt)
        robot_b_reached = numpy.where(db < 2*self.dt)

    def _make_state(self):

        robot_pos   = numpy.array([self.robot_a_x, self.robot_a_y, self.robot_b_x, self.robot_b_y])
        targets_pos = numpy.array([self.targets_x, self.targets_y, self.targets_a])

        result = []
        result.append(robot_pos)
        result.append(targets_pos)
        
        return result