import numpy as np


class Tracker:
    def __init__(self, cam_id):
        self.cam_id = cam_id
        self.max_travel = 180
        self.tracked = False
        self.ttl = 3
        self.missed = 0
        self.last_pos = None

    def update(self, pos):

        if pos is None:
            self.missed += 1
            if self.missed >= self.ttl:
                self.tracked = False
            return None
        else:
            x, y = pos
            if not self.tracked:
                self.last_pos = (x, y)
                self.missed = 0
                self.tracked = True
            else:
                maxdist = self.max_travel * (self.missed + 1)
                dist = np.linalg.norm(np.array(pos) - np.array(self.last_pos))
                if dist > maxdist:
                    print(f"{self.cam_id} TOO FAR {dist}")
                    self.missed += 1
                    if self.missed >= self.ttl:
                        print(f"{self.cam_id} TRACK LOST")
                        self.tracked = False
                    return None
                else:
                    self.last_pos = (x, y)
                    self.missed = 0
                    return pos
