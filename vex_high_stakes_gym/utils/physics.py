def calculate_acceleration(current_velocity, target_velocity, time_delta):
    return (target_velocity - current_velocity) / time_delta

def update_position(position, velocity, time_delta):
    return position + velocity * time_delta

def check_collision(obj1, obj2):
    return (obj1['x'] < obj2['x'] + obj2['width'] and
            obj1['x'] + obj1['width'] > obj2['x'] and
            obj1['y'] < obj2['y'] + obj2['height'] and
            obj1['y'] + obj1['height'] > obj2['y'])

class Robot:
    def __init__(self, position, velocity=(0, 0), acceleration=(0, 0)):
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration

    def update(self, time_delta):
        self.velocity = (
            self.velocity[0] + self.acceleration[0] * time_delta,
            self.velocity[1] + self.acceleration[1] * time_delta
        )
        self.position = (
            update_position(self.position[0], self.velocity[0], time_delta),
            update_position(self.position[1], self.velocity[1], time_delta)
        )

    def apply_force(self, force, time_delta):
        self.acceleration = (
            force[0] / self.mass,
            force[1] / self.mass
        )